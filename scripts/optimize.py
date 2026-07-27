import math
import os
import random
import optuna
import hydra
from omegaconf import DictConfig, OmegaConf
import numpy as np
import torch
import wandb

from src.data.datamodule import CoastalDataModule
from src.models.factory import SegmentationModelFactory
from src.models.losses import CoastalCompositeLoss
from src.engine.trainer import SpectralTrainer
from src.utils.stratified_metrics import StratifiedWaterAccumulator
from src.utils.perf import apply_perf_flags, build_adamw, maybe_compile
from dotenv import load_dotenv
import logging
from optuna.storages import RDBStorage

logger = logging.getLogger(__name__)

load_dotenv()

def objective(trial: optuna.Trial, cfg: DictConfig):
    # SAME fixed seed for every trial (cfg.seed=42) so trials differ only by the
    # sampled hyperparameters, not by uncontrolled init / shuffle noise. We seed
    # random/numpy/torch/cuda but deliberately do NOT enable cudnn.deterministic
    # / use_deterministic_algorithms: the requirement is reproducible seeding,
    # not bitwise determinism, and forcing deterministic cuDNN kernels carries a
    # real throughput cost across a long sweep.
    seed = int(cfg.seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Suggest hyperparameters
    #arch = trial.suggest_categorical("arch", ["unet", "upernet", "segformer"])
    base_lr = trial.suggest_float("base_learning_rate", 1e-5, 1e-2, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-1, log=True)
    label_smoothing = trial.suggest_float("label_smoothing", 0.0, 0.2)
    dice_weight = trial.suggest_float("dice_weight", 0.0, 1.0)
    
    #cfg.model.arch = arch
    cfg.trainer.base_learning_rate = base_lr
    cfg.trainer.weight_decay = weight_decay
    cfg.model.label_smoothing = label_smoothing
    cfg.model.dice_weight = dice_weight
    cfg.model.ce_weight = 1.0 - dice_weight
    

    run = wandb.init(
        project=cfg.project_name,
        group=cfg.study_name,
        job_type="trial",
        name=f"trial_{trial.number}",
        reinit=True,
        config=OmegaConf.to_container(cfg, resolve=True)
    )
    
    # --- HYPERPARAMETER LOGGING BLOCK ---
    logger.info("="*40)
    logger.info(f"STARTING TRIAL {trial.number}")
    logger.info("="*40)
    logger.info(f"Architecture:      {cfg.model.arch}")
    logger.info(f"Encoder:           {cfg.model.encoder_name}")
    logger.info(f"Base LR:           {base_lr:.2e}")
    logger.info(f"Weight Decay:      {weight_decay:.2e}")
    logger.info(f"Label Smoothing:   {label_smoothing:.4f}")
    logger.info(f"Dice Weight:       {dice_weight:.4f}")
    logger.info("="*40)
    # -------------------------

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    
    datamodule = CoastalDataModule(
        root_dir=cfg.data.memmap_root,
        H=cfg.data.H, W=cfg.data.W,
        batch_size=cfg.data.get("batch_size", 256),
        val_batch_size=cfg.data.get("val_batch_size", 256),
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.get("pin_memory", True),
        persistent_workers=cfg.data.get("persistent_workers", True),
        augment=cfg.data.augment,
        aug_params=cfg.data.get("aug", {}),
        seed=cfg.seed,
        dtype=cfg.data.get("dtype", "float32"),
    )
    datamodule.setup()

    _encoder_kwargs = cfg.model.get("encoder_kwargs", None)
    _encoder_kwargs = OmegaConf.to_container(_encoder_kwargs, resolve=True) if _encoder_kwargs else {}
    model = SegmentationModelFactory.build(
        arch=cfg.model.arch,
        encoder_name=cfg.model.encoder_name,
        encoder_weights=cfg.model.encoder_weights,
        in_channels=cfg.model.in_channels,
        classes=cfg.model.num_classes,
        **_encoder_kwargs,
    )
    # Move to device before the optimizer is built (fused AdamW needs CUDA
    # params at construction) and before the optional torch.compile wrap.
    model = model.to(device)
    model = maybe_compile(model, cfg.trainer.get("perf", None))

    loss_fn = CoastalCompositeLoss(
        ce_weight=cfg.model.ce_weight,
        dice_weight=cfg.model.dice_weight,
        label_smoothing=cfg.model.label_smoothing
    )

    optimizer = build_adamw(
        model.parameters(),
        lr=cfg.trainer.base_learning_rate,
        weight_decay=cfg.trainer.weight_decay,
        perf_cfg=cfg.trainer.get("perf", None),
    )
    
    train_dl = datamodule.train_dataloader()
    val_dl = datamodule.val_dataloader()

    # --- DATASET LOGGING BLOCK ---
    logger.info("="*40)
    logger.info("DATASET CONFIGURATION")
    logger.info(f"Train Samples: {len(train_dl.dataset):,}")
    logger.info(f"Val Samples:   {len(val_dl.dataset):,}")
    logger.info(f"Batch Size:    {cfg.data.batch_size}")
    logger.info(f"Train Steps:   {len(train_dl):,} per epoch")
    logger.info("="*40)
    # --------------------------------------
    
    if wandb.run is not None:
        wandb.config.update({
            "data_lineage/train_samples": len(train_dl.dataset),
            "data_lineage/val_samples": len(val_dl.dataset),
            "data_lineage/steps_per_epoch": len(train_dl)
        })

    max_steps = cfg.trainer.max_steps
    warmup_steps = cfg.trainer.warmup_steps
    val_check_interval = cfg.trainer.val_check_interval
    
    scheduler_warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1e-6, end_factor=1.0, total_iters=max(1, warmup_steps)
    )
    scheduler_decay = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, max_steps - warmup_steps), eta_min=1e-6
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[scheduler_warmup, scheduler_decay], milestones=[max(1, warmup_steps)]
    )

    # --- Optional stratified / pair-macro objective ---
    # When data.strata_index_path points at hpo_val_pair_strata.npz, the trainer
    # accumulates pair-macro water IoU on mixed chips (the ladder's re-ranking
    # metric) and the objective becomes that instead of pooled val mIoU. The
    # N-assert fails fast if the strata index and the val memmap disagree.
    strata_acc = None
    strata_path = cfg.data.get("strata_index_path", None)
    if strata_path:
        strata_acc = StratifiedWaterAccumulator.from_npz(
            strata_path, device, expected_n=len(datamodule.val_ds)
        )
        logger.info(
            f"Stratified objective ENABLED: N={strata_acc.N} pairs={strata_acc.P} "
            f"eligible={int(strata_acc.eligible.sum())}"
        )

    trainer = SpectralTrainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=loss_fn,
        device=device,
        use_amp=cfg.trainer.mixed_precision,
        precision=cfg.trainer.get("precision", "fp16"),
        gradient_clip_val=cfg.trainer.gradient_clip_val,
        num_classes=cfg.model.num_classes,
        accumulate_grad_batches=cfg.trainer.get("accumulate_grad_batches", 1),
        log_every_n_steps=cfg.trainer.get("log_every_n_steps", 1),
        strata_accumulator=strata_acc,
    )

    best_iou, _, final_metrics = trainer.fit(
        train_dataloader=train_dl,
        val_dataloader=val_dl,
        max_steps=max_steps,
        val_check_interval=val_check_interval,
        trial=trial
    )

    if strata_acc is not None:
        # Objective = FINAL val check's pair-macro water IoU (non-finite -> 0.0,
        # matching the trainer's guard). Guardrails + pooled diagnostic recorded
        # as user_attrs for post-hoc inspection, NOT folded into the objective.
        objective_value = final_metrics.get("strat/pair_macro_water_iou", float("nan"))
        if not math.isfinite(objective_value):
            objective_value = 0.0
        trial.set_user_attr("pair_macro_water_iou", objective_value)
        trial.set_user_attr("n_pairs_used", final_metrics.get("strat/n_pairs_used", float("nan")))
        trial.set_user_attr("pure_land_fpr", final_metrics.get("strat/pure_land_fpr", float("nan")))
        trial.set_user_attr("pure_water_fnr", final_metrics.get("strat/pure_water_fnr", float("nan")))
        trial.set_user_attr("mixed_precision", final_metrics.get("strat/mixed_precision", float("nan")))
        trial.set_user_attr("mixed_recall", final_metrics.get("strat/mixed_recall", float("nan")))
        trial.set_user_attr("pooled_best_miou", best_iou)
    else:
        objective_value = best_iou

    datamodule.teardown()
    run.finish()
    return objective_value

@hydra.main(version_base="1.3", config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    os.makedirs(cfg.output_dir, exist_ok=True)
    apply_perf_flags(cfg.trainer.get("perf", None))


    if cfg.optuna_storage:
        db_path = cfg.optuna_storage
        print(f"Connecting to remote Postgres database...")
        # Replace your current storage string with this configuration
        storage = RDBStorage(
            url=db_path,
            engine_kwargs={
                "pool_size": 20,           # Allow more concurrent connections
                "max_overflow": 0,
                "pool_recycle": 300,       # Reset connections every 5 minutes
                "pool_pre_ping": True,     # CRITICAL: Check if connection is alive before use
                "connect_args": {
                    "connect_timeout": 10
                }
            }
        )
    else:
        db_path = f"sqlite:///{cfg.output_dir}/optuna_sweep.db"
        storage = db_path
        print(f"No remote DB found. Falling back to local SQLite: {db_path}")
    
    # min_resource is in global steps; default 800 = the shipped HPO protocol.
    # See configs/config.yaml for why it must not drop below warmup_steps=500
    # (a rung at 400 would rank trials mid-warmup, biasing against high-LR
    # configs while base LR is itself a swept parameter).
    pruner = optuna.pruners.HyperbandPruner(min_resource=cfg.get("pruner_min_resource", 800))

    study = optuna.create_study(
        direction="maximize",
        pruner=pruner,
        storage=storage,
        study_name=cfg.study_name,
        load_if_exists=True
    )
    
    # default to 60 if n_trials is not provided
    study.optimize(lambda trial: objective(trial, cfg), n_trials=cfg.get("n_trials", 60))
    logger.info(f"Optimization Complete. Best params: {study.best_params}")

if __name__ == "__main__":
    main()
