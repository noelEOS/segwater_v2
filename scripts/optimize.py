import os
import optuna
import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import wandb

from src.data.datamodule import CoastalDataModule
from src.models.factory import SegmentationModelFactory
from src.models.losses import CoastalCompositeLoss
from src.engine.trainer import SpectralTrainer
from dotenv import load_dotenv 
import logging


logger = logging.getLogger(__name__)

load_dotenv()

def objective(trial: optuna.Trial, cfg: DictConfig):
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
        augment=cfg.data.augment
    )
    datamodule.setup()
    
    model = SegmentationModelFactory.build(
        arch=cfg.model.arch,
        encoder_name=cfg.model.encoder_name,
        encoder_weights=cfg.model.encoder_weights,
        in_channels=cfg.model.in_channels,
        classes=cfg.model.num_classes
    )
    
    loss_fn = CoastalCompositeLoss(
        ce_weight=cfg.model.ce_weight,
        dice_weight=cfg.model.dice_weight,
        label_smoothing=cfg.model.label_smoothing
    )
    
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=cfg.trainer.base_learning_rate, 
        weight_decay=cfg.trainer.weight_decay
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
    
    trainer = SpectralTrainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=loss_fn,
        device=device,
        use_amp=cfg.trainer.mixed_precision,
        precision=cfg.trainer.get("precision", "fp16"),
        gradient_clip_val=cfg.trainer.gradient_clip_val,
        num_classes=cfg.model.num_classes
    )
    
    best_iou = trainer.fit(
        train_dataloader=train_dl,
        val_dataloader=val_dl,
        max_steps=max_steps,
        val_check_interval=val_check_interval,
        trial=trial
    )
            
    datamodule.teardown()
    run.finish()
    return best_iou

@hydra.main(version_base="1.3", config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    os.makedirs(cfg.output_dir, exist_ok=True)


    if cfg.optuna_storage:
        db_path = cfg.optuna_storage
        print(f"Connecting to remote Postgres database...")
    else:
        db_path = f"sqlite:///{cfg.output_dir}/optuna_sweep.db"
        print(f"No remote DB found. Falling back to local SQLite: {db_path}")
    
    pruner = optuna.pruners.HyperbandPruner(min_resource=800)

    study = optuna.create_study(
        direction="maximize",
        pruner=pruner,
        storage=db_path,
        study_name=cfg.study_name,
        load_if_exists=True
    )
    
    # default to 60 if n_trials is not provided
    study.optimize(lambda trial: objective(trial, cfg), n_trials=cfg.get("n_trials", 60))
    logger.info(f"Optimization Complete. Best params: {study.best_params}")

if __name__ == "__main__":
    main()
