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

def objective(trial: optuna.Trial, cfg: DictConfig):
    # Suggest hyperparameters
    arch = trial.suggest_categorical("arch", ["unet", "upernet", "segformer"])
    base_lr = trial.suggest_float("base_lr", 1e-5, 1e-2, log=True)
    dice_weight = trial.suggest_float("dice_weight", 0.0, 1.0)
    
    cfg.model.arch = arch
    cfg.trainer.base_learning_rate = base_lr
    cfg.model.dice_weight = dice_weight
    cfg.model.ce_weight = 1.0 - dice_weight
    
    run = wandb.init(
        project=cfg.project_name,
        group="optuna-sweep",
        job_type="trial",
        name=f"trial_{trial.number}",
        reinit=True,
        config=OmegaConf.to_container(cfg, resolve=True)
    )
    
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    
    datamodule = CoastalDataModule(
        root_dir=cfg.data.memmap_root,
        H=cfg.data.H, W=cfg.data.W,
        batch_size=cfg.data.batch_size,
        val_batch_size=cfg.data.val_batch_size,
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
    
    trainer = SpectralTrainer(
        model=model,
        optimizer=optimizer,
        scheduler=None, # Simplifying scheduler for HPO
        loss_fn=loss_fn,
        device=device,
        use_amp=cfg.trainer.mixed_precision,
        gradient_clip_val=cfg.trainer.gradient_clip_val,
        num_classes=cfg.model.num_classes
    )
    
    best_iou = -1.0
    for epoch in range(cfg.trainer.epochs):
        _ = trainer.train_epoch(train_dl)
        val_metrics = trainer.val_epoch(val_dl)
        
        miou = val_metrics["mIoU"]
        if miou > best_iou:
            best_iou = miou
            
        wandb.log({"epoch": epoch, "trial_mIoU": miou})
        
        trial.report(miou, step=epoch)
        if trial.should_prune():
            datamodule.teardown()
            run.finish()
            raise optuna.TrialPruned(f"Pruned at epoch {epoch}")
            
    datamodule.teardown()
    run.finish()
    return best_iou

@hydra.main(version_base="1.3", config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    os.makedirs(cfg.output_dir, exist_ok=True)
    db_path = f"sqlite:///{cfg.output_dir}/optuna_sweep.db"
    
    pruner = optuna.pruners.HyperbandPruner()
    study = optuna.create_study(
        direction="maximize",
        pruner=pruner,
        storage=db_path,
        study_name="coastal_water_hpo"
    )
    
    study.optimize(lambda trial: objective(trial, cfg), n_trials=10)
    print("Optimization Complete. Best params:", study.best_params)

if __name__ == "__main__":
    main()
