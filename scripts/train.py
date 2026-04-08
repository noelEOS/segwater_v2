import os
import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import wandb

from src.data.datamodule import CoastalDataModule
from src.models.factory import SegmentationModelFactory
from src.models.losses import CoastalCompositeLoss
from src.engine.trainer import SpectralTrainer

@hydra.main(version_base="1.3", config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    torch.manual_seed(cfg.seed)
    
    wandb.init(
        project=cfg.project_name,
        job_type="train",
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
    
    # Stage 2 Budgeting
    steps_per_epoch = len(train_dl)
    epochs = 10
    total_steps = epochs * steps_per_epoch
    warmup_steps = 1000
    
    scheduler_warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1e-6, end_factor=1.0, total_iters=max(1, warmup_steps)
    )
    scheduler_decay = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, total_steps - warmup_steps), eta_min=1e-6
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
        gradient_clip_val=cfg.trainer.gradient_clip_val,
        num_classes=cfg.model.num_classes
    )
    
    os.makedirs(cfg.output_dir, exist_ok=True)
    
    trainer.fit(
        train_dataloader=train_dl,
        val_dataloader=val_dl,
        max_steps=total_steps,
        val_check_interval=max(1, steps_per_epoch),  # Check at the end of every epoch
        save_dir=cfg.output_dir
    )
            
    datamodule.teardown()
    wandb.finish()

if __name__ == "__main__":
    main()
