import torch
import torch.nn as nn
from typing import Dict, Any

from src.utils.metrics import SegmentationMetrics

class SpectralTrainer:
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        loss_fn: nn.Module,
        device: torch.device,
        use_amp: bool = True,
        gradient_clip_val: float = 1.0,
        num_classes: int = 2,
        ignore_index: int = 255
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_fn.to(device)
        self.device = device
        self.use_amp = use_amp
        self.gradient_clip_val = gradient_clip_val

        if device.type == "cuda":
            self.scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
        else:
            self.scaler = torch.amp.GradScaler("cpu", enabled=use_amp)
        
        self.train_metrics = SegmentationMetrics(num_classes=num_classes, ignore_index=ignore_index, device=device.type)
        self.val_metrics = SegmentationMetrics(num_classes=num_classes, ignore_index=ignore_index, device=device.type)

    def fit(
        self,
        train_dataloader,
        val_dataloader,
        max_steps: int,
        val_check_interval: int,
        trial=None,
        save_dir: str = None
    ) -> float:
        import optuna
        import wandb
        import os
        
        global_step = 0
        best_iou = -1.0
        
        train_iterator = iter(train_dataloader)
        
        while global_step < max_steps:
            self.model.train()
            self.train_metrics.reset()
            total_loss = 0.0
            num_batches = 0
            
            while num_batches < val_check_interval and global_step < max_steps:
                try:
                    batch = next(train_iterator)
                except StopIteration:
                    train_iterator = iter(train_dataloader)
                    batch = next(train_iterator)
                    
                x = batch["pixel_values"].to(self.device, non_blocking=True)
                y = batch["labels"].to(self.device, non_blocking=True)
                
                self.optimizer.zero_grad(set_to_none=True)
                
                with torch.autocast(device_type=self.device.type, enabled=self.use_amp):
                    logits = self.model(x)
                    loss_dict = self.loss_fn(logits, y)
                    loss = loss_dict["loss"]
                    
                self.scaler.scale(loss).backward()
                
                if self.gradient_clip_val > 0.0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_val)
                    
                self.scaler.step(self.optimizer)
                self.scaler.update()
                
                if self.scheduler is not None:
                    self.scheduler.step()
                    
                preds = torch.argmax(logits, dim=1)
                self.train_metrics.update(preds, y)
                total_loss += loss.item()
                num_batches += 1
                global_step += 1
                
            train_metrics_dict = self.train_metrics.compute()
            train_loss = total_loss / max(1, num_batches)
            
            val_metrics_dict = self.val_epoch(val_dataloader)
            val_miou = val_metrics_dict["mIoU"]
            
            if wandb.run is not None:
                wandb.log({
                    "global_step": global_step,
                    "train/loss": train_loss,
                    **{f"train/{k}": v for k, v in train_metrics_dict.items()},
                    **{f"val/{k}": v for k, v in val_metrics_dict.items()}
                })
                
            if val_miou > best_iou:
                best_iou = val_miou
                if save_dir is not None:
                    ckpt_path = os.path.join(save_dir, "best_model.pth")
                    torch.save({
                        "step": global_step,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "best_iou": best_iou
                    }, ckpt_path)
                    print(f"Step {global_step}: New best mIoU {best_iou:.4f} saved.")
                    
            if trial is not None:
                trial.report(val_miou, step=global_step)
                if trial.should_prune():
                    raise optuna.TrialPruned(f"Pruned at step {global_step} with mIoU {val_miou:.4f}")
                    
        return best_iou
        
    @torch.no_grad()
    def val_epoch(self, dataloader) -> Dict[str, Any]:
        self.model.eval()
        self.val_metrics.reset()
        
        total_loss = 0.0
        num_batches = 0
        
        for batch in dataloader:
            x = batch["pixel_values"].to(self.device, non_blocking=True)
            y = batch["labels"].to(self.device, non_blocking=True)
            
            with torch.autocast(device_type=self.device.type, enabled=self.use_amp):
                logits = self.model(x)
                loss_dict = self.loss_fn(logits, y)
                loss = loss_dict["loss"]
                
            preds = torch.argmax(logits, dim=1)
            self.val_metrics.update(preds, y)
            
            total_loss += loss.item()
            num_batches += 1
            
        metrics_dict = self.val_metrics.compute()
        metrics_dict["loss"] = total_loss / max(1, num_batches)
        
        return metrics_dict
