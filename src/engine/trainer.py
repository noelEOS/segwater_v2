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

        # Support CUDA/CPU depending on available device
        if device.type == "cuda":
            self.scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
        else:
            self.scaler = torch.amp.GradScaler("cpu", enabled=use_amp)
        
        self.train_metrics = SegmentationMetrics(num_classes=num_classes, ignore_index=ignore_index, device=device.type)
        self.val_metrics = SegmentationMetrics(num_classes=num_classes, ignore_index=ignore_index, device=device.type)

    def train_epoch(self, dataloader) -> Dict[str, Any]:
        self.model.train()
        self.train_metrics.reset()
        
        total_loss = 0.0
        num_batches = 0
        
        for batch in dataloader:
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
            
        metrics_dict = self.train_metrics.compute()
        metrics_dict["loss"] = total_loss / max(1, num_batches)
        
        return metrics_dict
        
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
