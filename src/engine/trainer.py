import math

import torch
import torch.nn as nn
import optuna
from typing import Dict, Any

from tqdm import tqdm

from src.utils.metrics import SegmentationMetrics
import logging

logger = logging.getLogger(__name__)


def compute_total_steps(len_train_dl: int, epochs: int, accumulate_grad_batches: int) -> int:
    """Optimizer-step budget for a stage-2 run.

    ``len_train_dl`` counts micro-batches per epoch, but ``max_steps`` and the
    LR scheduler count OPTIMIZER steps, and SpectralTrainer takes one optimizer
    step per ``accumulate_grad_batches`` micro-batches. So the optimizer-step
    budget per epoch is ``len_train_dl // accum``.

    Invariants this guarantees:
      * accum == 1 reproduces the previous ``epochs * len_train_dl`` exactly.
      * For a fixed effective batch (bs * accum) and a dataset whose loader
        length divides cleanly, ``accum=k, bs=B`` yields the same budget as
        ``accum=1, bs=k*B`` -- hence an identical LR schedule. (With
        drop_last=False and a non-divisible loader length the two can differ by
        up to ~epochs steps; this is accepted as negligible.)
    """
    accum = max(1, int(accumulate_grad_batches))
    steps_per_epoch = len_train_dl // accum
    return epochs * steps_per_epoch


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
        ignore_index: int = 255,
        precision: str = "fp16",
        arch: str = "arch",
        encoder: str = "encoder",
        seed: int = 42,
        accumulate_grad_batches: int = 1,
    ):
        self.model = model.to(device)
        self.arch = arch
        self.encoder = encoder
        self.seed = seed
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_fn.to(device)
        self.device = device
        self.use_amp = use_amp
        self.gradient_clip_val = gradient_clip_val
        self.accumulate_grad_batches = max(1, int(accumulate_grad_batches))

        # Resolve the torch dtype
        self.amp_dtype = torch.bfloat16 if precision == "bf16" else torch.float16

        scaler_enabled = use_amp and precision == "fp16"

        logger.info("="*40)
        logger.info("TRAINER CONFIGURATION SETUP")
        logger.info(f"Device:            {self.device}")
        logger.info(f"Mixed Precision:   {self.use_amp}")
        logger.info(f"AMP Dtype:         {self.amp_dtype}")
        logger.info(f"Gradient Clipping: {self.gradient_clip_val}")
        logger.info(f"Grad Accumulation: {self.accumulate_grad_batches}")
        logger.info(f"Scaler Enabled:    {scaler_enabled}")
        logger.info("="*40)

        if device.type == "cuda":
            self.scaler = torch.amp.GradScaler("cuda", enabled=scaler_enabled)
        else:
            self.scaler = torch.amp.GradScaler("cpu", enabled=scaler_enabled)
        
        self.train_metrics = SegmentationMetrics(num_classes=num_classes, ignore_index=ignore_index, device=device.type)
        self.val_metrics = SegmentationMetrics(num_classes=num_classes, ignore_index=ignore_index, device=device.type)

    def fit(
            self,
            train_dataloader,
            val_dataloader,
            max_steps: int,
            val_check_interval: int,
            trial=None,
            save_dir: str = None,
            keep_top_k: int = 3
        ):
            #import optuna
            import wandb
            import os
            
            global_step = 0
            top_k_checkpoints = [] # List to track (val_miou, ckpt_path)
            best_val_miou = 0.0

            train_iterator = iter(train_dataloader)
            
            # max_steps / val_check_interval are counted in OPTIMIZER steps. Each
            # optimizer step consumes `accumulate_grad_batches` micro-batches, so
            # an accumulated run sees the same data / does the same number of
            # weight updates / follows the same LR schedule as a non-accumulated
            # run with the same max_steps and the equivalent (accum x) batch size.
            # With accumulate_grad_batches == 1 this is identical to the prior
            # behaviour (one optimizer step per micro-batch).
            accum = self.accumulate_grad_batches

            while global_step < max_steps:
                self.model.train()
                self.train_metrics.reset()
                total_loss = 0.0
                num_opt_steps = 0

                with tqdm(total=val_check_interval, desc=f"Train [Steps {global_step} - {global_step+val_check_interval}]", leave=False) as pbar:
                    while num_opt_steps < val_check_interval and global_step < max_steps:
                        # Accumulate gradients over `accum` micro-batches, then take
                        # a single optimizer step (== one global_step).
                        self.optimizer.zero_grad(set_to_none=True)
                        step_loss = 0.0

                        for _ in range(accum):
                            try:
                                batch = next(train_iterator)
                            except StopIteration:
                                train_iterator = iter(train_dataloader)
                                batch = next(train_iterator)

                            x = batch["pixel_values"].to(self.device, non_blocking=True)
                            y = batch["labels"].to(self.device, non_blocking=True)

                            with torch.autocast(device_type=self.device.type, enabled=self.use_amp, dtype=self.amp_dtype):
                                logits = self.model(x)
                                loss_dict = self.loss_fn(logits, y)
                                loss = loss_dict["loss"]

                            # Divide by accum so summed gradients average over the
                            # window (matches a single large-batch step).
                            self.scaler.scale(loss / accum).backward()

                            preds = torch.argmax(logits, dim=1)
                            self.train_metrics.update(preds, y)
                            step_loss += loss.item() / accum

                        if self.gradient_clip_val > 0.0:
                            self.scaler.unscale_(self.optimizer)
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_val)

                        self.scaler.step(self.optimizer)
                        self.scaler.update()

                        if self.scheduler is not None:
                            self.scheduler.step()

                        total_loss += step_loss

                        current_lr = self.optimizer.param_groups[0]['lr']
                        if wandb.run is not None:
                            wandb.log({
                                "global_step": global_step,
                                "train/step_loss": step_loss,
                                "train/learning_rate": current_lr
                            })

                        num_opt_steps += 1
                        global_step += 1

                        pbar.update(1)
                    
                train_metrics_dict = self.train_metrics.compute()
                train_loss = total_loss / max(1, num_opt_steps)
                
                val_metrics_dict = self.val_epoch(val_dataloader)
                val_miou = val_metrics_dict["mIoU"]

                # Guard against a non-finite mIoU. macro JaccardIndex returns NaN
                # when a class has zero union over the whole val set (e.g. a
                # degenerate trial that predicts only background early on). NaN
                # propagates into max() and, worse, into Optuna's RDB commit
                # (trial.report / the returned objective), which raises a
                # StorageInternalError and kills the whole study. Treat a
                # non-finite score as the worst possible value so the trial is
                # simply pruned/ranked last instead.
                if not math.isfinite(val_miou):
                    logger.warning(f"Non-finite val mIoU ({val_miou}) at step {global_step}; treating as 0.0")
                    val_miou = 0.0

                best_val_miou = max(best_val_miou, val_miou)
                
                if wandb.run is not None:
                    wandb.log({
                        "global_step": global_step,
                        "train/interval_loss": train_loss,
                        **{f"train/{k}": v for k, v in train_metrics_dict.items()},
                        **{f"val/{k}": v for k, v in val_metrics_dict.items()}
                    })
                    
                # --- TOP-K CHECKPOINTING LOGIC ---
                if save_dir is not None:
                    # If we have less than K checkpoints, or the current mIoU is better than the worst in our top K
                    if len(top_k_checkpoints) < keep_top_k or val_miou > top_k_checkpoints[0][0]:
                        ckpt_name = f"{self.arch}_{self.encoder}_s{self.seed}_step{global_step}_miou{val_miou:.4f}.pth"
                        ckpt_path = os.path.join(save_dir, ckpt_name)
                        
                        torch.save({
                            "step": global_step,
                            "model_state_dict": self.model.state_dict(),
                            "optimizer_state_dict": self.optimizer.state_dict(),
                            "val_miou": val_miou
                        }, ckpt_path)
                        
                        print(f"Step {global_step}: Saved new Top-{keep_top_k} checkpoint -> {ckpt_name}")
                        
                        # Add to list and sort by mIoU (ascending, so index 0 is the lowest mIoU)
                        top_k_checkpoints.append((val_miou, ckpt_path))
                        top_k_checkpoints.sort(key=lambda x: x[0])
                        
                        # If we exceeded our keep limit, remove the lowest one from disk and the list
                        if len(top_k_checkpoints) > keep_top_k:
                            removed_miou, removed_path = top_k_checkpoints.pop(0)
                            if os.path.exists(removed_path):
                                os.remove(removed_path)
                                print(f"Removed older checkpoint -> {os.path.basename(removed_path)} (mIoU: {removed_miou:.4f})")
                        
                if trial is not None:
                    #import optuna
                    trial.report(val_miou, step=global_step)
                    if trial.should_prune():
                        raise optuna.TrialPruned(f"Pruned at step {global_step} with mIoU {val_miou:.4f}")
                        
            # Return the path to the best checkpoint (the last item in our sorted list)
            best_ckpt_path = top_k_checkpoints[-1][1] if top_k_checkpoints else None
            return best_val_miou, best_ckpt_path

    @torch.no_grad()
    def val_epoch(self, dataloader) -> Dict[str, Any]:
        self.model.eval()
        self.val_metrics.reset()
        
        total_loss = 0.0
        num_batches = 0
        
        for batch in tqdm(dataloader, desc="Validating", leave=False):
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
        
    @torch.no_grad()
    def test(self, dataloader) -> Dict[str, Any]:
        self.model.eval()
        self.val_metrics.reset()
        
        total_loss = 0.0
        num_batches = 0
        
        for batch in tqdm(dataloader, desc="Testing", leave=False):
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

    @torch.no_grad()
    def predict(self, dataloader):
        self.model.eval()
        for batch in tqdm(dataloader, desc="Predicting", leave=False):
            x = batch["pixel_values"].to(self.device, non_blocking=True)
            y = batch["labels"].to(self.device, non_blocking=True)
            
            with torch.autocast(device_type=self.device.type, enabled=self.use_amp):
                logits = self.model(x)
                
            preds = torch.argmax(logits, dim=1)
            
            yield {"inputs": x.cpu(), "predictions": preds.cpu(), "labels": y.cpu()}
