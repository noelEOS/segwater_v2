import math

import torch
import torch.nn as nn
import optuna
from typing import Dict, Any

from tqdm import tqdm

from src.utils.metrics import SegmentationMetrics
from src.utils.perf import unwrap_compiled
import logging

logger = logging.getLogger(__name__)

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
        log_every_n_steps: int = 1,
        strata_accumulator=None,
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
        self.log_every_n_steps = max(1, int(log_every_n_steps))
        # Opt-in stratified/pair-macro accumulator (HPO objective). None keeps
        # the train.py path byte-identical: no extra val work, pooled-mIoU return.
        self.strata_acc = strata_accumulator

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
        
        # Train metrics are diagnostic-only: keep just the confusion matrix and
        # derive mIoU/F1 from it at the logging interval, instead of updating
        # three torchmetrics objects every micro-batch.
        self.train_metrics = SegmentationMetrics(num_classes=num_classes, ignore_index=ignore_index, device=device.type, cm_only=True)
        self.val_metrics = SegmentationMetrics(num_classes=num_classes, ignore_index=ignore_index, device=device.type)

    def fit(
            self,
            train_dataloader,
            val_dataloader,
            max_steps: int,
            val_check_interval: int,
            trial=None,
            save_dir: str = None,
            keep_top_k: int = 3,
            save_last: bool = False,
            snapshot_last_frac: float = 0.0,
            snapshot_every_n_vals: int = 0,
        ):
            #import optuna
            import wandb
            import os

            if wandb.run is not None:
                # Plot every series against global_step by default. With
                # log_every_n_steps>1 wandb's internal _step counts log CALLS,
                # which misaligns sparse new runs vs legacy per-step runs.
                wandb.define_metric("global_step")
                wandb.define_metric("*", step_metric="global_step")

            global_step = 0
            n_vals = 0  # completed validations, for the snapshot cadence
            val_miou = float("nan")  # defined even if the loop body never runs
            top_k_checkpoints = [] # List to track (val_miou, ckpt_path)
            best_val_miou = 0.0
            # Full metrics dict of the LAST val check; the caller reads the
            # stratified objective from it (empty until the first val runs).
            final_val_metrics_dict = {}

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
                # Losses accumulate on-device; .item() (a hard GPU->CPU sync
                # that defeats async execution) is only called at the logging
                # interval, not per micro-batch.
                total_loss = torch.zeros((), device=self.device)
                num_opt_steps = 0

                with tqdm(total=val_check_interval, desc=f"Train [Steps {global_step} - {global_step+val_check_interval}]", leave=False) as pbar:
                    while num_opt_steps < val_check_interval and global_step < max_steps:
                        # Accumulate gradients over `accum` micro-batches, then take
                        # a single optimizer step (== one global_step).
                        self.optimizer.zero_grad(set_to_none=True)
                        step_loss = torch.zeros((), device=self.device)

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
                            step_loss += loss.detach()

                        if self.gradient_clip_val > 0.0:
                            self.scaler.unscale_(self.optimizer)
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_val)

                        self.scaler.step(self.optimizer)
                        self.scaler.update()

                        if self.scheduler is not None:
                            self.scheduler.step()

                        step_loss = step_loss / accum
                        total_loss += step_loss

                        if wandb.run is not None and global_step % self.log_every_n_steps == 0:
                            wandb.log({
                                "global_step": global_step,
                                "train/step_loss": step_loss.item(),
                                "train/learning_rate": self.optimizer.param_groups[0]['lr']
                            })

                        num_opt_steps += 1
                        global_step += 1

                        pbar.update(1)
                    
                train_metrics_dict = self.train_metrics.compute()
                train_loss = (total_loss / max(1, num_opt_steps)).item()
                
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

                final_val_metrics_dict = val_metrics_dict

                # The objective / pruner signal is the pair-macro water IoU when a
                # strata accumulator is wired (HPO), else pooled val mIoU (train.py
                # path unchanged). Same non-finite -> 0.0 guard as val_miou above,
                # so an early all-NaN pair-macro ranks last / prunes cleanly.
                if self.strata_acc is not None:
                    objective_metric = val_metrics_dict.get("strat/pair_macro_water_iou", float("nan"))
                    if not math.isfinite(objective_metric):
                        logger.warning(
                            f"Non-finite pair-macro water IoU ({objective_metric}) at step "
                            f"{global_step}; treating as 0.0")
                        objective_metric = 0.0
                else:
                    objective_metric = val_miou

                n_vals += 1
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
                        # 6 decimals: 4-dp names produced ties, and the offline
                        # "highest step wins" tie-break picked a non-best ckpt in
                        # 6 of 27 audited seed dirs (best_ckpt_audit.json 2026-07-15).
                        ckpt_name = f"{self.arch}_{self.encoder}_s{self.seed}_step{global_step}_miou{val_miou:.6f}.pth"
                        ckpt_path = os.path.join(save_dir, ckpt_name)
                        
                        torch.save({
                            "step": global_step,
                            # Unwrap so state-dict keys carry no torch.compile
                            # `_orig_mod.` prefix and stay loadable everywhere.
                            "model_state_dict": unwrap_compiled(self.model).state_dict(),
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

                # --- MODEL-ONLY SNAPSHOTS (stage-2 late-window / periodic spine) ---
                # Saved at every val in the final `snapshot_last_frac` of the run
                # (post-LR-decay window) and/or every `snapshot_every_n_vals` vals.
                # Model-only (no optimizer state): ~1/3 the size, and no consumer
                # reads optimizer state. Never enter top_k_checkpoints -> never
                # pruned and never picked by the best.pth symlink.
                in_late = snapshot_last_frac > 0 and global_step >= max_steps * (1 - snapshot_last_frac)
                periodic = snapshot_every_n_vals > 0 and n_vals % snapshot_every_n_vals == 0
                if save_dir is not None and (in_late or periodic):
                    snap_name = f"{self.arch}_{self.encoder}_s{self.seed}_step{global_step}_snap_miou{val_miou:.6f}.pth"
                    snap_path = os.path.join(save_dir, snap_name)
                    torch.save({
                        "step": global_step,
                        "model_state_dict": unwrap_compiled(self.model).state_dict(),
                        "val_miou": val_miou,
                    }, snap_path)
                    print(f"Step {global_step}: Saved snapshot -> {snap_name}")

                if trial is not None:
                    #import optuna
                    trial.report(objective_metric, step=global_step)
                    if trial.should_prune():
                        raise optuna.TrialPruned(
                            f"Pruned at step {global_step} with objective {objective_metric:.4f}")
                        
            # Optionally persist the FINAL-step weights alongside the top-k.
            # Rationale: with an honest (pair-pure) val, generalization peaks
            # early, so top-k keeps only early checkpoints — save_last preserves
            # a late checkpoint for early-vs-late external comparisons. Never
            # enters top-k and is never auto-pruned.
            if save_dir is not None and save_last:
                last_name = f"{self.arch}_{self.encoder}_s{self.seed}_step{global_step}_last.pth"
                last_path = os.path.join(save_dir, last_name)
                torch.save({
                    "step": global_step,
                    "model_state_dict": unwrap_compiled(self.model).state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "val_miou": val_miou,
                }, last_path)
                print(f"Saved final-step checkpoint -> {last_name}")

            # Return the path to the best checkpoint (the last item in our sorted list)
            best_ckpt_path = top_k_checkpoints[-1][1] if top_k_checkpoints else None
            # Persist the full-float argmax as best.pth (relative symlink) so
            # downstream loaders never depend on parsing rounded filenames.
            if best_ckpt_path is not None:
                link_path = os.path.join(save_dir, "best.pth")
                if os.path.lexists(link_path):
                    os.remove(link_path)
                os.symlink(os.path.basename(best_ckpt_path), link_path)
            # 3-tuple: slot 0 keeps train.py's pooled-mIoU semantics; slot 2 is
            # the last val check's full metrics dict (carries strat/* under HPO).
            return best_val_miou, best_ckpt_path, final_val_metrics_dict

    @torch.no_grad()
    def val_epoch(self, dataloader) -> Dict[str, Any]:
        self.model.eval()
        self.val_metrics.reset()
        if self.strata_acc is not None:
            self.strata_acc.reset()

        total_loss = torch.zeros((), device=self.device)
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
            if self.strata_acc is not None:
                # Accumulate on fp32-softmax water prob (ladder convention); relies
                # on the val loader running shuffle=False in strata/memmap order.
                self.strata_acc.update(logits, y)

            total_loss += loss.detach()
            num_batches += 1

        metrics_dict = self.val_metrics.compute()
        metrics_dict["loss"] = (total_loss / max(1, num_batches)).item()

        if self.strata_acc is not None:
            assert self.strata_acc.base == self.strata_acc.N, (
                f"strata accumulator saw {self.strata_acc.base} chips != {self.strata_acc.N}"
            )
            for k, v in self.strata_acc.compute().items():
                metrics_dict[f"strat/{k}"] = v

        return metrics_dict
        
    @torch.no_grad()
    def test(self, dataloader) -> Dict[str, Any]:
        self.model.eval()
        self.val_metrics.reset()
        
        total_loss = torch.zeros((), device=self.device)
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
            
            total_loss += loss.detach()
            num_batches += 1
            
        metrics_dict = self.val_metrics.compute()
        metrics_dict["loss"] = (total_loss / max(1, num_batches)).item()
        
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
