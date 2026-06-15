"""Tests for gradient accumulation in SpectralTrainer.

The core guarantee is that gradient accumulation is an apples-to-apples memory
trade: running with ``accumulate_grad_batches=N`` and ``batch_size=B`` must be
equivalent to running with ``accumulate_grad_batches=1`` and ``batch_size=N*B``
at the SAME ``max_steps`` -- same number of optimizer steps, same number of
samples seen, and the same learning-rate schedule.

The LR-schedule assertion is the one that guards the scheduler against silently
desynchronising from the optimizer step count under accumulation.
"""

import sys
import types

import pytest
import torch
import torch.nn as nn

import src.engine.trainer as trainer_module
from src.engine.trainer import SpectralTrainer


# ---------------------------------------------------------------------------
# Lightweight fixtures: a trivial seg model / loss and a deterministic loader,
# plus a no-op metrics stub so the trainer does not require torchmetrics.
# ---------------------------------------------------------------------------
class _TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 2, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class _DictLoss(nn.Module):
    """Matches the trainer's contract: loss_fn(logits, y) -> {"loss": ...}."""

    def forward(self, logits, y):
        return {"loss": nn.functional.cross_entropy(logits, y)}


class _CountingLoader:
    """Deterministic, effectively-infinite loader that records samples served."""

    def __init__(self, batch_size, img=4):
        self.batch_size = batch_size
        self.img = img
        self.served = 0

    def __iter__(self):
        gen = torch.Generator().manual_seed(7)
        while True:
            self.served += self.batch_size
            yield {
                "pixel_values": torch.randn(self.batch_size, 2, self.img, self.img, generator=gen),
                "labels": torch.randint(0, 2, (self.batch_size, self.img, self.img), generator=gen),
            }

    def __len__(self):  # some code paths call len(); value is unimportant here
        return 1_000_000


class _NoopMetrics:
    def __init__(self, *args, **kwargs):
        pass

    def reset(self):
        pass

    def update(self, *args, **kwargs):
        pass

    def compute(self):
        return {"mIoU": 0.0}


@pytest.fixture(autouse=True)
def _isolate_trainer(monkeypatch):
    """Make SpectralTrainer.fit runnable as a unit test:

    - swap SegmentationMetrics for a no-op (avoids torchmetrics state),
    - inject a stub `wandb` so the in-function `import wandb` resolves without a
      real W&B install / login. fit() only reads `wandb.run` (None disables
      logging) and may call `wandb.log`.
    """
    monkeypatch.setattr(trainer_module, "SegmentationMetrics", _NoopMetrics)

    wandb_stub = types.ModuleType("wandb")
    wandb_stub.run = None
    wandb_stub.log = lambda *a, **k: None
    monkeypatch.setitem(sys.modules, "wandb", wandb_stub)


def _run(accum, batch_size, max_steps, val_check_interval, warmup=3):
    """Run a short fit() and return (n_optimizer_steps, samples_seen, lr_history)."""
    torch.manual_seed(0)
    model = _TinyModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    # Mirror the project's warmup + cosine schedule, sized in optimizer steps.
    warmup_sched = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1e-3, end_factor=1.0, total_iters=warmup
    )
    decay_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, max_steps - warmup), eta_min=1e-4
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup_sched, decay_sched], milestones=[warmup]
    )

    # Count optimizer steps and record the LR after each scheduler step.
    opt_steps = {"n": 0}
    real_opt_step = optimizer.step

    def counting_opt_step(*a, **k):
        opt_steps["n"] += 1
        return real_opt_step(*a, **k)

    optimizer.step = counting_opt_step

    lr_history = []
    real_sched_step = scheduler.step

    def recording_sched_step(*a, **k):
        real_sched_step(*a, **k)
        lr_history.append(optimizer.param_groups[0]["lr"])

    scheduler.step = recording_sched_step

    loader = _CountingLoader(batch_size)
    trainer = SpectralTrainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=_DictLoss(),
        device=torch.device("cpu"),
        use_amp=False,
        gradient_clip_val=0.0,
        num_classes=2,
        accumulate_grad_batches=accum,
    )
    # Skip validation; it is irrelevant to the train-loop properties under test.
    trainer.val_epoch = lambda dl: {"mIoU": 0.0}
    trainer.fit(loader, loader, max_steps=max_steps, val_check_interval=val_check_interval)

    return opt_steps["n"], loader.served, lr_history


def test_accum_one_is_one_optimizer_step_per_max_step():
    """accumulate_grad_batches=1 must take exactly max_steps optimizer steps."""
    max_steps = 10
    n_opt, _, lrs = _run(accum=1, batch_size=8, max_steps=max_steps, val_check_interval=max_steps)
    assert n_opt == max_steps
    assert len(lrs) == max_steps


def test_accumulation_is_apples_to_apples_with_larger_batch():
    """accum=2 @ bs=128 must match accum=1 @ bs=256 at the same max_steps.

    This is the regression guard: it checks equal optimizer steps, equal
    samples seen, and an identical LR schedule. A scheduler that advanced per
    micro-batch (or per optimizer step against a micro-batch-sized horizon)
    would change the LR history and fail here.
    """
    max_steps, val_check = 12, 12
    base_steps, base_samples, base_lrs = _run(
        accum=1, batch_size=256, max_steps=max_steps, val_check_interval=val_check
    )
    acc_steps, acc_samples, acc_lrs = _run(
        accum=2, batch_size=128, max_steps=max_steps, val_check_interval=val_check
    )

    assert acc_steps == base_steps, "accumulated run must do the same number of optimizer steps"
    assert acc_samples == base_samples, "accumulated run must see the same number of samples"
    assert len(acc_lrs) == len(base_lrs), "accumulated run must have the same number of LR updates"

    max_lr_diff = max(abs(a - b) for a, b in zip(acc_lrs, base_lrs))
    assert max_lr_diff < 1e-12, f"LR schedules diverged (max diff {max_lr_diff})"


def test_accumulated_gradient_equals_large_batch_gradient():
    """Summing loss/accum over N micro-batches == one backward on the N*B batch."""
    torch.manual_seed(123)
    big_x = torch.randn(8, 2, 4, 4)
    big_y = torch.randint(0, 2, (8, 4, 4))

    big = _TinyModel()
    big.zero_grad()
    nn.functional.cross_entropy(big(big_x), big_y).backward()
    big_grad = big.conv.weight.grad.clone()

    accum = 4
    micro = _TinyModel()
    micro.load_state_dict(big.state_dict())
    micro.zero_grad()
    chunk = big_x.shape[0] // accum
    for i in range(accum):
        xs = big_x[i * chunk : (i + 1) * chunk]
        ys = big_y[i * chunk : (i + 1) * chunk]
        (nn.functional.cross_entropy(micro(xs), ys) / accum).backward()
    accum_grad = micro.conv.weight.grad.clone()

    assert torch.allclose(big_grad, accum_grad, atol=1e-6)
