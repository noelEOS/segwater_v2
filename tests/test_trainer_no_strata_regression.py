"""SpectralTrainer strata wiring: default path unchanged + smoke with accumulator.

Two guarantees:
  1. With NO strata_accumulator (the train.py path), fit() returns a 3-tuple
     whose slot 0 is a float pooled-mIoU, slot 2 a dict, and the val metrics
     dict carries NO `strat/` keys — i.e. behaviour is unchanged except the
     appended return slot.
  2. With a real StratifiedWaterAccumulator, val_epoch() populates
     `strat/pair_macro_water_iou` and asserts base==N over a tiny val loader.

Fixture pattern (tiny model / dict loss / stub wandb+metrics) mirrors
tests/test_gradient_accumulation.py.
"""
import sys
import types

import numpy as np
import pytest
import torch
import torch.nn as nn

import src.engine.trainer as trainer_module
from src.engine.trainer import SpectralTrainer
from src.utils.stratified_metrics import StratifiedWaterAccumulator


class _TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 2, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class _DictLoss(nn.Module):
    def forward(self, logits, y):
        return {"loss": nn.functional.cross_entropy(logits, y, ignore_index=255)}


class _CountingLoader:
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

    def __len__(self):
        return 1_000_000


class _NoopMetrics:
    def __init__(self, *args, **kwargs):
        pass

    def reset(self):
        pass

    def update(self, *args, **kwargs):
        pass

    def compute(self):
        return {"mIoU": 0.5}


@pytest.fixture(autouse=True)
def _isolate_trainer(monkeypatch):
    monkeypatch.setattr(trainer_module, "SegmentationMetrics", _NoopMetrics)
    wandb_stub = types.ModuleType("wandb")
    wandb_stub.run = None
    wandb_stub.log = lambda *a, **k: None
    wandb_stub.define_metric = lambda *a, **k: None
    monkeypatch.setitem(sys.modules, "wandb", wandb_stub)


def _make_trainer(strata_accumulator=None):
    torch.manual_seed(0)
    model = _TinyModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1.0, end_factor=1.0, total_iters=1
    )
    return SpectralTrainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=_DictLoss(),
        device=torch.device("cpu"),
        use_amp=False,
        gradient_clip_val=0.0,
        num_classes=2,
        strata_accumulator=strata_accumulator,
    )


def test_fit_returns_three_tuple_no_strata_keys():
    trainer = _make_trainer(strata_accumulator=None)
    # skip the real val loop; irrelevant to the return-contract check
    trainer.val_epoch = lambda dl: {"mIoU": 0.5, "loss": 0.1}
    loader = _CountingLoader(batch_size=8)

    result = trainer.fit(loader, loader, max_steps=4, val_check_interval=4)

    assert isinstance(result, tuple) and len(result) == 3
    best_iou, best_ckpt_path, final_metrics = result
    assert isinstance(best_iou, float)
    assert best_ckpt_path is None  # no save_dir passed
    assert isinstance(final_metrics, dict)
    assert not any(k.startswith("strat/") for k in final_metrics)


def _tiny_val_loader(n_chips, batch, img=4):
    """Deterministic finite val loader over exactly n_chips chips in order."""
    xs = torch.randn(n_chips, 2, img, img, generator=torch.Generator().manual_seed(3))
    ys = torch.randint(0, 2, (n_chips, img, img), generator=torch.Generator().manual_seed(4))

    class _FiniteLoader:
        def __iter__(self_inner):
            for lo in range(0, n_chips, batch):
                hi = min(lo + batch, n_chips)
                yield {"pixel_values": xs[lo:hi], "labels": ys[lo:hi]}

        def __len__(self_inner):
            return (n_chips + batch - 1) // batch

    return _FiniteLoader()


def test_val_epoch_with_accumulator_emits_strat_keys():
    n_chips = 6
    stratum_id = np.array([0, 1, 2, 2, 2, 2], dtype=np.int64)
    pair_id = np.array([0, 0, 0, 0, 1, 1], dtype=np.int64)
    eligible = np.array([True, True], dtype=bool)
    acc = StratifiedWaterAccumulator(
        stratum_id=stratum_id, pair_id=pair_id, eligible=eligible,
        device=torch.device("cpu"),
    )
    trainer = _make_trainer(strata_accumulator=acc)
    val_dl = _tiny_val_loader(n_chips, batch=4)

    metrics = trainer.val_epoch(val_dl)

    assert acc.base == acc.N == n_chips
    assert "strat/pair_macro_water_iou" in metrics
    assert "strat/pure_land_fpr" in metrics
    assert "strat/n_pairs_used" in metrics
    # pooled metrics still present
    assert "mIoU" in metrics
