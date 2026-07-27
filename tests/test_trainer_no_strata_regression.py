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


class _StubAccumulator:
    """Minimal strata accumulator: fit() only reads `strat/pair_macro_water_iou`
    (via val_epoch -> self.strata_acc.compute()) for selection. Yields a
    controllable pair-macro sequence, one value per val check, so the divergence
    scenario (pooled DOWN, pair-macro UP) can be driven deterministically.
    """

    def __init__(self, pmw_sequence):
        self._seq = list(pmw_sequence)
        self._i = 0
        # val_epoch asserts base == N after the val loop; keep them equal.
        self.base = 0
        self.N = 0

    def reset(self):
        pass

    def update(self, logits, y):
        pass

    def compute(self):
        val = self._seq[self._i]
        self._i += 1
        return {"pair_macro_water_iou": val}


def _list_ckpts(save_dir):
    import os

    return sorted(
        f for f in os.listdir(save_dir)
        if f.endswith(".pth") and not f.startswith("best")
    )


def test_topk_selection_uses_pair_macro_when_accumulator_active(tmp_path):
    """Divergence scenario: pooled mIoU DECREASES while pair-macro INCREASES.

    top-k must keep the pair-macro-best (i.e. LATEST) checkpoints, name them with
    the `pmwiou` tag, and store both metric keys in the payload.
    """
    import os

    # Four val checks. Pooled mIoU falls each check; pair-macro rises each check.
    pooled_seq = [0.90, 0.80, 0.70, 0.60]
    pmw_seq = [0.10, 0.20, 0.30, 0.40]

    acc = _StubAccumulator(pmw_seq)
    trainer = _make_trainer(strata_accumulator=acc)

    pooled_iter = iter(pooled_seq)
    # val_epoch is stubbed: return the falling pooled mIoU, and mirror the stub
    # accumulator's compute() into the strat/* key exactly as the real val_epoch
    # would (so fit() reads objective_metric from the metrics dict).
    def _fake_val(dl):
        pooled = next(pooled_iter)
        return {"mIoU": pooled, "loss": 0.1,
                "strat/pair_macro_water_iou": acc.compute()["pair_macro_water_iou"]}

    trainer.val_epoch = _fake_val
    loader = _CountingLoader(batch_size=8)

    keep_top_k = 2
    best_iou, best_ckpt_path, final_metrics = trainer.fit(
        loader, loader, max_steps=16, val_check_interval=4,
        save_dir=str(tmp_path), keep_top_k=keep_top_k,
    )

    kept = _list_ckpts(tmp_path)
    # keep_top_k=2 and pair-macro rising => the two LATEST checks survive
    # (steps 12 and 16), tagged pmwiou, NOT the early (pooled-best) ones.
    assert len(kept) == keep_top_k, kept
    assert all("pmwiou" in name for name in kept), kept
    assert all("_miou" not in name for name in kept), kept
    steps = sorted(int(name.split("_step")[1].split("_")[0]) for name in kept)
    assert steps == [12, 16], steps

    # best.pth -> pair-macro-best (latest, step 16, pmw 0.40)
    assert best_ckpt_path is not None
    assert "pmwiou0.400000" in os.path.basename(best_ckpt_path)

    # payload carries BOTH keys under the accumulator
    payload = torch.load(best_ckpt_path, map_location="cpu", weights_only=False)
    assert "val_miou" in payload
    assert "pair_macro_water_iou" in payload
    assert "optimizer_state_dict" in payload
    assert abs(payload["pair_macro_water_iou"] - 0.40) < 1e-9
    # slot-0 pooled bookkeeping is unchanged (max pooled seen)
    assert abs(best_iou - 0.90) < 1e-9


def test_topk_selection_pooled_names_when_no_accumulator(tmp_path):
    """No accumulator: `miou`-tagged names, ranked by pooled mIoU (unchanged)."""
    import os

    pooled_seq = [0.60, 0.70, 0.80, 0.90]  # rising: latest are the best
    trainer = _make_trainer(strata_accumulator=None)
    pooled_iter = iter(pooled_seq)
    trainer.val_epoch = lambda dl: {"mIoU": next(pooled_iter), "loss": 0.1}
    loader = _CountingLoader(batch_size=8)

    keep_top_k = 2
    _, best_ckpt_path, _ = trainer.fit(
        loader, loader, max_steps=16, val_check_interval=4,
        save_dir=str(tmp_path), keep_top_k=keep_top_k,
    )

    kept = _list_ckpts(tmp_path)
    assert len(kept) == keep_top_k, kept
    assert all("_miou" in name for name in kept), kept
    assert all("pmwiou" not in name for name in kept), kept
    steps = sorted(int(name.split("_step")[1].split("_")[0]) for name in kept)
    assert steps == [12, 16], steps  # pooled-best = latest two

    assert "miou0.900000" in os.path.basename(best_ckpt_path)
    payload = torch.load(best_ckpt_path, map_location="cpu", weights_only=False)
    assert "val_miou" in payload
    assert "pair_macro_water_iou" not in payload  # no key on the no-accum path
    assert "optimizer_state_dict" in payload


def test_model_weights_only_checkpoint_payload(tmp_path):
    """HPO mode saves serving weights without optimizer or metric payloads."""
    import os

    trainer = _make_trainer(strata_accumulator=None)
    trainer.val_epoch = lambda dl: {"mIoU": 0.75, "loss": 0.1}
    loader = _CountingLoader(batch_size=8)

    _, best_ckpt_path, _ = trainer.fit(
        loader,
        loader,
        max_steps=4,
        val_check_interval=4,
        save_dir=str(tmp_path),
        keep_top_k=1,
        save_last=True,
        model_weights_only=True,
    )

    assert best_ckpt_path is not None
    assert os.path.islink(tmp_path / "best.pth")
    saved = _list_ckpts(tmp_path)
    assert len(saved) == 2  # objective-best plus final-step weights

    for name in saved:
        payload = torch.load(tmp_path / name, map_location="cpu", weights_only=True)
        assert set(payload) == {"model_state_dict"}
        assert payload["model_state_dict"]


def test_pruned_trial_keeps_discoverable_best_weights(tmp_path):
    """Pruning raises before fit() returns, but best.pth must already be valid."""
    import os

    class _PruningTrial:
        def report(self, value, step):
            self.reported = (value, step)

        def should_prune(self):
            return True

    trainer = _make_trainer(strata_accumulator=None)
    trainer.val_epoch = lambda dl: {"mIoU": 0.65, "loss": 0.1}
    loader = _CountingLoader(batch_size=8)
    trial = _PruningTrial()

    with pytest.raises(trainer_module.optuna.TrialPruned):
        trainer.fit(
            loader,
            loader,
            max_steps=8,
            val_check_interval=4,
            trial=trial,
            save_dir=str(tmp_path),
            keep_top_k=1,
            model_weights_only=True,
        )

    best_link = tmp_path / "best.pth"
    assert os.path.islink(best_link)
    assert best_link.resolve().exists()
    payload = torch.load(best_link, map_location="cpu", weights_only=True)
    assert set(payload) == {"model_state_dict"}
    assert trial.reported == (0.65, 4)


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
