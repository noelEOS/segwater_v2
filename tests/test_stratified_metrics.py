"""StratifiedWaterAccumulator vs a pure-NumPy reference.

The accumulator is the HPO objective's ground truth, so we check it against an
independent NumPy reimplementation of the exact ladder convention (code =
2*target + pred on a tau=0.5 threshold over fp32 softmax water prob), on tiny
synthetic data that exercises every stratum, ignore pixels, >=2 batches, and a
pair whose mixed-pixel union is empty (its water IoU is NaN -> excluded).
"""
import numpy as np
import torch

from src.utils.stratified_metrics import StratifiedWaterAccumulator

IGNORE = 255
TAU = 0.5

# 12 chips, 3 pairs. Strata: 0 pure-land, 1 pure-water, 2 mixed, 3 all-ignore.
# Pair 0: mixed chips (eligible). Pair 1: mixed chips (eligible). Pair 2: a
# single mixed chip whose GT has no water AND we never predict water -> its
# pair confusion has union 0 -> NaN -> must be excluded from the macro mean.
STRATUM_ID = np.array([0, 1, 2, 2, 2, 2, 0, 1, 3, 2, 2, 2], dtype=np.int64)
PAIR_ID = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 2], dtype=np.int64)
# eligibility is supplied directly (not derived from a mixed-count gate here):
# pairs 0 and 1 eligible, pair 2 eligible too (so the NaN-exclusion is exercised
# via the finite-union filter, not via the eligibility gate).
ELIGIBLE = np.array([True, True, True], dtype=bool)

H = W = 4
N = len(STRATUM_ID)


def _make_batch(rng, n, force):
    """Return (logits[n,2,H,W] fp32, y[n,H,W] int64) with some ignore pixels.

    `force` is a per-chip callable(i) -> ("y_override" or None, "water_prob"
    override or None) so we can shape specific strata / the empty-union pair.
    """
    logits = torch.from_numpy(rng.standard_normal((n, 2, H, W)).astype(np.float32))
    y = torch.from_numpy(rng.integers(0, 2, size=(n, H, W)).astype(np.int64))
    # sprinkle ignore
    ign = rng.random((n, H, W)) < 0.15
    y[torch.from_numpy(ign)] = IGNORE
    return logits, y


def _ref_confusions(all_logits, all_y):
    """Independent NumPy reference: per-pair [P,4] and per-stratum [4,4] CMs."""
    P = len(ELIGIBLE)
    pair_cm = np.zeros((P, 4), dtype=np.int64)
    strat_cm = np.zeros((4, 4), dtype=np.int64)
    for i in range(N):
        logits = all_logits[i].numpy()  # [2,H,W]
        y = all_y[i].numpy()            # [H,W]
        prob_w = np.exp(logits[1]) / (np.exp(logits[0]) + np.exp(logits[1]))
        pred = (prob_w >= TAU).astype(np.int64)
        valid = y != IGNORE
        yb = np.clip(y, 0, 1)
        s = STRATUM_ID[i]
        p = PAIR_ID[i]
        for code in range(4):
            t, pr = code // 2, code % 2
            m = valid & (yb == t) & (pred == pr)
            cnt = int(m.sum())
            strat_cm[s, code] += cnt
            if s == 2:  # mixed only for pair_cm
                pair_cm[p, code] += cnt
    return pair_cm, strat_cm


def _water_iou(cm4):
    tp, fp, fn = float(cm4[3]), float(cm4[1]), float(cm4[2])
    u = tp + fp + fn
    return tp / u if u > 0 else float("nan")


def _build_data():
    rng = np.random.default_rng(0)
    logits, y = _make_batch(rng, N, None)
    # Force pair 2's single mixed chip (index 11) to have NO water GT and NO
    # water prediction -> empty water union -> NaN water IoU.
    y[11] = 0
    y[11][0, 0] = IGNORE  # keep an ignore pixel too
    logits[11, 1] = -10.0  # water logit very low -> pred all land
    logits[11, 0] = 10.0
    # Chip 8 is the all-ignore stratum (id 3): its labels are all ignore, so no
    # valid pixel exists and strat_cm row 3 must stay zero.
    y[8] = IGNORE
    return logits, y


def test_accumulator_matches_numpy_reference():
    logits, y = _build_data()

    acc = StratifiedWaterAccumulator(
        stratum_id=STRATUM_ID, pair_id=PAIR_ID, eligible=ELIGIBLE,
        device=torch.device("cpu"), tau=TAU, ignore_index=IGNORE,
    )
    # feed in >=2 batches (5 + 7)
    acc.update(logits[:5], y[:5])
    acc.update(logits[5:], y[5:])

    assert acc.base == acc.N == N

    ref_pair_cm, ref_strat_cm = _ref_confusions(logits, y)
    np.testing.assert_array_equal(acc.pair_cm.cpu().numpy(), ref_pair_cm)
    np.testing.assert_array_equal(acc.strat_cm.cpu().numpy(), ref_strat_cm)

    # all-ignore stratum row stays zero
    assert acc.strat_cm.cpu().numpy()[3].sum() == 0

    out = acc.compute()

    # pair-macro = mean over eligible & finite-union pairs
    pair_wious = np.array([_water_iou(ref_pair_cm[p]) for p in range(len(ELIGIBLE))])
    sel = ELIGIBLE & ~np.isnan(pair_wious)
    ref_macro = float(np.mean(pair_wious[sel]))
    assert np.isclose(out["pair_macro_water_iou"], ref_macro, rtol=0, atol=1e-9)

    # pair 2 has NaN union -> excluded -> only pairs 0,1 used
    assert out["n_pairs_used"] == 2.0

    # mixed_water_iou = water IoU over the aggregated mixed stratum row
    assert np.isclose(out["mixed_water_iou"], _water_iou(ref_strat_cm[2]), atol=1e-9)


def test_reset_zeros_state():
    logits, y = _build_data()
    acc = StratifiedWaterAccumulator(
        stratum_id=STRATUM_ID, pair_id=PAIR_ID, eligible=ELIGIBLE,
        device=torch.device("cpu"), tau=TAU, ignore_index=IGNORE,
    )
    acc.update(logits, y)
    acc.reset()
    assert acc.base == 0
    assert acc.pair_cm.sum().item() == 0
    assert acc.strat_cm.sum().item() == 0


def test_from_npz_uses_stored_eligible_mask(tmp_path):
    """HPO subset npz stores `eligible` (+ min_mixed) -> used verbatim."""
    npz = tmp_path / "hpo.npz"
    eligible = np.array([True, True, False], dtype=bool)
    np.savez(
        npz,
        stratum_id=STRATUM_ID,
        pair_id=PAIR_ID,
        eligible=eligible,
        min_mixed=np.int64(5),
        N=np.int64(N),
        # include a pair_mixed_count that would give a DIFFERENT mask if derived,
        # to prove the stored mask wins.
        pair_mixed_count=np.array([100, 0, 100], dtype=np.int64),
    )
    acc = StratifiedWaterAccumulator.from_npz(str(npz), torch.device("cpu"))
    np.testing.assert_array_equal(acc.eligible.cpu().numpy(), eligible)
    assert acc.N == N
    assert acc.P == len(eligible)


def test_from_npz_derives_eligible_from_mixed_count(tmp_path):
    """Ladder-style npz (no eligible/min_mixed) -> derived as count >= 20."""
    npz = tmp_path / "ladder.npz"
    pair_mixed_count = np.array([25, 19, 20], dtype=np.int64)  # >=20 -> [T, F, T]
    np.savez(
        npz,
        stratum_id=STRATUM_ID,
        pair_id=PAIR_ID,
        pair_mixed_count=pair_mixed_count,
        N=np.int64(N),
    )
    acc = StratifiedWaterAccumulator.from_npz(str(npz), torch.device("cpu"))
    np.testing.assert_array_equal(
        acc.eligible.cpu().numpy(), np.array([True, False, True])
    )
    # default_min_mixed override changes the derived gate.
    acc10 = StratifiedWaterAccumulator.from_npz(
        str(npz), torch.device("cpu"), default_min_mixed=10
    )
    np.testing.assert_array_equal(
        acc10.eligible.cpu().numpy(), np.array([True, True, True])
    )


def test_from_npz_expected_n_mismatch_raises(tmp_path):
    npz = tmp_path / "ladder.npz"
    np.savez(
        npz,
        stratum_id=STRATUM_ID,
        pair_id=PAIR_ID,
        pair_mixed_count=np.array([25, 25, 25], dtype=np.int64),
        N=np.int64(N),
    )
    # matching expected_n is fine
    StratifiedWaterAccumulator.from_npz(str(npz), torch.device("cpu"), expected_n=N)
    # mismatch raises
    import pytest

    with pytest.raises(ValueError):
        StratifiedWaterAccumulator.from_npz(
            str(npz), torch.device("cpu"), expected_n=N + 1
        )


def test_from_npz_compute_matches_direct(tmp_path):
    """Accumulator loaded from npz gives identical compute() to a direct build."""
    logits, y = _build_data()
    npz = tmp_path / "hpo.npz"
    np.savez(
        npz,
        stratum_id=STRATUM_ID,
        pair_id=PAIR_ID,
        eligible=ELIGIBLE,
        min_mixed=np.int64(1),
        N=np.int64(N),
    )
    acc_npz = StratifiedWaterAccumulator.from_npz(str(npz), torch.device("cpu"))
    acc_direct = StratifiedWaterAccumulator(
        stratum_id=STRATUM_ID, pair_id=PAIR_ID, eligible=ELIGIBLE,
        device=torch.device("cpu"), tau=TAU, ignore_index=IGNORE,
    )
    for acc in (acc_npz, acc_direct):
        acc.update(logits[:5], y[:5])
        acc.update(logits[5:], y[5:])

    out_npz = acc_npz.compute()
    out_direct = acc_direct.compute()
    assert out_npz.keys() == out_direct.keys()
    for k in out_direct:
        a, b = out_npz[k], out_direct[k]
        if isinstance(b, float) and np.isnan(b):
            assert np.isnan(a)
        else:
            assert np.isclose(a, b, atol=1e-9)


def test_guardrails_match_reference():
    logits, y = _build_data()
    acc = StratifiedWaterAccumulator(
        stratum_id=STRATUM_ID, pair_id=PAIR_ID, eligible=ELIGIBLE,
        device=torch.device("cpu"), tau=TAU, ignore_index=IGNORE,
    )
    acc.update(logits, y)
    out = acc.compute()
    _, ref_strat_cm = _ref_confusions(logits, y)

    def safe(n, d):
        return float(n / d) if d else 0.0

    land = ref_strat_cm[0]
    water = ref_strat_cm[1]
    mixed = ref_strat_cm[2]
    tot = ref_strat_cm.sum(axis=0)

    assert np.isclose(out["pure_land_fpr"], safe(land[1], land[1] + land[0]), atol=1e-9)
    assert np.isclose(out["pure_water_fnr"], safe(water[2], water[2] + water[3]), atol=1e-9)
    assert np.isclose(out["mixed_precision"], safe(mixed[3], mixed[3] + mixed[1]), atol=1e-9)
    assert np.isclose(out["mixed_recall"], safe(mixed[3], mixed[3] + mixed[2]), atol=1e-9)
    assert np.isclose(out["overall_precision"], safe(tot[3], tot[3] + tot[1]), atol=1e-9)
    assert np.isclose(out["overall_recall"], safe(tot[3], tot[3] + tot[2]), atol=1e-9)
