"""Unit tests for the pure functions of the dataset_v3 split/memmap stages.

These cover the pieces that can be wrong silently: the rect-to-rect distance,
the stratified draw's determinism and quota behavior, and the histogram
moments. The heavy stages are covered by their own runtime assertions on the
VM (regression anchors, identity checks); these tests protect the arithmetic.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "dataset_v3"))

from assess_eval_distance import KM_PER_DEG, rect_to_rect_km  # noqa: E402
from apply_splits import PINNED_PAIRS, assign_strata  # noqa: E402
from derive_norm_constants import moments_from_hist  # noqa: E402


class TestRectToRect:
    AOI = (10.0, 50.0, 11.0, 50.5)  # 1 deg x 0.5 deg box at 50 N

    def test_overlap_is_zero(self):
        boxes = np.array([
            [10.2, 50.1, 10.4, 50.2],   # fully inside
            [9.9, 49.9, 10.1, 50.1],    # corner overlap
            [10.0, 50.0, 11.0, 50.5],   # identical
        ])
        assert (rect_to_rect_km(boxes, self.AOI) == 0.0).all()

    def test_pure_north_separation(self):
        # 0.1 deg latitude gap => 11.132 km, no cos scaling on latitude.
        boxes = np.array([[10.2, 50.6, 10.4, 50.7]])
        got = rect_to_rect_km(boxes, self.AOI)[0]
        assert got == pytest.approx(0.1 * KM_PER_DEG, rel=1e-9)

    def test_pure_east_separation_cos_scaled(self):
        # 0.2 deg longitude gap at the AOI's mean latitude (50.25 deg).
        boxes = np.array([[11.2, 50.1, 11.4, 50.2]])
        expected = 0.2 * KM_PER_DEG * np.cos(np.deg2rad(50.25))
        assert rect_to_rect_km(boxes, self.AOI)[0] == pytest.approx(expected, rel=1e-9)

    def test_diagonal_is_hypot_not_axis(self):
        boxes = np.array([[11.1, 50.6, 11.2, 50.7]])  # 0.1 deg east, 0.1 deg north
        dx = 0.1 * KM_PER_DEG * np.cos(np.deg2rad(50.25))
        dy = 0.1 * KM_PER_DEG
        assert rect_to_rect_km(boxes, self.AOI)[0] == pytest.approx(np.hypot(dx, dy), rel=1e-9)


def _synthetic_chips(n_pairs=40, chips_per_pair=(50, 400), seed=7, stratum="2_C"):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_pairs):
        pair = "PAIR_T%04d" % i
        for c in range(int(rng.integers(*chips_per_pair))):
            rows.append((pair, c, stratum))
    return pd.DataFrame(rows, columns=["pair_name", "chip_id", "stratum_coarse"])


class TestDraw:
    def test_deterministic(self):
        chips = _synthetic_chips()
        s1, r1 = assign_strata(chips, seed=42)
        s2, r2 = assign_strata(chips, seed=42)
        assert s1.equals(s2) and r1.equals(r2)

    def test_seed_changes_assignment(self):
        chips = _synthetic_chips()
        s1, _ = assign_strata(chips, seed=42)
        s2, _ = assign_strata(chips, seed=43)
        assert not s1.equals(s2)

    def test_scene_purity_and_shares(self):
        chips = _synthetic_chips(n_pairs=200)
        split, reason = assign_strata(chips, seed=42)
        frame = chips.assign(split=split.values)
        assert (frame.groupby("pair_name")["split"].nunique() == 1).all()
        shares = frame["split"].value_counts(normalize=True)
        # Whole-pair granularity on 200 pairs: expect within a few points.
        assert shares["train"] == pytest.approx(0.70, abs=0.05)
        assert shares["val"] == pytest.approx(0.20, abs=0.05)
        assert shares["test"] == pytest.approx(0.10, abs=0.05)
        assert set(reason.unique()) == {"stratified_draw"}

    def test_thin_stratum_goes_to_train(self):
        chips = pd.DataFrame({
            "pair_name": ["PAIR_TA"] * 10 + ["PAIR_TB"] * 10,
            "chip_id": list(range(10)) * 2,
            "stratum_coarse": "1_D",
        })
        split, reason = assign_strata(chips, seed=42)
        assert (split == "train").all()
        assert (reason == "thin_stratum_train").all()

    def test_pinned_pair_forced_to_train(self):
        pinned = sorted(PINNED_PAIRS)[0]
        chips = _synthetic_chips(n_pairs=30)
        extra = pd.DataFrame({"pair_name": pinned, "chip_id": range(300),
                              "stratum_coarse": "2_C"})
        chips = pd.concat([chips, extra], ignore_index=True)
        split, reason = assign_strata(chips, seed=42)
        frame = chips.assign(split=split.values, reason=reason.values)
        mine = frame[frame["pair_name"] == pinned]
        assert (mine["split"] == "train").all()
        assert (mine["reason"] == "pinned_train").all()


class TestMoments:
    def test_known_distribution(self):
        # Two-point distribution on bin centers: mean/std computable by hand.
        centres = np.array([-10.0, 0.0, 10.0])
        counts = np.array([100, 0, 100], dtype=np.int64)
        n, mean, std = moments_from_hist(counts, centres)
        assert n == 200
        assert mean == pytest.approx(0.0)
        assert std == pytest.approx(10.0)

    def test_weighted_mean(self):
        centres = np.array([1.0, 3.0])
        counts = np.array([3, 1], dtype=np.int64)
        n, mean, std = moments_from_hist(counts, centres)
        assert n == 4
        assert mean == pytest.approx(1.5)
        assert std == pytest.approx(np.sqrt((3 * 0.25 + 1 * 2.25) / 4))
