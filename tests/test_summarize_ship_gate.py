"""Unit tests for the Demak-gate summary extraction.

``summarize_ship_gate.one_arm`` reads three scorer CSVs and returns one row, so
the tests write minimal CSVs to tmp_path -- no VM, no rasters, no torch.

The column DERIVATIONS these pin were validated end-to-end against the committed
Swin-B ``demak_gate_ship_summary.csv``: applying this same arithmetic to that
campaign's raw scorer dirs reproduced all six columns to <5e-7 (tau_star and
n_pairs exactly), the residual being the committed CSV's 6-decimal rounding.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
VM = REPO / "scripts" / "evaluation" / "vm"
sys.path.insert(0, str(VM))
sys.path.insert(0, str(VM / "ship"))

from summarize_ship_gate import one_arm  # noqa: E402

MODEL = "cnxb_s42_last"


def write_arm(d: Path, *, model=MODEL, n_pairs=6, n_success=6,
              thresholds=(0.0, 0.5, 1.0), pred_px=920, ref_px=1000,
              iou_at_05=0.6717, tau=0.35, iou_at_tau=0.6872, auc=0.9826,
              best_metrics=("iou", "f1", "mcc")):
    d.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([dict(model_name=model, n_pairs=n_pairs,
                       n_success_pairs=n_success, roc_auc_mean=auc)]
                 ).to_csv(d / "metrics_summary.csv", index=False)
    # tau* must come from the `iou` row; f1/mcc rows carry decoy values.
    pd.DataFrame([
        dict(model_name=model, optimize_metric=m,
             best_threshold=tau if m == "iou" else tau + 0.07,
             best_metric_mean=iou_at_tau if m == "iou" else 0.111)
        for m in best_metrics
    ]).to_csv(d / "threshold_best_summary.csv", index=False)
    pd.DataFrame([
        dict(model_name=model, threshold=t,
             iou_mean=iou_at_05 if t == 0.5 else 0.999,
             prediction_water_pixels_total=pred_px if t == 0.5 else 12345,
             reference_water_pixels_total=ref_px)
        for t in thresholds
    ]).to_csv(d / "threshold_summary.csv", index=False)
    return d


def call(d, **kw):
    problems = []
    row = one_arm(d, kw.pop("tag", "cnxb"), kw.pop("seed", "s42"),
                  kw.pop("variant", "last"), kw.pop("expect_pairs", 6), problems)
    return row, problems


def test_happy_path_derives_every_column(tmp_path):
    row, problems = call(write_arm(tmp_path / "a"))
    assert problems == []
    assert row["seed"] == "s42" and row["variant"] == "last"
    assert row["iou_at_0p5"] == pytest.approx(0.6717)
    assert row["tau_star"] == pytest.approx(0.35)          # the iou row, not f1/mcc
    assert row["iou_at_tau_star"] == pytest.approx(0.6872)
    assert row["area_bias"] == pytest.approx(0.92)         # 920 / 1000
    assert row["roc_auc"] == pytest.approx(0.9826)
    assert row["n_pairs"] == 6


def test_threshold_grid_without_0p5_is_fatal(tmp_path):
    """A grid missing 0.5 must fail, never fall back to a neighbour."""
    d = write_arm(tmp_path / "a", thresholds=(0.0, 0.49, 0.51, 1.0))
    row, problems = call(d)
    assert row is None
    assert any("threshold 0.5" in p for p in problems)


def test_short_scored_run_is_fatal(tmp_path):
    """record_and_continue can leave n_success < n_pairs; that must not pass."""
    row, problems = call(write_arm(tmp_path / "a", n_pairs=6, n_success=5))
    assert row is None
    assert any("n_success=5" in p for p in problems)


def test_wrong_pair_count_is_fatal(tmp_path):
    row, problems = call(write_arm(tmp_path / "a", n_pairs=5, n_success=5))
    assert row is None
    assert any("expected 6" in p for p in problems)


def test_mispointed_config_detected_by_model_name(tmp_path):
    """The dir claims s42/last but the CSVs were scored from another arm."""
    row, problems = call(write_arm(tmp_path / "a", model="cnxb_s19_best"))
    assert row is None
    assert any("model_name" in p for p in problems)


def test_missing_csv_is_reported_not_raised(tmp_path):
    d = write_arm(tmp_path / "a")
    (d / "threshold_summary.csv").unlink()
    row, problems = call(d)
    assert row is None and problems


def test_zero_reference_water_is_fatal(tmp_path):
    """Guards the area_bias denominator rather than emitting inf."""
    row, problems = call(write_arm(tmp_path / "a", ref_px=0))
    assert row is None
    assert any("reference_water_pixels_total" in p for p in problems)


def test_missing_iou_row_is_fatal(tmp_path):
    """tau* has no defined value if the iou optimize row is absent."""
    row, problems = call(write_arm(tmp_path / "a", best_metrics=("f1", "mcc")))
    assert row is None
    assert any("'iou' rows" in p for p in problems)
