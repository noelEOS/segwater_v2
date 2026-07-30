"""Collapse the per-arm Demak-gate AUC-ROC scorer outputs into one summary CSV.

One row per arm, matching the Swin-B `ship` campaign's
`demak_gate_ship_summary.csv` column set exactly, so the two campaigns'
gate tables are directly comparable:

    seed, variant, iou_at_0p5, tau_star, iou_at_tau_star, area_bias, roc_auc, n_pairs

WHERE EACH NUMBER COMES FROM (all are read, none recomputed from rasters)
------------------------------------------------------------------------
* ``roc_auc``          <- ``metrics_summary.csv`` : ``roc_auc_mean``
                          (macro-average across pairs, threshold-free)
* ``tau_star``         <- ``threshold_best_summary.csv``, the ``iou`` row's
                          ``best_threshold``. IoU is the gate's optimize metric;
                          the f1/mcc rows are deliberately ignored (mcc's optimum
                          sits a step lower and would give a different tau*).
* ``iou_at_tau_star``  <- same row's ``best_metric_mean``
* ``iou_at_0p5``       <- ``threshold_summary.csv`` : ``iou_mean`` at threshold 0.5
* ``area_bias``        <- ``prediction_water_pixels_total`` /
                          ``reference_water_pixels_total`` at threshold 0.5
                          (1.00 = unbiased; the gate's health floor is ~0.9)
* ``n_pairs``          <- ``metrics_summary.csv`` : ``n_pairs``, cross-checked
                          against ``n_success_pairs``

GUARDS (all fatal — a gate table that quietly drops an arm is worse than none)
  * the threshold grid must actually contain 0.5 (float-tolerant match), else the
    ``iou_at_0p5`` column would silently come from a neighbouring threshold;
  * ``n_pairs == n_success_pairs == --expect-pairs``, so a scorer that skipped a
    pair under ``missing_prediction_policy: record_and_continue`` cannot pass;
  * ``model_name`` inside every CSV must match the arm the directory name claims,
    which catches a mis-pointed config;
  * every requested arm must be present.

Usage:
    python scripts/evaluation/vm/ship/summarize_ship_gate.py \\
        --raw-dir ~/workspace/results/ship_decision_cnxb_2026-07/demak_gate/raw \\
        --tag cnxb --out .../demak_gate/demak_gate_cnxb_summary.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from naming import atomic_to_csv  # noqa: E402

SEEDS = ["s19", "s42", "s58"]
VARIANTS = ["best", "last"]


def one_arm(d: Path, tag: str, seed: str, variant: str, expect_pairs: int,
            problems: list[str]) -> dict | None:
    """Read one arm's scorer output dir into a summary row (None on failure)."""
    arm = "%s/%s" % (seed, variant)
    try:
        ms = pd.read_csv(d / "metrics_summary.csv")
        tb = pd.read_csv(d / "threshold_best_summary.csv")
        ts = pd.read_csv(d / "threshold_summary.csv")
    except FileNotFoundError as exc:
        problems.append("%s: %s" % (arm, exc))
        return None

    expect_model = "%s_%s_%s" % (tag, seed, variant)
    for name, df in (("metrics_summary", ms), ("threshold_best_summary", tb),
                     ("threshold_summary", ts)):
        got = set(df.model_name.unique())
        if got != {expect_model}:
            problems.append("%s: %s model_name=%s, expected {%r}"
                            % (arm, name, sorted(got), expect_model))
            return None

    n_pairs = int(ms.n_pairs.iloc[0])
    n_ok = int(ms.n_success_pairs.iloc[0])
    if not (n_pairs == n_ok == expect_pairs):
        problems.append("%s: n_pairs=%d n_success=%d, expected %d"
                        % (arm, n_pairs, n_ok, expect_pairs))
        return None

    # tau* from the IoU row specifically -- see module docstring.
    iou_row = tb[tb.optimize_metric == "iou"]
    if len(iou_row) != 1:
        problems.append("%s: %d 'iou' rows in threshold_best_summary, expected 1"
                        % (arm, len(iou_row)))
        return None
    iou_row = iou_row.iloc[0]

    # threshold 0.5 must be ON the grid; never take a nearest neighbour.
    at05 = ts[(ts.threshold - 0.5).abs() < 1e-9]
    if len(at05) != 1:
        problems.append("%s: %d rows at threshold 0.5 (grid must contain it)"
                        % (arm, len(at05)))
        return None
    at05 = at05.iloc[0]

    ref_px = float(at05.reference_water_pixels_total)
    if ref_px <= 0:
        problems.append("%s: reference_water_pixels_total=%r" % (arm, ref_px))
        return None

    return dict(
        seed=seed, variant=variant,
        iou_at_0p5=float(at05.iou_mean),
        tau_star=float(iou_row.best_threshold),
        iou_at_tau_star=float(iou_row.best_metric_mean),
        area_bias=float(at05.prediction_water_pixels_total) / ref_px,
        roc_auc=float(ms.roc_auc_mean.iloc[0]),
        n_pairs=n_pairs,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw-dir", type=Path, required=True,
                    help="dir holding one <gate>_<tag>_<seed>_<variant>_aucroc "
                         "subdir per arm")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--variants", nargs="+", default=VARIANTS,
                    choices=["best", "last", "swa5"])
    ap.add_argument("--expect-pairs", type=int, default=6,
                    help="S1/S2 concurrent pairs the gate must have scored "
                         "(default: %(default)s)")
    a = ap.parse_args()

    rows, problems = [], []
    for seed in SEEDS:
        for variant in a.variants:
            d = a.raw_dir / ("demak_gate_%s_%s_%s_aucroc" % (a.tag, seed, variant))
            if not d.is_dir():
                problems.append("%s/%s: missing scorer dir %s" % (seed, variant, d))
                continue
            r = one_arm(d, a.tag, seed, variant, a.expect_pairs, problems)
            if r:
                rows.append(r)

    n_expected = len(SEEDS) * len(a.variants)
    if rows:
        df = pd.DataFrame(rows)
        a.out.parent.mkdir(parents=True, exist_ok=True)
        atomic_to_csv(df, a.out, index=False)
        print(df.to_string(index=False))
        print("\nwrote %s (%d rows)" % (a.out, len(df)))

    if len(rows) != n_expected:
        problems.append("got %d arms, expected %d" % (len(rows), n_expected))
    if problems:
        print("\n=== %d PROBLEM(S) ===" % len(problems))
        for p in problems:
            print("  " + p)
        raise SystemExit(1)
    print("=== ALL GATE SUMMARY CHECKS PASSED ===")


if __name__ == "__main__":
    main()
