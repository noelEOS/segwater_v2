"""Assemble the four ship-campaign gate CSVs into one decision table.

Produces `SHIP_DECISION_TABLE.csv` (one row per arm, one column block per gate)
plus a `SHIP_DECISION_SUMMARY.md` with the per-variant 3-seed aggregates.

Reads only what the scoring agents wrote; computes no metrics of its own beyond
aggregation, so a discrepancy here means a discrepancy upstream.

Two uncertainty kinds appear and are NOT interchangeable -- this script keeps
them in separate columns and labels both:
  * `*_hac_se`  -- within-arm Newey-West SE from one arm's time series;
  * `*_seed_sd` -- across-seed SD (ddof=1) over s19/s42/s58 for one variant.

Usage:
    python scripts/evaluation/vm/ship/consolidate_ship_results.py [--root DIR]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

SEEDS = ["s19", "s42", "s58"]
VARIANTS = ["best", "last"]

# Reference values, canonical spec, 206-scene window, thr 0.5. Context only --
# never recomputed here.
REFERENCES = {
    "S2 optical anchor": (314.0, 36.5),
    "Landsat anchor": (293.1, 31.4),
    "registered chip-based": (340.4, 30.8),
}


def _read(p: Path, what: str):
    if not p.exists():
        print("  MISSING (%s): %s" % (what, p))
        return None
    return pd.read_csv(p)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path,
                    default=Path.home() / "workspace/results/ship_decision_2026-07")
    a = ap.parse_args()
    R = a.root

    rows = {(s, v): {"seed": s, "variant": v, "arm": "swinb_%s_mx630s2_%s" % (s, v)}
            for s in SEEDS for v in VARIANTS}

    # --- Demak concurrent gate (the acceptance test) ------------------------
    g = _read(R / "demak_gate/demak_gate_ship_summary.csv", "demak gate")
    if g is not None:
        for _, r in g.iterrows():
            k = (r["seed"], r["variant"])
            if k in rows:
                rows[k].update(gate_iou=r.get("iou_at_0p5"), gate_tau=r.get("tau_star"),
                               gate_iou_tau=r.get("iou_at_tau_star"),
                               gate_area_bias=r.get("area_bias"), gate_auc=r.get("roc_auc"))

    # --- Hampyeong (corroboration only; cannot gate) ------------------------
    h = _read(R / "hampyeong/hampyeong_ship_per_date_metrics.csv", "hampyeong")
    if h is not None:
        for (s, v), sub in h.groupby(["seed", "variant"]):
            s = "s%s" % s if not str(s).startswith("s") else str(s)
            if (s, v) in rows:
                rows[(s, v)]["hamp_mean3_iou"] = sub["iou"].mean()

    # --- Narrabeen SDS at thr 0.5 ------------------------------------------
    n = _read(R / "narrabeen/sds_narrabeen_ship_msl.csv", "narrabeen sds")
    if n is not None:
        t = n[(n.threshold - 0.5).abs() < 1e-9]
        for (s, v), sub in t.groupby(["seed", "variant"]):
            s = "s%s" % s if not str(s).startswith("s") else str(s)
            if (s, v) in rows:
                rows[(s, v)].update(
                    sds_rmse_mean=sub["rmse"].mean(), sds_bias_mean=sub["bias"].mean(),
                    sds_std_mean=sub["std"].mean(),
                    **{"sds_rmse_s%d" % st: sub[sub.stride == st]["rmse"].mean()
                       for st in (8, 32, 112) if (sub.stride == st).any()})

    # --- Demak trend, both strides -----------------------------------------
    for stride in (32, 112):
        t = _read(R / ("demak_trend/demak_full_ship_trend_s%d.csv" % stride), "trend s%d" % stride)
        if t is None:
            continue
        for _, r in t.iterrows():
            k = (r["seed"], r["variant"])
            if k in rows:
                rows[k]["trend_s%d" % stride] = r.get("slope_ha_yr")
                rows[k]["trend_s%d_hac_se" % stride] = r.get("hac_se")

    df = pd.DataFrame([rows[(s, v)] for v in VARIANTS for s in SEEDS])
    out = R / "SHIP_DECISION_TABLE.csv"
    df.to_csv(out, index=False)
    print("\nwrote %s (%d arms)" % (out, len(df)))
    print(df.to_string(index=False))

    # --- per-variant 3-seed aggregates -------------------------------------
    lines = ["# Ship decision — consolidated\n",
             "Per-variant aggregates are **3-seed mean ± SD (ddof=1)** across "
             "s19/s42/s58. This is a different quantity from a per-arm HAC SE "
             "(within-arm, from one time series); the two are not "
             "interchangeable and are never combined.\n"]
    num = [c for c in df.columns
           if c not in ("seed", "variant", "arm")
           and pd.api.types.is_numeric_dtype(df[c]) and df[c].notna().any()]
    if not num:
        # Mid-campaign: no gate CSV has landed yet. Emit the arm skeleton and
        # stop rather than raising -- this script is meant to be safe to run
        # repeatedly while results accumulate.
        print("\n(no gate results yet -- skeleton written, aggregates skipped)")
        return
    agg = df.groupby("variant")[num].agg(["mean", lambda x: x.std(ddof=1)])
    agg.columns = ["%s_%s" % (c, "mean" if k == "mean" else "seed_sd")
                   for c, k in agg.columns]
    print("\n=== per-variant 3-seed aggregates ===")
    print(agg.T.to_string())
    lines.append("```\n" + agg.T.to_string() + "\n```\n")

    lines.append("\n## Trend references (context; not recomputed)\n")
    for k, (v, se) in REFERENCES.items():
        lines.append("- %s: %+.1f ± %.1f ha/yr" % (k, v, se))
    (R / "SHIP_DECISION_SUMMARY.md").write_text("\n".join(lines) + "\n")
    print("\nwrote %s" % (R / "SHIP_DECISION_SUMMARY.md"))


if __name__ == "__main__":
    main()
