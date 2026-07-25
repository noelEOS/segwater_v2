"""Collect the Demak gate table from the per-variant AUC-ROC scoring outputs.

Statistic rules (docs/RUNBOOK_demak_vm_eval.md -- quote nothing else):
  ROC-AUC     mean `roc_auc` over the 6 pairs            auc_per_pair.csv
  tau*, IoU@tau*  best_threshold / best_metric_mean       threshold_best_summary.csv
                  at optimize_metric == iou
  IoU@0.5, F1@0.5 iou_mean / f1_mean at threshold == 0.5  threshold_summary.csv
                  and estimate_type == macro_average_across_pairs_at_threshold
  LOO-IoU     iou_mean at optimize_metric == iou          loo_threshold_summary.csv
  area_bias   prediction_water_pixels_total / reference_water_pixels_total
              at threshold 0.5 (FINDINGS 4.2; 1.00 = unbiased, <1 under-predicts)
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# Defaults are the VM-side layout (docs/RUNBOOK_demak_vm_eval.md); override with
# --root/--out to run against a pulled copy on the Mac.
DEFAULT_ROOT = Path("/home/noel/pairbased_vm_eval")
DEFAULT_OUT = Path("/home/noel/demak_gate_summary.csv")
SEEDS = ["s19", "s42", "s58"]
ARMS = ["best", "last", "swa5"]


def collect(root: Path, arm: str, seed: str) -> dict | None:
    d = root / ("demak_gate_%s_%s_s32" % (arm, seed))
    if not d.is_dir():
        print("  !! missing eval dir: %s" % d.name)
        return None
    row = {"seed": seed, "arm": arm}

    auc = pd.read_csv(d / "auc_per_pair.csv")
    ok = auc[auc.get("status", "success") == "success"] if "status" in auc else auc
    row["n_pairs_auc"] = int(len(ok))
    row["roc_auc"] = float(ok["roc_auc"].mean())

    ts = pd.read_csv(d / "threshold_summary.csv")
    t50 = ts[(np.isclose(ts.threshold, 0.5))
             & (ts.estimate_type == "macro_average_across_pairs_at_threshold")]
    assert len(t50) == 1, "%s: expected 1 row at thr 0.5, got %d" % (d.name, len(t50))
    r = t50.iloc[0]
    row["iou_at_050"] = float(r.iou_mean)
    row["f1_at_050"] = float(r.f1_mean)
    row["precision_at_050"] = float(r.precision_mean)
    row["recall_at_050"] = float(r.recall_mean)
    row["area_bias_at_050"] = (float(r.prediction_water_pixels_total)
                               / float(r.reference_water_pixels_total))

    tb = pd.read_csv(d / "threshold_best_summary.csv")
    tb = tb[tb.optimize_metric == "iou"]
    assert len(tb) == 1, "%s: expected 1 best-iou row, got %d" % (d.name, len(tb))
    row["tau_star"] = float(tb.iloc[0].best_threshold)
    row["iou_at_tau_star"] = float(tb.iloc[0].best_metric_mean)

    loo = pd.read_csv(d / "loo_threshold_summary.csv")
    loo = loo[loo.optimize_metric == "iou"]
    assert len(loo) == 1, "%s: expected 1 loo-iou row, got %d" % (d.name, len(loo))
    row["loo_iou"] = float(loo.iloc[0].iou_mean)
    return row


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT,
                    help="dir holding the per-variant demak_gate_<arm>_<seed>_s32 eval dirs")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    rows = [r for seed in SEEDS for arm in ARMS
            if (r := collect(args.root, arm, seed)) is not None]
    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    print("Wrote %s (%d rows)\n" % (args.out, len(df)))

    cols = ["seed", "arm", "iou_at_050", "tau_star", "iou_at_tau_star",
            "area_bias_at_050", "roc_auc", "loo_iou"]
    print(df[cols].to_string(index=False,
                             float_format=lambda v: "%.4f" % v))
    print("\n=== MEAN +/- SEED-SD (ddof=1) PER ARM ===")
    g = df.groupby("arm")[["iou_at_050", "tau_star", "area_bias_at_050",
                           "roc_auc", "loo_iou"]]
    summ = g.agg(["mean", lambda s: s.std(ddof=1)])
    summ.columns = ["%s_%s" % (a, "sd" if "lambda" in b else b)
                    for a, b in summ.columns]
    print(summ.to_string(float_format=lambda v: "%.4f" % v))


if __name__ == "__main__":
    main()
