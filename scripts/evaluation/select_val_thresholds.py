"""Phase 2 of Option 1 (docs/threshold_calibration/PLAN_option1_val_threshold_selection.md):
select per-model decision thresholds tau* on the training validation split.

Selection rule (fixed ex ante, plan section 2.3): tau* = argmax water-class IoU on
the grid 0.05..0.95 step 0.005, ties to the LOWEST tau (mirrors
select_best_thresholds in scripts/evaluate_indonesia_inference_run_aucroc.py).
Each grid tau is evaluated exactly via the margin histograms: the cut logit(tau)
is snapped to the nearest bin edge (bin width ~0.0098 in margin space, i.e.
~0.0024 in probability near 0.5 — finer than the grid step).

Also reported per model: F1-optimal tau (sensitivity), the unrestricted
all-edges argmax (sensitivity), IoU@0.5, curve flatness around tau*, and the
checkpoint provenance carried through from Phase 1.

Pure numpy; runs locally on the Phase 1 artifacts.

Usage:
    python scripts/evaluation/select_val_thresholds.py \
        --calib-root outputs/evaluation/val_split_calibration \
        [--out val_thresholds.csv]  (default: <calib-root>/val_thresholds.csv)

Writes additionally, per histogram dir: threshold_curve.csv (metrics at every
grid tau) for figures.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from margin_hist_common import (  # noqa: E402
    MarginHist, confusion_curves, discover_hists, edge_index_for_tau,
    load_margin_hist,
)

TAU_GRID = np.round(np.arange(0.05, 0.95 + 1e-9, 0.005), 3)  # 181 values, ex ante
FLATNESS_WINDOW = 0.05


def select_for_hist(h: MarginHist) -> tuple[dict, pd.DataFrame]:
    curves = confusion_curves(h)
    edge_idx = np.array([edge_index_for_tau(h, t) for t in TAU_GRID])
    iou_grid = curves["iou_water"][edge_idx]
    f1_grid = curves["f1_water"][edge_idx]
    miou_grid = curves["miou"][edge_idx]
    mcc_grid = curves["mcc"][edge_idx]

    # argmax with ties to the LOWEST tau: argmax returns the first maximum and
    # TAU_GRID is ascending.
    i_iou = int(np.argmax(iou_grid))
    i_f1 = int(np.argmax(f1_grid))
    tau_star = float(TAU_GRID[i_iou])

    # unrestricted argmax over every bin edge inside the grid range (sensitivity)
    lo_k = edge_index_for_tau(h, TAU_GRID[0])
    hi_k = edge_index_for_tau(h, TAU_GRID[-1])
    k_cont = lo_k + int(np.argmax(curves["iou_water"][lo_k:hi_k + 1]))
    tau_cont = float(curves["tau"][k_cont])

    # operating point 0.5 (exact: margin 0 is a bin edge)
    k05 = edge_index_for_tau(h, 0.5)
    assert abs(curves["edge_margin"][k05]) < 1e-9, "0.5 must fall on a bin edge"
    iou05 = float(curves["iou_water"][k05])
    miou05 = float(curves["miou"][k05])

    # flatness: worst IoU drop on grid taus within +/- 0.05 of tau*
    near = np.abs(TAU_GRID - tau_star) <= FLATNESS_WINDOW + 1e-9
    flat_drop = float(iou_grid[i_iou] - iou_grid[near].min())

    meta = h.meta
    row = {
        "name": h.name,
        "model": meta.get("model"),
        "arch": meta.get("arch"),
        "encoder": meta.get("encoder"),
        "seed": meta.get("seed"),
        "kind": meta.get("kind"),
        "tau_star_iou": tau_star,
        "iou_at_tau_star": float(iou_grid[i_iou]),
        "tau_star_f1": float(TAU_GRID[i_f1]),
        "f1_at_tau_star_f1": float(f1_grid[i_f1]),
        "tau_star_iou_alledges": tau_cont,
        "iou_at_tau_star_alledges": float(curves["iou_water"][k_cont]),
        "iou_at_0.5": iou05,
        "miou_at_0.5": miou05,
        "delta_iou_tau_star_vs_0.5": float(iou_grid[i_iou]) - iou05,
        "flatness_iou_drop_pm0.05": flat_drop,
        "n_water": int(h.hist_water.sum()),
        "n_land": int(h.hist_land.sum()),
        "n_ignored": h.n_ignored,
        "partial": meta.get("partial", False),
        "checkpoint_resolved": meta.get("checkpoint_resolved", ""),
        "checkpoint_step": meta.get("checkpoint_step", ""),
        "checkpoint_val_miou_stored": meta.get("checkpoint_val_miou_stored", ""),
        "precision": meta.get("precision"),
    }

    curve_df = pd.DataFrame({
        "tau": TAU_GRID,
        "iou_water": iou_grid,
        "f1_water": f1_grid,
        "miou": miou_grid,
        "mcc": mcc_grid,
    })

    # sanity: histogram-derived numbers must reproduce Phase 1's meta values
    for key, val in (("iou_water@0.5", iou05), ("miou@0.5", miou05)):
        if key in meta and not np.isclose(meta[key], val, atol=1e-12):
            raise AssertionError(f"{h.name}: {key} mismatch meta={meta[key]} here={val}")
    return row, curve_df


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--calib-root", type=Path,
                    default=Path("outputs/evaluation/val_split_calibration"))
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    hist_dirs = discover_hists(args.calib_root)
    if not hist_dirs:
        raise SystemExit(f"No hist_margin.npz dirs under {args.calib_root}")

    rows = []
    for d in hist_dirs:
        h = load_margin_hist(d)
        row, curve_df = select_for_hist(h)
        curve_df.to_csv(d / "threshold_curve.csv", index=False)
        rows.append(row)
        stored = row["checkpoint_val_miou_stored"]
        stored_s = f"{stored:.4f}" if isinstance(stored, float) else "n/a"
        print(f"{h.name}: tau*_iou={row['tau_star_iou']:.3f} "
              f"iou@tau*={row['iou_at_tau_star']:.6f} iou@0.5={row['iou_at_0.5']:.6f} "
              f"(delta {row['delta_iou_tau_star_vs_0.5']:+.6f}); "
              f"miou@0.5={row['miou_at_0.5']:.6f} vs stored val mIoU {stored_s}"
              + ("  [PARTIAL]" if row["partial"] else ""))

    df = pd.DataFrame(rows)
    out = args.out or (args.calib_root / "val_thresholds.csv")
    df.to_csv(out, index=False)
    print(f"\nWrote {out} ({len(df)} rows)")
    if df["partial"].any():
        print("WARNING: some rows come from --max-batches partial histograms; "
              "do not quote them.")


if __name__ == "__main__":
    main()
