"""Figure: water-IoU vs decision threshold on the val split, per architecture.

Small-multiples grid (one panel per architecture): the 3 seed curves as thin
gray lines, the 3-seed ensemble as the single accented line, with the
val-selected tau* (ensemble) and the 0.5 deployment operating point marked.
Val-side half of the Option 1 Phase 5 figure (the Hampyeong overlay is added
once the deferred Hampyeong re-scoring runs).

Reads the threshold_curve.csv files + val_thresholds.csv written by
select_val_thresholds.py.

Usage:
    python scripts/evaluation/plot_val_threshold_curves.py \
        --calib-root outputs/evaluation/val_split_calibration \
        [--out <calib-root>/val_threshold_curves.png]
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ACCENT = "#1f4e79"   # ensemble (matches the repo's reliability-figure blue)
SEED_GRAY = "#b0b0b0"
TAU_STAR = "#c05020"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--calib-root", type=Path,
                    default=Path("outputs/evaluation/val_split_calibration"))
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    thr = pd.read_csv(args.calib_root / "val_thresholds.csv")
    models = list(dict.fromkeys(thr["model"].dropna()))
    ncols = 3
    nrows = math.ceil(len(models) / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(3.6 * ncols, 2.9 * nrows),
                             sharex=True)
    axes = axes.ravel()

    for ax, model in zip(axes, models):
        rows = thr[thr["model"] == model]
        ens = rows[rows["kind"] == "ensemble"]
        for _, r in rows[rows["kind"] == "seed"].iterrows():
            curve = pd.read_csv(args.calib_root / r["name"] / "threshold_curve.csv")
            ax.plot(curve["tau"], curve["iou_water"], color=SEED_GRAY,
                    lw=0.9, zorder=1)
        if len(ens):
            r = ens.iloc[0]
            curve = pd.read_csv(args.calib_root / r["name"] / "threshold_curve.csv")
            ax.plot(curve["tau"], curve["iou_water"], color=ACCENT, lw=1.8, zorder=3)
            ax.axvline(r["tau_star_iou"], color=TAU_STAR, lw=1.0, ls="--", zorder=2)
            ax.annotate(f"τ* = {r['tau_star_iou']:.3f}", xy=(r["tau_star_iou"], 0.02),
                        xycoords=("data", "axes fraction"), fontsize=7,
                        color=TAU_STAR, ha="left", va="bottom", rotation=90)
        ax.axvline(0.5, color="#888888", lw=0.8, ls=":", zorder=2)
        ax.set_title(model, fontsize=8)
        ax.grid(True, lw=0.3, alpha=0.4)
        ax.tick_params(labelsize=7)
    for ax in axes[len(models):]:
        ax.set_axis_off()
    for ax in axes[:len(models)]:
        ax.set_xlabel("decision threshold τ", fontsize=8)
    for i in range(0, len(models), ncols):
        axes[i].set_ylabel("water IoU (val)", fontsize=8)

    handles = [plt.Line2D([], [], color=SEED_GRAY, lw=0.9, label="seeds (s19/s42/s58)"),
               plt.Line2D([], [], color=ACCENT, lw=1.8, label="3-seed ensemble"),
               plt.Line2D([], [], color=TAU_STAR, lw=1.0, ls="--", label="τ* (val, ensemble)"),
               plt.Line2D([], [], color="#888888", lw=0.8, ls=":", label="0.5 operating point")]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=8, frameon=False)
    fig.suptitle("Val-split water IoU vs decision threshold "
                 "(selection data for τ*; test sites untouched)", fontsize=11)
    fig.tight_layout(rect=[0, 0.04, 1, 0.96])

    out = args.out or (args.calib_root / "val_threshold_curves.png")
    fig.savefig(out, dpi=200)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
