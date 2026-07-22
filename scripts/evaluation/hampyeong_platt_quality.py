"""Option 2 Phase 4 at Hampyeong: probability-quality before/after Platt scaling.

For each of the 8 new architectures (3-seed MEAN probability rasters, matching
hampyeong_pooled_ap_iou), pooled over the 5 clean dates:
  - ECE / MCE (15 fixed-width bins, count-weighted, water-class reliability —
    same definition as fit_platt_scaling.py so val and Hampyeong are comparable),
  - Brier, mean NLL,
computed on the raw probabilities and on the Platt-recalibrated probabilities
p' = sigmoid(a*logit(p) + b), with per-model ENSEMBLE (a, b) fitted on the
training validation split (platt_params.csv). This is the out-of-distribution
transfer test: does val-fitted calibration hold at a macrotidal site?

Also runs the decision-identity cross-check required by the plan: pooled
IoU(p' >= 0.5) must equal pooled IoU(p >= tau_eq) with tau_eq = sigmoid(-b/a).

Pixel access is the verified loader of hampyeong_pooled_ap_iou (identical valid
pixels). Rasters on disk are untouched; the transform is applied in memory.

Run: /opt/homebrew/Caskroom/miniforge/base/envs/eda/bin/python \
        scripts/evaluation/hampyeong_platt_quality.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_SCRIPTS = Path(__file__).resolve().parent.parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from evaluation.hampyeong_pooled_ap_iou import (  # noqa: E402
    CKPT_KEY_BY_ARCH_EXT,
    DATES_5,
    DEFAULT_NAS_ROOT,
    DEFAULT_OUT_DIR,
    DEFAULT_RUNS_ROOT,
    DEFAULT_TOPLEVEL_ROOT,
    NEW_MODEL_ORDER,
    load_model_scores,
    valid_mask_path,
)

ECE_BINS = 15   # matches fit_platt_scaling.py and the repo's coarse-bin convention
EPS = 1e-7      # probability clamp for logit/NLL (float32 rasters saturate at 1.0)


def platt(p: np.ndarray, a: float, b: float) -> np.ndarray:
    p = np.clip(p.astype(np.float64), EPS, 1.0 - EPS)
    z = a * (np.log(p) - np.log1p(-p)) + b
    return 1.0 / (1.0 + np.exp(-z))


def quality(y: np.ndarray, p: np.ndarray) -> dict:
    """Count-weighted water-probability ECE/MCE + Brier + mean NLL."""
    p64 = np.clip(p.astype(np.float64), EPS, 1.0 - EPS)
    idx = np.minimum((p64 * ECE_BINS).astype(np.int64), ECE_BINS - 1)
    cnt = np.bincount(idx, minlength=ECE_BINS).astype(np.float64)
    pos = np.bincount(idx, weights=y, minlength=ECE_BINS)
    psum = np.bincount(idx, weights=p64, minlength=ECE_BINS)
    nz = cnt > 0
    gap = np.abs(pos[nz] / cnt[nz] - psum[nz] / cnt[nz])
    return {
        "ece": float((cnt[nz] / cnt.sum() * gap).sum()),
        "mce": float(gap.max()),
        "brier": float(np.mean((p64 - y) ** 2)),
        "nll": float(-np.mean(y * np.log(p64) + (1.0 - y) * np.log1p(-p64))),
    }


def iou_at(y: np.ndarray, p: np.ndarray, thr: float) -> float:
    pred = p >= thr
    tp = float(np.sum(pred & (y == 1)))
    fp = float(np.sum(pred & (y == 0)))
    fn = float(np.sum(~pred & (y == 1)))
    return tp / max(tp + fp + fn, 1.0)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nas-root", type=Path, default=Path(DEFAULT_NAS_ROOT))
    ap.add_argument("--runs-root", type=Path, default=DEFAULT_RUNS_ROOT)
    ap.add_argument("--toplevel-root", type=Path, default=DEFAULT_TOPLEVEL_ROOT)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--platt-params", type=Path,
                    default=Path("outputs/evaluation/val_split_calibration/platt_params.csv"))
    args = ap.parse_args()
    if not args.nas_root.exists():
        raise SystemExit(f"NAS root not found: {args.nas_root}")

    params = pd.read_csv(args.platt_params).set_index("name")
    mask = valid_mask_path(args.nas_root)

    rows = []
    for arch in NEW_MODEL_ORDER:
        name = f"{CKPT_KEY_BY_ARCH_EXT[arch]}_ensemble"
        if name not in params.index:
            raise SystemExit(f"No row named {name} in {args.platt_params}")
        a, b = float(params.loc[name, "a"]), float(params.loc[name, "b"])
        tau_eq = 1.0 / (1.0 + np.exp(b / a))

        ys, ps = [], []
        for date in DATES_5:
            y, p = load_model_scores("new", arch, date, args.runs_root,
                                     args.toplevel_root, args.nas_root, mask)
            ys.append(y)
            ps.append(p)
        y = np.concatenate(ys)
        p = np.concatenate(ps)
        p_cal = platt(p, a, b)

        before = quality(y, p)
        after = quality(y, p_cal)

        # decision-identity cross-check: p' >= 0.5 <=> p >= tau_eq
        iou_platt05 = iou_at(y, p_cal, 0.5)
        iou_tau_eq = iou_at(y, p, tau_eq)
        if abs(iou_platt05 - iou_tau_eq) > 1e-9:
            raise AssertionError(
                f"{arch}: decision identity violated: IoU(p'>=0.5)={iou_platt05!r}"
                f" vs IoU(p>=tau_eq)={iou_tau_eq!r}")

        rows.append({
            "arch": arch, "a": a, "b": b, "tau_eq": tau_eq,
            **{f"{k}_before": v for k, v in before.items()},
            **{f"{k}_after": v for k, v in after.items()},
            "iou_at_tau_eq": iou_tau_eq, "iou_at_0.5": iou_at(y, p, 0.5),
            "n_pixels": int(y.size), "n_dates": len(DATES_5),
            "platt_source": name,
        })
        r = rows[-1]
        print(f"{arch:12s} a={a:.3f} b={b:+.3f} | ECE {before['ece']:.4f}->{after['ece']:.4f}  "
              f"Brier {before['brier']:.4f}->{after['brier']:.4f}  "
              f"NLL {before['nll']:.4f}->{after['nll']:.4f} | "
              f"IoU@tau_eq={iou_tau_eq:.4f} (identity OK)")

    df = pd.DataFrame(rows)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / "hampyeong_platt_quality_5date.csv"
    df.to_csv(out, index=False)
    print(f"\nWrote {out} ({len(df)} rows; pooled over {len(DATES_5)} dates, "
          f"val-fitted ensemble Platt params)")


if __name__ == "__main__":
    main()
