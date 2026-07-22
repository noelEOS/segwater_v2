"""Phase 2 of Option 2 (docs/threshold_calibration/PLAN_option2_platt_scaling.md):
fit per-model Platt scaling p' = sigmoid(a*m + b) on the val-split margin histograms.

The fit minimizes the binned NLL
    sum_bins [ n_water * softplus(-(a*m_c + b)) + n_land * softplus(a*m_c + b) ]
which is statistically equivalent to fitting on the raw pixels (binning error at
4096 bins is negligible; the halved-bin sensitivity below quantifies it).
Class imbalance is deliberately NOT corrected: Platt calibrates the posterior
including the prior.

Decision-wise identity: thresholding p' at 0.5 equals thresholding p at
tau_eq = sigmoid(-b/a); tau_eq is reported next to Option 1's tau*_IoU when
val_thresholds.csv is present.

Guardrails: a > 0 asserted; fit sensitivity to halving the bin count reported
(expect < 1e-3 on a and b).

Pure numpy/scipy; runs locally on the Phase 1 artifacts.

Usage:
    python scripts/evaluation/fit_platt_scaling.py \
        --calib-root outputs/evaluation/val_split_calibration \
        [--out platt_params.csv]  (default: <calib-root>/platt_params.csv)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from margin_hist_common import (  # noqa: E402
    MarginHist, discover_hists, load_margin_hist, sigmoid,
)

ECE_BINS = 15  # matches COARSE_BINS in the repo's calibration machinery


def _softplus(x: np.ndarray) -> np.ndarray:
    return np.logaddexp(0.0, x)


def binned_nll(a: float, b: float, m_c: np.ndarray, n_w: np.ndarray,
               n_l: np.ndarray) -> float:
    """Mean per-pixel NLL of the recalibrated water probability."""
    z = a * m_c + b
    total = float((n_w + n_l).sum())
    return float((n_w * _softplus(-z) + n_l * _softplus(z)).sum() / total)


def fit_platt(h: MarginHist) -> tuple[float, float]:
    m_c = h.bin_centers
    n_w = h.hist_water.astype(np.float64)
    n_l = h.hist_land.astype(np.float64)
    total = (n_w + n_l).sum()

    def objective(params):
        a, b = params
        z = a * m_c + b
        nll = (n_w * _softplus(-z) + n_l * _softplus(z)).sum() / total
        s = sigmoid(z)
        r = ((n_w + n_l) * s - n_w) / total
        return nll, np.array([(m_c * r).sum(), r.sum()])

    res = minimize(objective, x0=np.array([1.0, 0.0]), jac=True, method="L-BFGS-B")
    if not res.success:
        raise RuntimeError(f"{h.name}: Platt fit failed to converge: {res.message}")
    a, b = float(res.x[0]), float(res.x[1])
    assert a > 0, f"{h.name}: fitted a = {a} <= 0 — something is deeply wrong"
    return a, b


def calibration_metrics(h: MarginHist, a: float, b: float) -> dict:
    """Weighted water-probability ECE/MCE/Brier from the binned margins."""
    m_c = h.bin_centers
    n_w = h.hist_water.astype(np.float64)
    n_l = h.hist_land.astype(np.float64)
    n = n_w + n_l
    total = n.sum()
    p = sigmoid(a * m_c + b)

    pbin = np.minimum((p * ECE_BINS).astype(np.int64), ECE_BINS - 1)
    cnt = np.bincount(pbin, weights=n, minlength=ECE_BINS)
    pos = np.bincount(pbin, weights=n_w, minlength=ECE_BINS)
    psum = np.bincount(pbin, weights=n * p, minlength=ECE_BINS)
    nz = cnt > 0
    gap = np.abs(pos[nz] / cnt[nz] - psum[nz] / cnt[nz])
    ece = float((cnt[nz] / total * gap).sum())
    mce = float(gap.max())
    brier = float((n_w * (1.0 - p) ** 2 + n_l * p ** 2).sum() / total)
    return {"ece": ece, "mce": mce, "brier": brier}


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
        m_c, n_w, n_l = h.bin_centers, h.hist_water.astype(float), h.hist_land.astype(float)

        a, b = fit_platt(h)
        a_half, b_half = fit_platt(h.halved())

        before = calibration_metrics(h, 1.0, 0.0)
        after = calibration_metrics(h, a, b)
        nll_before = binned_nll(1.0, 0.0, m_c, n_w, n_l)
        nll_after = binned_nll(a, b, m_c, n_w, n_l)
        tau_eq = float(sigmoid(-b / a))

        meta = h.meta
        rows.append({
            "name": h.name,
            "model": meta.get("model"),
            "arch": meta.get("arch"),
            "encoder": meta.get("encoder"),
            "seed": meta.get("seed"),
            "kind": meta.get("kind"),
            "a": a, "b": b, "tau_eq": tau_eq,
            "nll_before": nll_before, "nll_after": nll_after,
            "ece_before": before["ece"], "ece_after": after["ece"],
            "mce_before": before["mce"], "mce_after": after["mce"],
            "brier_before": before["brier"], "brier_after": after["brier"],
            "delta_a_halfbins": abs(a - a_half),
            "delta_b_halfbins": abs(b - b_half),
            "partial": meta.get("partial", False),
            "checkpoint_resolved": meta.get("checkpoint_resolved", ""),
            "checkpoint_step": meta.get("checkpoint_step", ""),
            "precision": meta.get("precision"),
        })
        r = rows[-1]
        sens_flag = "" if max(r["delta_a_halfbins"], r["delta_b_halfbins"]) < 1e-3 \
            else "  <-- bin-sensitivity above 1e-3"
        print(f"{h.name}: a={a:.4f} b={b:+.4f} tau_eq={tau_eq:.4f} | "
              f"NLL {nll_before:.5f}->{nll_after:.5f} ECE {before['ece']:.5f}->"
              f"{after['ece']:.5f}{sens_flag}"
              + ("  [PARTIAL]" if r["partial"] else ""))

    df = pd.DataFrame(rows)

    # Overlay Option 1's tau*_IoU when available: NLL-optimal vs IoU-optimal boundary.
    thr_csv = args.calib_root / "val_thresholds.csv"
    if thr_csv.exists():
        thr = pd.read_csv(thr_csv)[["name", "tau_star_iou", "iou_at_tau_star", "iou_at_0.5"]]
        df = df.merge(thr, on="name", how="left")
        df["tau_eq_minus_tau_star_iou"] = df["tau_eq"] - df["tau_star_iou"]
    else:
        print(f"(no {thr_csv}; run select_val_thresholds.py for the tau overlay)")

    out = args.out or (args.calib_root / "platt_params.csv")
    df.to_csv(out, index=False)
    print(f"\nWrote {out} ({len(df)} rows)")
    if df["partial"].any():
        print("WARNING: some rows come from --max-batches partial histograms; "
              "do not quote them.")


if __name__ == "__main__":
    main()
