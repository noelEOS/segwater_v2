#!/usr/bin/env python3
"""Residual semivariogram for Hampyeong Bay -> measured block-size justification.

WHY THIS EXISTS
---------------
The Hampyeong spatial block bootstrap hardcodes BLOCK_PX = 200 (2 km at 10 m/px)
in three scripts (hampyeong_block_bootstrap.py, hampyeong_pooled_ap_iou.py,
hampyeong_new_vs_legacy_block_bootstrap.py) with no site-specific measurement
behind it. The only variogram in the repo
(scripts/analysis/spatial_block_bootstrap.py) was run on Demak/Semarang and
recommended ~6 km blocks from a ~1.8 km residual range. This script measures the
range AT HAMPYEONG so the 2 km choice can be defended or corrected.

METHOD (identical estimator to the Demak Step 0, so the two are comparable)
--------------------------------------------------------------------------
  gamma(h) = 0.5 * mean( (r_i - r_j)^2 ) over pixel pairs at lag ~ h,
  r = |p - y| the absolute residual of the probability against the binary label.
  Pairs are drawn WITHIN a single date, so the lag is purely spatial.
  Practical range = first lag at which gamma reaches 95% of the far-lag sill
  (sill = mean gamma over the far half of the lag range).
  Block-side rule = >= 3x the practical range.

TWO DELIBERATE DIFFERENCES FROM THE DEMAK SCRIPT
------------------------------------------------
  1. METRES_PER_PIXEL = 10.0 EXACTLY. Hampyeong is projected UTM (EPSG:32652)
     with 10 m pixels, so px->m needs no latitude correction. Demak's 9.93 is a
     geographic-grid (EPSG:4326) approximation at -6.9 deg and does NOT apply.
  2. Pixel access goes through the verified Hampyeong loader
     (load_overlap_reference_and_score + the mask-grid replication trick from
     hampyeong_block_bootstrap.block_ids_for_valid_pixels), not the Demak
     run-dir manifest layout, which Hampyeong does not have.

Run:
  /opt/homebrew/Caskroom/miniforge/base/envs/eda/bin/python \
      scripts/evaluation/hampyeong_variogram.py
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

_SCRIPTS = Path(__file__).resolve().parent.parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from inference_overlap_utils import load_overlap_reference_and_score  # noqa: E402
from evaluation.hampyeong_model_comparison import (  # noqa: E402
    DATES,
    EXPECTED_VALID_PIXELS,
    RESOLUTION_ATOL,
    SCENE_IDS,
    gt_path,
    new_pred_path,
    valid_mask_path,
)

DEFAULT_NAS_ROOT = "/Volumes/WD_8tb_RedPlus_NAS_A/MACKBOOK_AIR_M2_BACKUP/Documents/EOS/ACDC"
DEFAULT_RUNS_ROOT = "experiments/hampyeong/runs"
DEFAULT_OUT_DIR = "experiments/hampyeong/evaluation/variogram"

# Hampyeong is EPSG:32652 at exactly 10 m. Verified from the valid mask profile.
METRES_PER_PIXEL = 10.0

# The block bootstrap's current hardcoded choice, for direct comparison.
CURRENT_BLOCK_PX = 200

# Swin-B is the model of record and matches the Demak variogram model choice,
# so the two sites' ranges are estimated from the same architecture's residuals.
DEFAULT_RUN_DIR = (
    "dev_sweep_all_hampyeong_s42_20260716T063710Z_"
    "upernet_swin_base_224_native224_weighted_224_b0_s32"
)


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def valid_pixel_coords(mask_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Global (row, col) per valid pixel, in the row-major order the loader returns.

    Same replication trick as hampyeong_block_bootstrap.block_ids_for_valid_pixels:
    the loader flattens valid pixels of the reference grid in row-major order, so
    np.nonzero on the same mask lines up 1:1 with the returned vectors.
    """
    with rasterio.open(mask_path) as src:
        valid = src.read(1) == 1
    n = int(valid.sum())
    if n != EXPECTED_VALID_PIXELS:
        raise AssertionError(f"valid mask has {n} px, expected {EXPECTED_VALID_PIXELS}")
    rows, cols = np.nonzero(valid)
    return rows.astype(np.int64), cols.astype(np.int64)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nas-root", type=Path, default=Path(DEFAULT_NAS_ROOT))
    ap.add_argument("--runs-root", type=Path, default=Path(DEFAULT_RUNS_ROOT))
    ap.add_argument("--run-dir", default=DEFAULT_RUN_DIR)
    ap.add_argument("--out-dir", type=Path, default=Path(DEFAULT_OUT_DIR))
    ap.add_argument("--subsample", type=int, default=200_000,
                    help="valid pixels sampled per date (Demak Step 0 default)")
    ap.add_argument("--n-pairs", type=int, default=4_000_000,
                    help="random within-date pairs per date (Demak Step 0 default)")
    ap.add_argument("--max-lag-px", type=int, default=300, help="max lag in px (~3 km)")
    ap.add_argument("--n-lag-bins", type=int, default=40)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    mask = valid_mask_path(args.nas_root)
    rows_all, cols_all = valid_pixel_coords(mask)

    print(f"Hampyeong variogram | run {args.run_dir}")
    print(f"{EXPECTED_VALID_PIXELS:,} valid px @ {METRES_PER_PIXEL} m  "
          f"(~{EXPECTED_VALID_PIXELS * METRES_PER_PIXEL**2 / 1e6:.0f} km2)\n")

    samples, per_date = [], []
    for di, date in enumerate(DATES):
        reference = gt_path(args.nas_root, date)
        prediction = new_pred_path(args.runs_root, args.run_dir, date)
        y_true, y_score, diag = load_overlap_reference_and_score(
            reference_path=str(reference),
            prediction_path=str(prediction),
            reference_water_values=[1],
            reference_nodata_values=None,
            resolution_atol=RESOLUTION_ATOL,
            valid_mask_path=str(mask),
            valid_mask_value=1,
        )
        n_valid = diag["valid_pixels_after_all_masks"]
        if n_valid != EXPECTED_VALID_PIXELS:
            raise AssertionError(f"{date}: {n_valid} valid px != {EXPECTED_VALID_PIXELS}")

        resid = np.abs(y_score.astype(np.float32) - y_true.astype(np.float32))
        n_take = min(args.subsample, resid.size)
        sel = rng.choice(resid.size, size=n_take, replace=False)
        samples.append((rows_all[sel], cols_all[sel], resid[sel]))
        per_date.append({"date": date, "n_valid": int(n_valid), "n_sampled": int(n_take),
                         "mean_abs_resid": float(resid.mean())})
        print(f"  {date}: {n_valid:,} valid -> sampled {n_take:,}  "
              f"mean|resid| {resid.mean():.4f}")

    # Empirical variogram, pairs drawn WITHIN each date so lag is purely spatial.
    n_bins = args.n_lag_bins
    max_lag = args.max_lag_px
    sq_sum = np.zeros(n_bins)
    cnt = np.zeros(n_bins, dtype=np.int64)

    for gr, gc, resid in samples:
        a = rng.choice(gr.size, size=args.n_pairs)
        b = rng.choice(gr.size, size=args.n_pairs)
        d_px = np.sqrt((gr[a] - gr[b]) ** 2.0 + (gc[a] - gc[b]) ** 2.0)
        within = d_px < max_lag
        a, b, d_px = a[within], b[within], d_px[within]
        bin_idx = np.minimum((d_px / max_lag * n_bins).astype(int), n_bins - 1)
        sq = (resid[a] - resid[b]) ** 2.0
        np.add.at(sq_sum, bin_idx, sq)
        np.add.at(cnt, bin_idx, 1)

    lag_edges = np.linspace(0, max_lag, n_bins + 1)
    lag_mid_px = 0.5 * (lag_edges[:-1] + lag_edges[1:])
    gamma = np.where(cnt > 0, 0.5 * sq_sum / np.maximum(cnt, 1), np.nan)

    vg = pd.DataFrame({"lag_px": lag_mid_px,
                       "lag_m": lag_mid_px * METRES_PER_PIXEL,
                       "n_pairs": cnt, "gamma": gamma})
    vg.to_csv(args.out_dir / "variogram.csv", index=False)

    finite = vg.dropna()
    sill = float(finite[finite["lag_px"] > max_lag / 2]["gamma"].mean())
    reached = finite[finite["gamma"] >= 0.95 * sill]
    range_px = float(reached["lag_px"].iloc[0]) if not reached.empty else float(max_lag)
    range_m = range_px * METRES_PER_PIXEL

    primary_px = int(np.ceil(3 * range_px / 100.0) * 100)
    current_multiple = CURRENT_BLOCK_PX / range_px if range_px > 0 else float("nan")

    rec = {
        "timestamp_utc": utc_now(),
        "site": "Hampyeong Bay",
        "run_dir": args.run_dir,
        "crs": "EPSG:32652",
        "metres_per_pixel": METRES_PER_PIXEL,
        "n_valid_pixels": EXPECTED_VALID_PIXELS,
        "sill": sill,
        "range_px": range_px,
        "range_m": range_m,
        "range_threshold_frac_of_sill": 0.95,
        "recommended_block_px": primary_px,
        "recommended_block_m": primary_px * METRES_PER_PIXEL,
        "current_block_px": CURRENT_BLOCK_PX,
        "current_block_m": CURRENT_BLOCK_PX * METRES_PER_PIXEL,
        "current_block_as_multiple_of_range": current_multiple,
        "current_block_meets_3x_rule": bool(current_multiple >= 3.0),
        "max_lag_px": max_lag,
        "n_pairs_per_date": args.n_pairs,
        "subsample_per_date": args.subsample,
        "seed": args.seed,
        "per_date": per_date,
        "note": (
            "Practical range = first lag at which the residual semivariance reaches "
            "95% of the far-lag sill. Same estimator and rules as Demak Step 0 in "
            "scripts/analysis/spatial_block_bootstrap.py, so the two sites' ranges "
            "are directly comparable. EPSG:32652, exactly 10 m pixels -- no latitude "
            "correction (unlike Demak's 9.93 m geographic approximation)."
        ),
    }
    (args.out_dir / "block_grid_recommendation.json").write_text(json.dumps(rec, indent=2))

    print(f"\nSill                ~ {sill:.5f}")
    print(f"Practical range     ~ {range_px:.1f} px ({range_m:.0f} m)")
    print(f"3x-rule block       = {primary_px} px ({primary_px * METRES_PER_PIXEL:.0f} m)")
    print(f"Current block       = {CURRENT_BLOCK_PX} px ({CURRENT_BLOCK_PX * METRES_PER_PIXEL:.0f} m)"
          f"  = {current_multiple:.2f}x range  "
          f"[{'MEETS' if current_multiple >= 3 else 'BELOW'} the 3x rule]")
    print(f"\nWrote {args.out_dir / 'variogram.csv'}")
    print(f"Wrote {args.out_dir / 'block_grid_recommendation.json'}")


if __name__ == "__main__":
    main()
