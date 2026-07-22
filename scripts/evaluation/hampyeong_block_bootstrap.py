"""Spatial block bootstrap for the ConvNeXtV2 vs Swin-B paired IoU difference.

Bounds ONE nuisance: does each date's architecture difference survive spatial
autocorrelation, or is it an artifact of a few large correlated patches? This is a
robustness check on estimand E1 (these fixed scenes), NOT a generalization claim.

Design (deliberately minimal, per the approved plan):
  - 2 km blocks only (no block-size sweep).
  - PAIRED: within each bootstrap iteration one block-multiplicity vector is drawn and
    reused for BOTH architectures, so the difference cancels shared spatial variance
    (the shared-m_vec idea from scripts/analysis/spatial_block_bootstrap.py).
  - PER DATE x PER SEED: run separately for each of the 3 seeds. Seed spread stays
    descriptive and separate -- seeds are NOT folded into the resampling (that would be
    false precision at n=3 seeds).
  - Null control: two identical rasters must yield a paired-diff CI straddling 0.

Reuses the verified loader hampyeong_model_comparison.load_pair for reference/valid-mask
masking, and derives block ids from the valid-mask reference grid (the AOI is the
reference frame, so block ids align 1:1 with load_pair's valid-pixel ordering).

Run: /opt/homebrew/Caskroom/miniforge/base/envs/eda/bin/python <this file>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

_SCRIPTS = Path(__file__).resolve().parent.parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from evaluation.hampyeong_model_comparison import (  # noqa: E402
    DATES,
    EXPECTED_VALID_PIXELS,
    NEW_RUNS,
    gt_path,
    load_pair,
    new_pred_path,
    valid_mask_path,
)

DEFAULT_NAS_ROOT = "/Volumes/WD_8tb_RedPlus_NAS_A/MACKBOOK_AIR_M2_BACKUP/Documents/EOS/ACDC"
DEFAULT_RUNS_ROOT = "experiments/hampyeong/runs"
DEFAULT_OUT_DIR = "experiments/hampyeong/evaluation"

ARCH_A = "ConvNeXtV2"
ARCH_B = "Swin-B"
SEEDS = (19, 42, 58)
BLOCK_PX = 200          # 2 km at 10 m/px
N_BOOT = 2000
BOOT_SEED = 42
METRES_PER_PIXEL = 10.0


def block_ids_for_valid_pixels(mask_path: Path, block_px: int) -> np.ndarray:
    """Block id per valid pixel, in the same row-major order load_pair returns.

    load_pair reads the valid mask on the reference grid and flattens valid pixels in
    row-major order (numpy boolean indexing). We replicate that grid here so block ids
    line up 1:1 with the y_true / y_pred vectors.
    """
    with rasterio.open(mask_path) as src:
        valid = src.read(1) == 1
    if int(valid.sum()) != EXPECTED_VALID_PIXELS:
        raise AssertionError(f"valid mask has {valid.sum()} px, expected {EXPECTED_VALID_PIXELS}")
    rows, cols = np.nonzero(valid)                     # row-major, matches boolean indexing
    br = rows // block_px
    bc = cols // block_px
    return (br.astype(np.int64) * 100_000 + bc).astype(np.int64)


def _block_confusion(y_true: np.ndarray, y_pred: np.ndarray, block_index: np.ndarray, n_blocks: int) -> np.ndarray:
    """Per-block [tp, fp, fn] counts. IoU = sum(tp) / sum(tp+fp+fn)."""
    tp = y_true & y_pred
    fp = (~y_true) & y_pred
    fn = y_true & (~y_pred)
    out = np.zeros((n_blocks, 3), dtype=np.float64)
    out[:, 0] = np.bincount(block_index, weights=tp, minlength=n_blocks)
    out[:, 1] = np.bincount(block_index, weights=fp, minlength=n_blocks)
    out[:, 2] = np.bincount(block_index, weights=fn, minlength=n_blocks)
    return out


def _iou_from_blockcounts(counts: np.ndarray, m_vec: np.ndarray) -> float:
    agg = m_vec @ counts                                # [3]
    denom = agg[0] + agg[1] + agg[2]
    return float(agg[0] / denom) if denom > 0 else float("nan")


def paired_block_bootstrap(cA: np.ndarray, cB: np.ndarray, rng: np.random.Generator, n_boot: int) -> dict:
    """Paired IoU difference (A - B) via shared block multiplicities. Percentile CI."""
    n_blocks = cA.shape[0]
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        draw = rng.integers(0, n_blocks, size=n_blocks)
        m_vec = np.bincount(draw, minlength=n_blocks).astype(np.float64)
        diffs[i] = _iou_from_blockcounts(cA, m_vec) - _iou_from_blockcounts(cB, m_vec)
    full = np.ones(n_blocks)
    point = _iou_from_blockcounts(cA, full) - _iou_from_blockcounts(cB, full)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {"point_diff": point, "boot_mean": float(np.mean(diffs)),
            "ci_lo": float(lo), "ci_hi": float(hi),
            "excludes_zero": bool(lo > 0 or hi < 0), "n_blocks": n_blocks}


def run(nas_root: Path, runs_root: Path) -> pd.DataFrame:
    mask_path = valid_mask_path(nas_root)
    block_index_raw = block_ids_for_valid_pixels(mask_path, BLOCK_PX)
    # densify block ids to 0..n_blocks-1 for bincount
    uniq, block_index = np.unique(block_index_raw, return_inverse=True)
    n_blocks = uniq.size
    print(f"AOI splits into {n_blocks} blocks of {BLOCK_PX*METRES_PER_PIXEL/1000:.0f} km "
          f"({EXPECTED_VALID_PIXELS} valid px)\n")

    run_dir = {(arch, seed): rd for _, arch, seed, rd in NEW_RUNS}
    rows: list[dict] = []

    for date in DATES:
        reference = gt_path(nas_root, date)
        null_done = False
        for seed in SEEDS:
            _, ypA = load_pair(reference, new_pred_path(runs_root, run_dir[(ARCH_A, seed)], date), mask_path)
            yt, ypB = load_pair(reference, new_pred_path(runs_root, run_dir[(ARCH_B, seed)], date), mask_path)
            yt_b = yt.astype(bool)

            cA = _block_confusion(yt_b, ypA.astype(bool), block_index, n_blocks)
            cB = _block_confusion(yt_b, ypB.astype(bool), block_index, n_blocks)

            rng = np.random.default_rng(BOOT_SEED)
            res = paired_block_bootstrap(cA, cB, rng, N_BOOT)
            rows.append({"date": date, "seed": seed, "contrast": f"{ARCH_A}-{ARCH_B}",
                         "block_km": BLOCK_PX * METRES_PER_PIXEL / 1000, **res})
            print(f"  {date} s{seed}: point diff {res['point_diff']:+.4f}  "
                  f"95% block-CI [{res['ci_lo']:+.4f}, {res['ci_hi']:+.4f}]  "
                  f"{'excludes 0' if res['excludes_zero'] else 'includes 0'}")

            # null control (once per date): A vs itself must straddle 0
            if not null_done:
                rng0 = np.random.default_rng(BOOT_SEED)
                null = paired_block_bootstrap(cA, cA, rng0, N_BOOT)
                rows.append({"date": date, "seed": seed, "contrast": f"NULL_{ARCH_A}_vs_self",
                             "block_km": BLOCK_PX * METRES_PER_PIXEL / 1000, **null})
                assert null["ci_lo"] <= 0 <= null["ci_hi"] and null["point_diff"] == 0.0, \
                    "null control (raster vs itself) must give a zero point diff and a CI straddling 0"
                null_done = True

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--nas-root", type=Path, default=Path(DEFAULT_NAS_ROOT))
    parser.add_argument("--runs-root", type=Path, default=Path(DEFAULT_RUNS_ROOT))
    parser.add_argument("--out-dir", type=Path, default=Path(DEFAULT_OUT_DIR))
    args = parser.parse_args()
    if not args.nas_root.exists():
        raise SystemExit(f"NAS root not found: {args.nas_root}\nIs the external volume mounted?")

    df = run(args.nas_root, args.runs_root)
    real = df[~df.contrast.str.startswith("NULL")]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / "statistical_assessment_block_bootstrap.csv"
    df.to_csv(out, index=False)

    n_excl = int(real.excludes_zero.sum())
    print(f"\nSpatial robustness: {n_excl}/{len(real)} per-date x per-seed contrasts have a "
          f"2 km block-CI excluding zero.")
    print(f"Null control passed on every date (raster vs itself -> CI straddles 0).")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
