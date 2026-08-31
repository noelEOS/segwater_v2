"""Emit per-(arm, date, block) [tp, fp, fn] counts for the hamp24 campaign.

Runs against a hamp24 scoring spec (same schema as score_hamp24.py): loads
every (arm, date) pair over the spec's valid mask, assigns each valid pixel a
2 km block id derived from the mask grid (rows//200, cols//200 — all three
extent masks share the raster origin, so block boundaries coincide spatially
across extents), and writes one long CSV of block confusion counts. These
counts are the sufficient statistics for the date-macro paired block
bootstrap, computed Mac-side; only this KB-scale CSV leaves the VM.

Run:
    python scripts/evaluation/vm/hamp24_block_counts.py --spec <spec.yaml> --out <csv>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from score_pairbased_hampyeong import audit, require  # noqa: E402

BLOCK_PX = 200


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--spec", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    spec = yaml.safe_load(args.spec.read_text())
    spec.setdefault("repo", "/home/noel/segwater_v2")
    spec.setdefault("pair_root", "outputs/inference")
    spec.setdefault("variant_column", False)
    repo = Path(spec["repo"])
    sys.path.insert(0, str(repo / "scripts"))
    from inference_overlap_utils import load_overlap_reference_and_prediction  # noqa: E402

    require(not args.out.exists(), f"refusing to clobber {args.out}")
    pair_root = repo / spec["pair_root"]
    gt_dir = Path(spec["gt_dir"])
    mask_path = Path(spec["valid_mask"])
    expected_valid = int(spec["expected_valid_pixels"])

    with rasterio.open(mask_path) as src:
        valid = src.read(1) == 1
    require(int(valid.sum()) == expected_valid,
            f"mask {mask_path.name}: {valid.sum()} px != {expected_valid}")
    rr, cc = np.nonzero(valid)
    raw = (rr // BLOCK_PX).astype(np.int64) * 100_000 + (cc // BLOCK_PX)
    uniq, bidx = np.unique(raw, return_inverse=True)
    n_blocks = len(uniq)
    print(f"{mask_path.name}: {expected_valid:,} px, {n_blocks} blocks")

    pairing = pd.read_csv(spec["dates_csv"])
    scenes = {str(s)[17:25]: f"{s}_Clipped" for s in pairing["scene_id"]}
    entries = audit(spec, repo, pair_root)

    rows = []
    for date in sorted(scenes):
        ref = gt_dir / f"DEM_FLOOD_S1_DESC_{date}_VAL_AOI.tif"
        scene = scenes[date]
        for e in entries:
            pred = pair_root / e["resolved_run_dir"] / scene / f"{scene}_probability_water.tif"
            y_true, y_pred, diag = load_overlap_reference_and_prediction(
                reference_path=str(ref), prediction_path=str(pred),
                reference_water_values=[1], reference_nodata_values=None,
                probability_threshold=0.5, probability_comparison="greater_equal",
                resolution_atol=1e-6, valid_mask_path=str(mask_path), valid_mask_value=1,
            )
            require(diag["valid_pixels_after_all_masks"] == expected_valid,
                    f"{date} {e['label']} s{e['seed']}: pixel count mismatch")
            yt = y_true.astype(bool)
            yp = y_pred.astype(bool)
            tp = np.bincount(bidx, weights=(yt & yp), minlength=n_blocks)
            fp = np.bincount(bidx, weights=(~yt & yp), minlength=n_blocks)
            fn = np.bincount(bidx, weights=(yt & ~yp), minlength=n_blocks)
            for b in range(n_blocks):
                rows.append((e["label"], e["seed"], date, int(uniq[b]),
                             int(tp[b]), int(fp[b]), int(fn[b])))
        print(f"  {date} done")

    df = pd.DataFrame(rows, columns=["arch", "seed", "date", "block_id", "tp", "fp", "fn"])
    df.to_csv(args.out, index=False)
    print(f"wrote {args.out} ({len(df)} rows, {n_blocks} blocks)")


if __name__ == "__main__":
    main()
