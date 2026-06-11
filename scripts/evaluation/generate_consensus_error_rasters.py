"""
Write per-scene consensus-error GeoTIFFs from the probability rasters: for every
valid pixel, the number of ensemble members that misclassify it. Pixels wrong in
all (or most) members are the irreducible error core no ensemble can fix; their
spatial pattern (river edges, ponds, label noise, ...) guides the discussion.

Each output raster is on the reference grid (uint8, LZW, nodata=255):

  band 1  n_models_wrong at per-member TUNED thresholds (pooled-IoU-optimal)
  band 2  n_models_wrong at the default threshold 0.5
  band 3  reference class (1=water, 0=non-water)
          -> errors at ref=1 are FN (missed water), at ref=0 are FP

Members and scene/reference pairs come from evaluation_manifest.csv (success
rows only); the external valid mask is applied as in the benchmark evaluator.

Usage
-----
    python scripts/evaluation/generate_consensus_error_rasters.py \
        path/to/evaluation_manifest.csv \
        --valid-mask-path mask.tif --output-dir path
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import from_bounds as window_from_bounds

sys.path.insert(0, str(Path(__file__).parent))
from analyze_error_decorrelation import DIR_RENAMES

NODATA = np.uint8(255)
EDGES = np.linspace(0.0, 1.0, 1001)


def windowed_read(path: str, bounds, expected_shape, boundless=False):
    with rasterio.open(path) as src:
        win = window_from_bounds(*bounds, transform=src.transform)
        win = win.round_offsets().round_lengths()
        arr = src.read(1, window=win, boundless=boundless,
                       fill_value=0 if boundless else None)
    if arr.shape != expected_shape:
        raise ValueError(f"{path}: window shape {arr.shape} != {expected_shape}")
    return arr


def main():
    p = argparse.ArgumentParser()
    p.add_argument("manifest_csv")
    p.add_argument("--valid-mask-path", required=True)
    p.add_argument("--valid-mask-value", type=int, default=1)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--reference-water-values", type=int, nargs="+", default=[1])
    p.add_argument("--reference-nodata-values", type=int, nargs="+", default=[255])
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.manifest_csv)
    df = df[df["status"] == "success"].copy()
    df["model_name"] = df["model_name"].replace(DIR_RENAMES)
    members = sorted(df["model_name"].unique())
    scenes = sorted(df["s1_id"].unique())
    print(f"{len(members)} members, {len(scenes)} scenes")

    # ---- pass 1: load everything, tune per-member thresholds (pooled IoU)
    scene_arrays = {}  # scene -> dict(ref2d, valid2d, probs[n,H,W], profile)
    tp = np.zeros((len(members), 1000))
    fp = np.zeros((len(members), 1000))
    total_w = 0
    for scene in scenes:
        sub = df[df["s1_id"] == scene].set_index("model_name")
        ref_path = sub["reference_path"].iloc[0]
        with rasterio.open(ref_path) as src:
            reference = src.read(1)
            bounds = src.bounds
            profile = src.profile.copy()
        valid = ~np.isin(reference, args.reference_nodata_values)
        mask = windowed_read(args.valid_mask_path, bounds, reference.shape,
                             boundless=True)
        valid &= mask == args.valid_mask_value
        ref_bin = np.isin(reference, args.reference_water_values)

        probs = np.empty((len(members), *reference.shape), dtype=np.float32)
        for i, m in enumerate(members):
            probs[i] = windowed_read(sub.loc[m, "prediction_path"], bounds,
                                     reference.shape)
            hw, _ = np.histogram(probs[i][valid & ref_bin], bins=EDGES)
            hn, _ = np.histogram(probs[i][valid & ~ref_bin], bins=EDGES)
            tp[i] += hw[::-1].cumsum()[::-1]
            fp[i] += hn[::-1].cumsum()[::-1]
        total_w += int((valid & ref_bin).sum())
        scene_arrays[scene] = dict(ref=ref_bin, valid=valid, probs=probs,
                                   profile=profile)
        print(f"  loaded {scene}")

    iou = tp / np.maximum(tp + fp + (total_w - tp), 1)
    tuned = EDGES[iou.argmax(axis=1)]
    for m, t in zip(members, tuned):
        print(f"  tuned threshold {t:.3f}  {m}")

    # ---- pass 2: write multiplicity rasters
    for scene, d in scene_arrays.items():
        ref, valid, probs = d["ref"], d["valid"], d["probs"]
        n_tuned = np.zeros(ref.shape, dtype=np.uint8)
        n_05 = np.zeros(ref.shape, dtype=np.uint8)
        for i in range(len(members)):
            n_tuned += ((probs[i] > tuned[i]) != ref).astype(np.uint8)
            n_05 += ((probs[i] > 0.5) != ref).astype(np.uint8)
        ref_band = ref.astype(np.uint8)
        for band in (n_tuned, n_05, ref_band):
            band[~valid] = NODATA

        profile = d["profile"]
        profile.update(driver="GTiff", dtype="uint8", count=3, nodata=int(NODATA),
                       compress="lzw")
        out_path = out_dir / f"{scene}_consensus_errors.tif"
        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(np.stack([n_tuned, n_05, ref_band]))
            dst.set_band_description(1, "n_models_wrong_tuned_thresholds")
            dst.set_band_description(2, "n_models_wrong_threshold_0p5")
            dst.set_band_description(3, "reference_water")
        n_all = int(((n_tuned == len(members)) & valid).sum())
        n_most = int(((n_tuned >= len(members) - 2) & valid & (n_tuned != NODATA)).sum())
        print(f"  {out_path.name}: all-wrong={n_all} px, >=10-wrong={n_most} px")

    pd.DataFrame({"member": members, "tuned_threshold": tuned}).to_csv(
        out_dir / "tuned_thresholds.csv", index=False
    )
    print(f"\nOutputs in {out_dir}")


if __name__ == "__main__":
    main()
