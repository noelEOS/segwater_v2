"""
Generate per-scene confusion-matrix rasters from an evaluation_manifest.csv.

For every row with status=='success' the script reads the reference and
prediction TIFFs, reproduces the exact overlap logic used by
evaluate_indonesia_inference_run_benchmark.py, and writes two GeoTIFFs next
to (or below) the manifest:

  <output_dir>/<model>/<s1_id>/
      binary_mask.tif        – uint8, 1=water 0=non-water (prediction after threshold)
      confusion_matrix.tif   – uint8, pixel-wise CM label over the OVERLAP window

Confusion-matrix encoding
  1 = TP   (pred=1, ref=1)
  2 = FP   (pred=1, ref=0)
  3 = FN   (pred=0, ref=1)
  4 = TN   (pred=0, ref=0)
  0 = nodata (reference nodata pixels inside the overlap window)

Metrics recomputed from the rasters must match the per_scene_metrics.csv values
produced by the benchmark evaluator.

Usage
-----
    python scripts/evaluation/generate_confusion_rasters.py \\
        path/to/evaluation_manifest.csv \\
        [--output-dir path/to/output] \\
        [--threshold 0.5] \\
        [--comparison greater_than] \\
        [--reference-water-values 1] \\
        [--reference-nodata-values 255] \\
        [--resolution-atol 1e-12] \\
        [--overwrite]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_bounds as transform_from_bounds

# ---------------------------------------------------------------------------
# Inline copies of the overlap utilities so this script is self-contained
# (mirrors inference_overlap_utils.py exactly).
# ---------------------------------------------------------------------------

from rasterio.windows import from_bounds as window_from_bounds


def _read_profile(path: str) -> dict:
    with rasterio.open(path) as src:
        return {
            "path": path,
            "shape": (src.height, src.width),
            "height": src.height,
            "width": src.width,
            "crs": src.crs,
            "transform": src.transform,
            "bounds": src.bounds,
            "res_x": abs(src.transform.a),
            "res_y": abs(src.transform.e),
            "dtype": src.dtypes[0],
            "nodata": src.nodata,
        }


def _rounded_window(bounds, transform):
    left, bottom, right, top = bounds
    return window_from_bounds(left, bottom, right, top, transform=transform).round_offsets().round_lengths()


def _intersection_bounds(ref_bounds, pred_bounds):
    left = max(ref_bounds.left, pred_bounds.left)
    bottom = max(ref_bounds.bottom, pred_bounds.bottom)
    right = min(ref_bounds.right, pred_bounds.right)
    top = min(ref_bounds.top, pred_bounds.top)
    if left >= right or bottom >= top:
        raise ValueError("Reference and prediction rasters do not overlap spatially.")
    return left, bottom, right, top


def _threshold(probability: np.ndarray, threshold: float, comparison: str) -> np.ndarray:
    if comparison == "greater_than":
        return (probability > threshold).astype(np.uint8)
    if comparison == "greater_equal":
        return (probability >= threshold).astype(np.uint8)
    raise ValueError(f"Unknown comparison: {comparison!r}")


# ---------------------------------------------------------------------------
# Core per-scene logic
# ---------------------------------------------------------------------------

CM_TP = np.uint8(1)
CM_FP = np.uint8(2)
CM_FN = np.uint8(3)
CM_TN = np.uint8(4)
CM_NODATA = np.uint8(0)


def _confusion_matrix_array(
    reference: np.ndarray,
    probability: np.ndarray,
    reference_water_values: list,
    reference_nodata_values: list,
    probability_threshold: float,
    probability_comparison: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns (binary_mask_2d, confusion_map_2d, valid_mask_2d).

    binary_mask   – uint8 array with same shape as reference (overlap window)
    confusion_map – uint8 CM label array, same shape
    valid_mask    – bool mask of pixels used in metric computation
    """
    valid_mask = np.ones(reference.shape, dtype=bool)
    if reference_nodata_values:
        valid_mask &= ~np.isin(reference, reference_nodata_values)

    binary_mask = _threshold(probability, probability_threshold, probability_comparison)

    ref_binary = np.isin(reference, reference_water_values).astype(np.uint8)

    cm = np.full(reference.shape, CM_NODATA, dtype=np.uint8)
    # TP
    cm[valid_mask & (binary_mask == 1) & (ref_binary == 1)] = CM_TP
    # FP
    cm[valid_mask & (binary_mask == 1) & (ref_binary == 0)] = CM_FP
    # FN
    cm[valid_mask & (binary_mask == 0) & (ref_binary == 1)] = CM_FN
    # TN
    cm[valid_mask & (binary_mask == 0) & (ref_binary == 0)] = CM_TN

    return binary_mask, cm, valid_mask


def _metrics_from_cm_array(cm: np.ndarray) -> dict:
    tp = int((cm == CM_TP).sum())
    fp = int((cm == CM_FP).sum())
    fn = int((cm == CM_FN).sum())
    # tn = int((cm == CM_TN).sum())  # not needed for IoU/P/R

    union = tp + fp + fn
    iou = tp / union if union > 0 else float("nan")
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return {"iou": iou, "precision": precision, "recall": recall, "tp": tp, "fp": fp, "fn": fn}


def _window_transform(src_transform, window) -> rasterio.Affine:
    return rasterio.windows.transform(window, src_transform)


def process_scene(
    reference_path: str,
    prediction_path: str,
    output_dir: Path,
    model: str,
    s1_id: str,
    reference_water_values: list,
    reference_nodata_values: list,
    probability_threshold: float,
    probability_comparison: str,
    resolution_atol: float,
    overwrite: bool,
) -> dict:
    """Process one scene and write binary_mask.tif + confusion_matrix.tif."""
    scene_dir = output_dir / model / s1_id
    binary_path = scene_dir / "binary_mask.tif"
    cm_path = scene_dir / "confusion_matrix.tif"

    if not overwrite and binary_path.exists() and cm_path.exists():
        return {"status": "skipped", "binary_path": str(binary_path), "cm_path": str(cm_path)}

    ref_profile = _read_profile(reference_path)
    pred_profile = _read_profile(prediction_path)

    # -- CRS check
    if ref_profile["crs"] != pred_profile["crs"]:
        raise ValueError(f"CRS mismatch: {ref_profile['crs']} vs {pred_profile['crs']}")

    # -- Resolution check
    dx = abs(ref_profile["res_x"] - pred_profile["res_x"])
    dy = abs(ref_profile["res_y"] - pred_profile["res_y"])
    if dx > resolution_atol or dy > resolution_atol:
        raise ValueError(
            f"Resolution mismatch: ref=({ref_profile['res_x']},{ref_profile['res_y']}), "
            f"pred=({pred_profile['res_x']},{pred_profile['res_y']})"
        )

    overlap_bounds = _intersection_bounds(ref_profile["bounds"], pred_profile["bounds"])

    with rasterio.open(reference_path) as src_ref, rasterio.open(prediction_path) as src_pred:
        ref_window = _rounded_window(overlap_bounds, src_ref.transform)
        pred_window = _rounded_window(overlap_bounds, src_pred.transform)
        reference = src_ref.read(1, window=ref_window)
        probability = src_pred.read(1, window=pred_window)
        overlap_transform = _window_transform(src_ref.transform, ref_window)
        crs = src_ref.crs

    if reference.shape != probability.shape:
        raise ValueError(
            f"Shape mismatch after windowing: ref={reference.shape}, pred={probability.shape}"
        )

    binary_mask, cm_array, valid_mask = _confusion_matrix_array(
        reference=reference,
        probability=probability,
        reference_water_values=reference_water_values,
        reference_nodata_values=reference_nodata_values,
        probability_threshold=probability_threshold,
        probability_comparison=probability_comparison,
    )

    metrics = _metrics_from_cm_array(cm_array)

    scene_dir.mkdir(parents=True, exist_ok=True)

    out_profile = {
        "driver": "GTiff",
        "dtype": "uint8",
        "count": 1,
        "height": reference.shape[0],
        "width": reference.shape[1],
        "crs": crs,
        "transform": overlap_transform,
        "compress": "lzw",
        "nodata": None,
    }

    with rasterio.open(binary_path, "w", **out_profile) as dst:
        dst.write(binary_mask[np.newaxis, :, :])

    with rasterio.open(cm_path, "w", **out_profile) as dst:
        dst.write(cm_array[np.newaxis, :, :])

    return {
        "status": "success",
        "binary_path": str(binary_path),
        "cm_path": str(cm_path),
        "overlap_shape": reference.shape,
        "valid_pixels": int(valid_mask.sum()),
        **metrics,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate binary-mask and confusion-matrix GeoTIFFs from evaluation_manifest.csv"
    )
    p.add_argument("manifest_csv", help="Path to evaluation_manifest.csv")
    p.add_argument(
        "--output-dir",
        default=None,
        help="Root output directory (default: manifest parent dir / confusion_rasters)",
    )
    p.add_argument("--threshold", type=float, default=0.5, help="Probability threshold (default: 0.5)")
    p.add_argument(
        "--comparison",
        choices=["greater_than", "greater_equal"],
        default="greater_than",
        help="Threshold comparison operator (default: greater_than)",
    )
    p.add_argument(
        "--reference-water-values",
        type=int,
        nargs="+",
        default=[1],
        metavar="V",
        help="Reference raster values treated as water (default: 1)",
    )
    p.add_argument(
        "--reference-nodata-values",
        type=int,
        nargs="+",
        default=[255],
        metavar="V",
        help="Reference raster nodata values to mask out (default: 255)",
    )
    p.add_argument(
        "--resolution-atol",
        type=float,
        default=1e-12,
        help="Absolute tolerance for resolution comparison (default: 1e-12)",
    )
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing output files")
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)

    manifest_path = Path(args.manifest_csv)
    if not manifest_path.exists():
        print(f"ERROR: manifest not found: {manifest_path}", file=sys.stderr)
        sys.exit(1)

    output_dir = Path(args.output_dir) if args.output_dir else manifest_path.parent / "confusion_rasters"
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(manifest_path)
    success_rows = df[df["status"] == "success"].copy()

    print(f"Manifest: {manifest_path}")
    print(f"Output dir: {output_dir}")
    print(f"Rows with status=success: {len(success_rows)} / {len(df)}")
    print(f"Threshold: {args.comparison} {args.threshold}")

    results = []
    for _, row in success_rows.iterrows():
        model = str(row["model"])
        s1_id = str(row["s1_id"])
        reference_path = str(row["reference_path"])
        prediction_path = str(row["prediction_path"])

        print(f"  {model} / {s1_id} ... ", end="", flush=True)
        try:
            result = process_scene(
                reference_path=reference_path,
                prediction_path=prediction_path,
                output_dir=output_dir,
                model=model,
                s1_id=s1_id,
                reference_water_values=args.reference_water_values,
                reference_nodata_values=args.reference_nodata_values,
                probability_threshold=args.threshold,
                probability_comparison=args.comparison,
                resolution_atol=args.resolution_atol,
                overwrite=args.overwrite,
            )
            status = result["status"]
            if status == "success":
                print(
                    f"ok  iou={result['iou']:.4f}  p={result['precision']:.4f}  r={result['recall']:.4f}"
                )
            else:
                print(status)
        except Exception as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            result = {"model": model, "s1_id": s1_id, "status": "error", "error": str(exc)}

        results.append({"model": model, "s1_id": s1_id, **result})

    summary_df = pd.DataFrame(results)
    summary_csv = output_dir / "confusion_rasters_summary.csv"
    summary_df.to_csv(summary_csv, index=False)
    print(f"\nSummary written to: {summary_csv}")

    n_ok = (summary_df["status"] == "success").sum() if not summary_df.empty else 0
    n_skip = (summary_df["status"] == "skipped").sum() if not summary_df.empty else 0
    n_err = (summary_df["status"] == "error").sum() if not summary_df.empty else 0
    print(f"Done: {n_ok} written, {n_skip} skipped, {n_err} errors")


if __name__ == "__main__":
    main()
