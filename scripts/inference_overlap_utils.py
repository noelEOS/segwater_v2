from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from rasterio.windows import from_bounds


def read_profile(path: str) -> dict[str, Any]:
    with rasterio.open(path) as src:
        return {
            "path": path,
            "name": Path(path).name,
            "shape": (src.height, src.width),
            "height": src.height,
            "width": src.width,
            "crs": src.crs.to_string() if src.crs else None,
            "transform": tuple(src.transform),
            "bounds": src.bounds,
            "res_x": abs(src.transform.a),
            "res_y": abs(src.transform.e),
        }


def assert_same_crs(reference: dict[str, Any], prediction: dict[str, Any]) -> None:
    if reference["crs"] != prediction["crs"]:
        raise ValueError(
            f"CRS mismatch: reference={reference['crs']}, prediction={prediction['crs']}. "
            "Official overlap evaluation does not reproject rasters."
        )


def assert_same_resolution(reference: dict[str, Any], prediction: dict[str, Any], resolution_atol: float) -> None:
    dx = abs(reference["res_x"] - prediction["res_x"])
    dy = abs(reference["res_y"] - prediction["res_y"])
    if dx > resolution_atol or dy > resolution_atol:
        raise ValueError(
            f"Resolution mismatch: reference=({reference['res_x']}, {reference['res_y']}), "
            f"prediction=({prediction['res_x']}, {prediction['res_y']}), "
            f"resolution_atol={resolution_atol}. "
            "Official overlap evaluation does not resample rasters."
        )


def intersection_bounds(reference_bounds, prediction_bounds) -> tuple[float, float, float, float]:
    left = max(reference_bounds.left, prediction_bounds.left)
    bottom = max(reference_bounds.bottom, prediction_bounds.bottom)
    right = min(reference_bounds.right, prediction_bounds.right)
    top = min(reference_bounds.top, prediction_bounds.top)
    if left >= right or bottom >= top:
        raise ValueError("Reference and prediction rasters do not overlap spatially.")
    return left, bottom, right, top


def rounded_window_from_bounds(bounds: tuple[float, float, float, float], transform) -> Any:
    left, bottom, right, top = bounds
    return from_bounds(left, bottom, right, top, transform=transform).round_offsets().round_lengths()


def threshold_probability(probability: np.ndarray, threshold: float, comparison: str) -> np.ndarray:
    if comparison == "greater_than":
        return (probability > threshold).astype(np.uint8)
    if comparison == "greater_equal":
        return (probability >= threshold).astype(np.uint8)
    raise ValueError("comparison must be 'greater_than' or 'greater_equal'.")


def load_overlap_reference_and_prediction(
    reference_path: str,
    prediction_path: str,
    reference_water_values: list[int | float],
    reference_nodata_values: list[int | float] | None,
    probability_threshold: float,
    probability_comparison: str,
    resolution_atol: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    reference_profile = read_profile(reference_path)
    prediction_profile = read_profile(prediction_path)

    assert_same_crs(reference_profile, prediction_profile)
    assert_same_resolution(reference_profile, prediction_profile, resolution_atol)

    overlap_bounds = intersection_bounds(reference_profile["bounds"], prediction_profile["bounds"])

    with rasterio.open(reference_path) as src_ref, rasterio.open(prediction_path) as src_pred:
        ref_window = rounded_window_from_bounds(overlap_bounds, src_ref.transform)
        pred_window = rounded_window_from_bounds(overlap_bounds, src_pred.transform)
        reference = src_ref.read(1, window=ref_window)
        probability = src_pred.read(1, window=pred_window)

    if reference.shape != probability.shape:
        raise ValueError(
            f"Overlap window shape mismatch after geospatial windowing: "
            f"reference={reference.shape}, prediction={probability.shape}. "
            "This usually indicates the rasters are not on the same grid."
        )

    valid_mask = np.ones(reference.shape, dtype=bool)
    if reference_nodata_values:
        valid_mask &= ~np.isin(reference, reference_nodata_values)

    if valid_mask.sum() == 0:
        raise ValueError("No valid reference pixels remain inside the overlap window.")

    y_true = np.isin(reference[valid_mask], reference_water_values).astype(np.uint8)
    y_pred = threshold_probability(probability[valid_mask], probability_threshold, probability_comparison)

    reference_total_pixels = reference_profile["height"] * reference_profile["width"]
    overlap_pixels = int(reference.shape[0] * reference.shape[1])
    overlap_reference_fraction = overlap_pixels / reference_total_pixels if reference_total_pixels else 0.0

    diagnostics = {
        "reference_shape": str(reference_profile["shape"]),
        "prediction_shape": str(prediction_profile["shape"]),
        "overlap_shape": str(reference.shape),
        "overlap_pixels": overlap_pixels,
        "overlap_reference_fraction": overlap_reference_fraction,
        "valid_pixels_after_nodata": int(valid_mask.sum()),
        "reference_res_x": reference_profile["res_x"],
        "reference_res_y": reference_profile["res_y"],
        "prediction_res_x": prediction_profile["res_x"],
        "prediction_res_y": prediction_profile["res_y"],
        "resolution_atol": resolution_atol,
        "probability_threshold": probability_threshold,
        "probability_comparison": probability_comparison,
    }
    return y_true, y_pred, diagnostics


def calculate_iou_precision_recall(y_pred: np.ndarray, y_true: np.ndarray) -> dict[str, float]:
    pred = y_pred.flatten().astype(bool)
    true = y_true.flatten().astype(bool)
    intersection = np.logical_and(pred, true).sum()
    union = np.logical_or(pred, true).sum()
    if union == 0:
        return {"iou": np.nan, "precision": np.nan, "recall": np.nan}

    predicted_water = pred.sum()
    reference_water = true.sum()
    return {
        "iou": intersection / union,
        "precision": intersection / predicted_water if predicted_water > 0 else 0.0,
        "recall": intersection / reference_water if reference_water > 0 else 0.0,
    }
