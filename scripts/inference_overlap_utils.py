from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from rasterio.windows import from_bounds

from evaluation.metrics import compute_binary_metrics


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


def assert_same_crs(reference: dict[str, Any], other: dict[str, Any], other_label: str = "prediction") -> None:
    if reference["crs"] != other["crs"]:
        raise ValueError(
            f"CRS mismatch: reference={reference['crs']}, {other_label}={other['crs']}. "
            "Official overlap evaluation does not reproject rasters."
        )


def assert_same_resolution(reference: dict[str, Any], other: dict[str, Any], resolution_atol: float, other_label: str = "prediction") -> None:
    dx = abs(reference["res_x"] - other["res_x"])
    dy = abs(reference["res_y"] - other["res_y"])
    if dx > resolution_atol or dy > resolution_atol:
        raise ValueError(
            f"Resolution mismatch: reference=({reference['res_x']}, {reference['res_y']}), "
            f"{other_label}=({other['res_x']}, {other['res_y']}), "
            f"resolution_atol={resolution_atol}. "
            "Official overlap evaluation does not resample rasters."
        )


def intersection_bounds(*bounds_list) -> tuple[float, float, float, float]:
    left = max(bounds.left for bounds in bounds_list)
    bottom = max(bounds.bottom for bounds in bounds_list)
    right = min(bounds.right for bounds in bounds_list)
    top = min(bounds.top for bounds in bounds_list)
    if left >= right or bottom >= top:
        raise ValueError("Input rasters do not overlap spatially.")
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
    valid_mask_path: str | None = None,
    valid_mask_value: int | float = 1,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    reference_profile = read_profile(reference_path)
    prediction_profile = read_profile(prediction_path)
    mask_profile = read_profile(valid_mask_path) if valid_mask_path else None

    assert_same_crs(reference_profile, prediction_profile, "prediction")
    assert_same_resolution(reference_profile, prediction_profile, resolution_atol, "prediction")

    bounds_inputs = [reference_profile["bounds"], prediction_profile["bounds"]]
    if mask_profile:
        assert_same_crs(reference_profile, mask_profile, "valid_mask")
        assert_same_resolution(reference_profile, mask_profile, resolution_atol, "valid_mask")
        bounds_inputs.append(mask_profile["bounds"])

    overlap_bounds = intersection_bounds(*bounds_inputs)

    with rasterio.open(reference_path) as src_ref, rasterio.open(prediction_path) as src_pred:
        ref_window = rounded_window_from_bounds(overlap_bounds, src_ref.transform)
        pred_window = rounded_window_from_bounds(overlap_bounds, src_pred.transform)
        reference = src_ref.read(1, window=ref_window)
        probability = src_pred.read(1, window=pred_window)

    valid_mask_external = None
    if valid_mask_path:
        with rasterio.open(valid_mask_path) as src_mask:
            mask_window = rounded_window_from_bounds(overlap_bounds, src_mask.transform)
            valid_mask_external = src_mask.read(1, window=mask_window)

    if reference.shape != probability.shape:
        raise ValueError(
            f"Overlap window shape mismatch after geospatial windowing: "
            f"reference={reference.shape}, prediction={probability.shape}. "
            "This usually indicates the rasters are not on the same grid."
        )
    if valid_mask_external is not None and reference.shape != valid_mask_external.shape:
        raise ValueError(
            f"Overlap window shape mismatch for valid mask: reference={reference.shape}, "
            f"valid_mask={valid_mask_external.shape}."
        )

    valid_mask = np.ones(reference.shape, dtype=bool)
    if reference_nodata_values:
        valid_mask &= ~np.isin(reference, reference_nodata_values)
    if valid_mask_external is not None:
        valid_mask &= valid_mask_external == valid_mask_value

    if valid_mask.sum() == 0:
        raise ValueError("No valid pixels remain inside the overlap window after applying masks.")

    y_true = np.isin(reference[valid_mask], reference_water_values).astype(np.uint8)
    y_pred = threshold_probability(probability[valid_mask], probability_threshold, probability_comparison)

    reference_total_pixels = reference_profile["height"] * reference_profile["width"]
    overlap_pixels = int(reference.shape[0] * reference.shape[1])
    valid_after_reference_nodata = int((~np.isin(reference, reference_nodata_values)).sum()) if reference_nodata_values else overlap_pixels
    valid_after_external_mask = int((valid_mask_external == valid_mask_value).sum()) if valid_mask_external is not None else overlap_pixels
    final_valid_pixels = int(valid_mask.sum())

    diagnostics = {
        "reference_shape": str(reference_profile["shape"]),
        "prediction_shape": str(prediction_profile["shape"]),
        "valid_mask_shape": str(mask_profile["shape"]) if mask_profile else "",
        "overlap_shape": str(reference.shape),
        "overlap_pixels": overlap_pixels,
        "overlap_reference_fraction": overlap_pixels / reference_total_pixels if reference_total_pixels else 0.0,
        "valid_pixels_after_reference_nodata": valid_after_reference_nodata,
        "valid_pixels_after_external_mask": valid_after_external_mask,
        "valid_pixels_after_all_masks": final_valid_pixels,
        "external_valid_mask_path": valid_mask_path or "",
        "external_valid_mask_value": valid_mask_value if valid_mask_path else "",
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
    """Backward-compatible wrapper around the centralized metric utility."""
    metrics = compute_binary_metrics(y_true=y_true, y_pred=y_pred, include_counts=False)
    return {metric: metrics[metric] for metric in ["iou", "precision", "recall"]}
