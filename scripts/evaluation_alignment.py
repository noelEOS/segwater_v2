import math
from pathlib import Path
from typing import Any

import numpy as np
import rasterio


def read_raster_profile(path: str) -> dict[str, Any]:
    with rasterio.open(path) as src:
        crs = src.crs
        bounds = src.bounds
        return {
            "path": path,
            "shape": (src.height, src.width),
            "crs": crs.to_string() if crs else None,
            "crs_is_geographic": bool(crs and crs.is_geographic),
            "transform": tuple(src.transform),
            "center_lat": (bounds.top + bounds.bottom) / 2.0,
        }


def meters_per_degree(latitude: float) -> tuple[float, float]:
    lat_rad = math.radians(latitude)
    meters_per_degree_lat = 111_132.92 - 559.82 * math.cos(2 * lat_rad) + 1.175 * math.cos(4 * lat_rad)
    meters_per_degree_lon = 111_412.84 * math.cos(lat_rad) - 93.5 * math.cos(3 * lat_rad)
    return abs(meters_per_degree_lon), abs(meters_per_degree_lat)


def transform_diff_stats(base: dict[str, Any], other: dict[str, Any]) -> dict[str, float]:
    base_t = np.asarray(base["transform"], dtype=float)
    other_t = np.asarray(other["transform"], dtype=float)
    diff = np.abs(base_t - other_t)

    if base.get("crs_is_geographic"):
        meters_per_x_unit, meters_per_y_unit = meters_per_degree(float(base.get("center_lat", 0.0)))
    else:
        # For projected CRS, raster transform units are assumed to be meters.
        meters_per_x_unit = 1.0
        meters_per_y_unit = 1.0

    max_x_m = float(diff[[0, 1, 2]].max() * meters_per_x_unit)
    max_y_m = float(diff[[3, 4, 5]].max() * meters_per_y_unit)
    max_m = max(max_x_m, max_y_m)

    pixel_x_m = abs(float(base_t[0]) * meters_per_x_unit)
    pixel_y_m = abs(float(base_t[4]) * meters_per_y_unit)
    max_x_px = max_x_m / pixel_x_m if pixel_x_m > 0 else np.nan
    max_y_px = max_y_m / pixel_y_m if pixel_y_m > 0 else np.nan

    return {
        "max_abs_transform_diff": float(diff.max()),
        "max_transform_diff_meters": float(max_m),
        "max_transform_diff_pixels": float(np.nanmax([max_x_px, max_y_px])),
    }


def check_transform(base: dict[str, Any], other: dict[str, Any], label: str, transform_atol: float) -> dict[str, Any]:
    base_t = np.asarray(base["transform"], dtype=float)
    other_t = np.asarray(other["transform"], dtype=float)
    stats = transform_diff_stats(base, other)

    if not np.allclose(base_t, other_t, rtol=0.0, atol=transform_atol):
        raise ValueError(
            f"Raster alignment mismatch for transform ({label}): "
            f"max_abs_transform_diff={stats['max_abs_transform_diff']:.12g}, "
            f"max_transform_diff_meters={stats['max_transform_diff_meters']:.6g}, "
            f"max_transform_diff_pixels={stats['max_transform_diff_pixels']:.6g}, "
            f"transform_atol={transform_atol:.12g}. "
            "Official evaluation does not crop or resample rasters."
        )

    if stats["max_abs_transform_diff"] > 0:
        print(
            f"  Alignment note ({label}): transform differs only within tolerance; "
            f"max_abs_diff={stats['max_abs_transform_diff']:.12g}, "
            f"approx_max_shift={stats['max_transform_diff_meters']:.6g} m, "
            f"approx_max_shift={stats['max_transform_diff_pixels']:.6g} pixels, "
            f"tolerance={transform_atol:.12g}."
        )

    return {"label": label, "status": "passed", **stats}


def validate_pair_alignment(reference_path: str, prediction_path: str, transform_atol: float) -> list[dict[str, Any]]:
    reference = read_raster_profile(reference_path)
    prediction = read_raster_profile(prediction_path)

    for key in ["shape", "crs"]:
        if reference[key] != prediction[key]:
            raise ValueError(
                f"Raster alignment mismatch for {Path(reference_path).name} vs {Path(prediction_path).name}: "
                f"{key}: reference={reference[key]}, prediction={prediction[key]}. "
                "Official evaluation does not crop rasters."
            )

    return [check_transform(reference, prediction, "reference vs prediction", transform_atol)]


def validate_triplet_alignment(
    reference_path: str,
    prediction_path: str,
    valid_mask_path: str,
    transform_atol: float,
) -> list[dict[str, Any]]:
    reference = read_raster_profile(reference_path)
    prediction = read_raster_profile(prediction_path)
    valid_mask = read_raster_profile(valid_mask_path)

    for key in ["shape", "crs"]:
        values = {"reference": reference[key], "prediction": prediction[key], "valid_mask": valid_mask[key]}
        if len(set(values.values())) != 1:
            raise ValueError(
                f"Raster alignment mismatch for {key}: "
                f"reference={values['reference']}, prediction={values['prediction']}, valid_mask={values['valid_mask']}"
            )

    return [
        check_transform(reference, prediction, "reference vs prediction", transform_atol),
        check_transform(reference, valid_mask, "reference vs valid_mask", transform_atol),
    ]
