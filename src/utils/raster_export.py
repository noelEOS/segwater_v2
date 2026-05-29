from pathlib import Path

import numpy as np
import rasterio


def memmap_to_geotiff(
    memmap_path: str | Path,
    reference_tif_path: str | Path,
    output_tif_path: str | Path,
    shape: tuple[int, int],
    precision: str,
    band_description: str,
    tags: dict[str, str] | None = None,
) -> Path:
    """Export a probability memmap as a georeferenced single-band GeoTIFF."""
    memmap_path = Path(memmap_path)
    output_tif_path = Path(output_tif_path)

    memmap_dtype = np.float32 if precision == "float32" else np.float16
    output_dtype = "float32"  # Broad GeoTIFF/GIS compatibility, even when the memmap is float16.

    array = np.memmap(memmap_path, dtype=memmap_dtype, mode="r", shape=shape)

    with rasterio.open(reference_tif_path) as src:
        profile = src.profile.copy()

    profile.update(
        driver="GTiff",
        height=shape[0],
        width=shape[1],
        count=1,
        dtype=output_dtype,
        nodata=None,
        compress="deflate",
        predictor=2,
        tiled=True,
        blockxsize=512,
        blockysize=512,
    )

    output_tif_path.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(output_tif_path, "w", **profile) as dst:
        dst.write(np.asarray(array, dtype=np.float32), 1)
        dst.set_band_description(1, band_description)
        if tags:
            dst.update_tags(**tags)

    return output_tif_path


def write_binary_mask_geotiff(
    probability_memmap_path: str | Path,
    reference_tif_path: str | Path,
    output_tif_path: str | Path,
    shape: tuple[int, int],
    precision: str,
    threshold: float,
    tags: dict[str, str] | None = None,
) -> Path:
    """Export a thresholded water mask GeoTIFF from the probability memmap."""
    probability_memmap_path = Path(probability_memmap_path)
    output_tif_path = Path(output_tif_path)

    memmap_dtype = np.float32 if precision == "float32" else np.float16
    prob_map = np.memmap(probability_memmap_path, dtype=memmap_dtype, mode="r", shape=shape)
    mask = (prob_map >= threshold).astype(np.uint8)

    with rasterio.open(reference_tif_path) as src:
        profile = src.profile.copy()

    profile.update(
        driver="GTiff",
        height=shape[0],
        width=shape[1],
        count=1,
        dtype="uint8",
        nodata=255,
        compress="deflate",
        tiled=True,
        blockxsize=512,
        blockysize=512,
    )

    output_tif_path.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(output_tif_path, "w", **profile) as dst:
        dst.write(mask, 1)
        dst.set_band_description(1, "water_mask")
        if tags:
            dst.update_tags(**tags)

    return output_tif_path
