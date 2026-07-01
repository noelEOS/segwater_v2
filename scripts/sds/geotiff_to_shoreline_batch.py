#!/usr/bin/env python3
"""Extract shoreline vectors from probability-water GeoTIFFs.

For every ``*_probability_water.tif`` under a search root, run marching-squares
contouring at a probability threshold (default 0.5), keep the longest ``--keep``
contours (default 3) by true metric length, optionally simplify and densify, and
write a GeoPackage ``*_shoreline.gpkg`` next to it (the name the run metadata
reserves under ``outputs.shoreline_geojson``).

This is the GeoTIFF-sourced counterpart to ``src/utils/vectorizer.py``'s
``ShorelineVectorizer`` (which reads a raw memmap + a separate reference tif).
We deleted the memmaps, but the GeoTIFFs carry both the probability band *and*
the CRS/affine transform, so they are a complete source for vectorization. The
contour math, metric top-k, simplify, and densify steps mirror the vectorizer
exactly so outputs are consistent with the rest of the pipeline.

Defaults here: threshold=0.5, keep=3 longest, simplify_tolerance=1.0 m,
densify spacing=1.0 m, no min-length filter (matches the run's
``apply_length_filter: false``).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from shapely.geometry import LineString
from skimage import measure

GEOTIFF_SUFFIX = "_probability_water.tif"
SHORELINE_SUFFIX = "_shoreline.gpkg"


def find_geotiffs(root: Path) -> list[Path]:
    return sorted(root.rglob(f"*{GEOTIFF_SUFFIX}"))


def shoreline_path_for(geotiff_path: Path) -> Path:
    scene_id = geotiff_path.name[: -len(GEOTIFF_SUFFIX)]
    return geotiff_path.parent / f"{scene_id}{SHORELINE_SUFFIX}"


def extract_shoreline(
    geotiff_path: Path,
    out_path: Path,
    threshold: float,
    keep_top_k: int,
    min_length_meters: float,
    simplify_tolerance_meters: float,
    densify_spacing_meters: float,
) -> int:
    """Vectorize one GeoTIFF; returns the number of geometries written."""
    with rasterio.open(geotiff_path) as src:
        prob_map = src.read(1)
        transform = src.transform
        crs = src.crs

    # Marching squares -> list of (N, 2) arrays of (row, col) sub-pixel coords.
    contours = measure.find_contours(prob_map, level=threshold)

    records = []
    for contour_id, contour in enumerate(contours):
        xs, ys = rasterio.transform.xy(
            transform, contour[:, 0], contour[:, 1], offset="center"
        )
        if len(xs) < 2:
            continue
        line = LineString(zip(xs, ys))
        if line.is_empty or line.length == 0:
            continue
        records.append(
            {
                "contour_id": contour_id,
                "threshold": threshold,
                "n_vertices": len(line.coords),
                "geometry": line,
            }
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    gdf = gpd.GeoDataFrame(records, geometry="geometry", crs=crs)

    if len(gdf) == 0:
        gdf.to_file(out_path, driver="GPKG")
        return 0

    # Metric post-processing: lengths/filter/top-k/simplify/densify in a metric CRS.
    original_crs = gdf.crs
    metric_crs = gdf.estimate_utm_crs()
    gdf_m = gdf.to_crs(metric_crs)
    gdf_m["length_m"] = gdf_m.geometry.length

    if min_length_meters > 0:
        gdf_m = gdf_m[gdf_m["length_m"] >= min_length_meters].copy()

    if keep_top_k > 0 and len(gdf_m) > 0:
        gdf_m = (
            gdf_m.sort_values("length_m", ascending=False)
            .head(keep_top_k)
            .reset_index(drop=True)
        )
        gdf_m["rank"] = range(1, len(gdf_m) + 1)

    if simplify_tolerance_meters > 0 and len(gdf_m) > 0:
        gdf_m["geometry"] = gdf_m.geometry.simplify(
            tolerance=simplify_tolerance_meters, preserve_topology=True
        )
        gdf_m["length_simplified_m"] = gdf_m.geometry.length
        gdf_m["n_vertices_simplified"] = gdf_m.geometry.apply(
            lambda g: len(g.coords) if g is not None and not g.is_empty else 0
        )

    if densify_spacing_meters > 0 and len(gdf_m) > 0:
        gdf_m["geometry"] = gdf_m.geometry.segmentize(densify_spacing_meters)
        gdf_m["n_vertices_densified"] = gdf_m.geometry.apply(
            lambda g: len(g.coords) if g is not None and not g.is_empty else 0
        )

    gdf = gdf_m.to_crs(original_crs)
    gdf.to_file(out_path, driver="GPKG")
    return len(gdf)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "root",
        type=Path,
        help="Directory to search recursively for *_probability_water.tif files.",
    )
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument(
        "--keep",
        type=int,
        default=3,
        help="Keep the N longest contours by metric length (default 3).",
    )
    parser.add_argument("--min-length-meters", type=float, default=0.0)
    parser.add_argument("--simplify-tolerance-meters", type=float, default=1.0)
    parser.add_argument("--densify-spacing-meters", type=float, default=1.0)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="Rewrite shoreline files that already exist (default: skip existing).",
    )
    parser.add_argument("--dry-run", action="store_true", default=False)
    args = parser.parse_args(argv)

    root: Path = args.root
    if not root.exists():
        print(f"ERROR: root does not exist: {root}", file=sys.stderr)
        return 2

    geotiffs = find_geotiffs(root)
    if not geotiffs:
        print(f"No '*{GEOTIFF_SUFFIX}' files found under {root}")
        return 0

    print(
        f"Found {len(geotiffs)} GeoTIFF(s) under {root} | "
        f"threshold={args.threshold} keep={args.keep} longest"
    )

    written = skipped = failed = empty = 0
    for tif in geotiffs:
        out_path = shoreline_path_for(tif)
        rel = tif.relative_to(root)

        if out_path.exists() and not args.overwrite:
            skipped += 1
            continue

        if args.dry_run:
            print(f"WRITE {rel} -> {out_path.name}")
            written += 1
            continue

        try:
            n = extract_shoreline(
                tif,
                out_path,
                threshold=args.threshold,
                keep_top_k=args.keep,
                min_length_meters=args.min_length_meters,
                simplify_tolerance_meters=args.simplify_tolerance_meters,
                densify_spacing_meters=args.densify_spacing_meters,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"FAIL  {rel}: {exc}")
            failed += 1
            continue
        if n == 0:
            empty += 1
        written += 1

    print(
        f"\nDone. written={written} (empty={empty}) "
        f"skipped(existing)={skipped} failed={failed}"
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
