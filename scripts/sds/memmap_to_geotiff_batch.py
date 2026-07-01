#!/usr/bin/env python3
"""Convert probability-water memmaps to georeferenced GeoTIFFs.

Inference runs write a raw ``*_probability_water.memmap`` (float32) plus a
sidecar ``*_metadata.json`` per scene. The metadata carries everything needed to
georeference the array (width, height, CRS, affine transform), so we do not need
the original input ``.tif`` (whose recorded path may point at the training
machine and not exist here).

For every memmap found under the search root we write a single-band float32
GeoTIFF next to it, named ``*_probability_water.tif`` (the name the run metadata
already reserves under ``outputs.probability_geotiff``).

Profile matches the repo's canonical writer (``src/utils/raster_export.py``):
deflate + predictor=2, tiled 512x512, band description "probability_water".

The ``--delete-memmaps`` flag (OFF by default) removes each source memmap only
*after* its GeoTIFF has been written and re-opened/validated, so a memmap is
never deleted without a confirmed replacement.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import Affine

MEMMAP_SUFFIX = "_probability_water.memmap"
GEOTIFF_SUFFIX = "_probability_water.tif"
BAND_DESCRIPTION = "probability_water"


def find_memmaps(root: Path) -> list[Path]:
    """Return all probability-water memmaps under ``root`` (sorted)."""
    return sorted(root.rglob(f"*{MEMMAP_SUFFIX}"))


def metadata_path_for(memmap_path: Path) -> Path:
    """Sidecar metadata path for a memmap (``..._metadata.json`` in same dir)."""
    scene_id = memmap_path.name[: -len(MEMMAP_SUFFIX)]
    return memmap_path.parent / f"{scene_id}_metadata.json"


def geotiff_path_for(memmap_path: Path) -> Path:
    """Output GeoTIFF path: same scene id, ``.tif`` extension."""
    scene_id = memmap_path.name[: -len(MEMMAP_SUFFIX)]
    return memmap_path.parent / f"{scene_id}{GEOTIFF_SUFFIX}"


def profile_from_metadata(meta: dict) -> tuple[dict, tuple[int, int], str]:
    """Build a rasterio profile, (height, width) shape, and precision from metadata."""
    inp = meta["input"]
    width = int(inp["width"])
    height = int(inp["height"])
    crs = inp["crs"]
    # transform stored as a flat 9-element list (a, b, c, d, e, f, 0, 0, 1)
    a, b, c, d, e, f = inp["transform"][:6]
    transform = Affine(a, b, c, d, e, f)

    precision = meta.get("inference", {}).get("precision", "float32")

    profile = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": 1,
        "dtype": "float32",  # broad GIS compatibility even if memmap were float16
        "crs": crs,
        "transform": transform,
        "nodata": None,
        "compress": "deflate",
        "predictor": 2,
        "tiled": True,
        "blockxsize": 512,
        "blockysize": 512,
    }
    return profile, (height, width), precision


def write_geotiff(memmap_path: Path, meta: dict, out_path: Path) -> None:
    """Read the memmap and write a georeferenced GeoTIFF at ``out_path``."""
    profile, shape, precision = profile_from_metadata(meta)

    expected_bytes = shape[0] * shape[1] * (4 if precision == "float32" else 2)
    actual_bytes = memmap_path.stat().st_size
    if actual_bytes != expected_bytes:
        raise ValueError(
            f"memmap size {actual_bytes} != expected {expected_bytes} "
            f"for shape {shape} precision {precision}"
        )

    memmap_dtype = np.float32 if precision == "float32" else np.float16
    array = np.memmap(memmap_path, dtype=memmap_dtype, mode="r", shape=shape)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    with rasterio.open(tmp_path, "w", **profile) as dst:
        dst.write(np.asarray(array, dtype=np.float32), 1)
        dst.set_band_description(1, BAND_DESCRIPTION)
        dst.update_tags(
            source_memmap=memmap_path.name,
            scene_id=meta.get("scene_id", ""),
            model_architecture=meta.get("model", {}).get("architecture", ""),
            model_encoder=meta.get("model", {}).get("encoder", ""),
            threshold=str(meta.get("post_processing", {}).get("threshold", "")),
        )
    tmp_path.replace(out_path)


def validate_geotiff(out_path: Path, shape: tuple[int, int]) -> None:
    """Re-open the GeoTIFF and confirm it is readable with the right shape/CRS."""
    with rasterio.open(out_path) as src:
        if (src.height, src.width) != shape:
            raise ValueError(
                f"written GeoTIFF shape {(src.height, src.width)} != expected {shape}"
            )
        if src.crs is None:
            raise ValueError("written GeoTIFF has no CRS")
        if src.transform == Affine.identity():
            raise ValueError("written GeoTIFF has identity transform (not georeferenced)")
        # touch the data to ensure the band is decodable
        _ = src.read(1, window=((0, min(8, src.height)), (0, min(8, src.width))))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "root",
        type=Path,
        help="Directory to search recursively for *_probability_water.memmap files.",
    )
    parser.add_argument(
        "--delete-memmaps",
        action="store_true",
        default=False,
        help="Delete each source memmap AFTER its GeoTIFF is written and validated. "
        "Off by default.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="Rewrite GeoTIFFs that already exist (default: skip existing).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="List what would be done without writing or deleting anything.",
    )
    args = parser.parse_args(argv)

    root: Path = args.root
    if not root.exists():
        print(f"ERROR: root does not exist: {root}", file=sys.stderr)
        return 2

    memmaps = find_memmaps(root)
    if not memmaps:
        print(f"No '*{MEMMAP_SUFFIX}' files found under {root}")
        return 0

    print(f"Found {len(memmaps)} memmap(s) under {root}")
    if args.delete_memmaps and not args.dry_run:
        print("--delete-memmaps is ON: source memmaps will be removed after validation.")

    written = skipped = deleted = failed = 0

    for memmap_path in memmaps:
        meta_path = metadata_path_for(memmap_path)
        out_path = geotiff_path_for(memmap_path)
        rel = memmap_path.relative_to(root)

        if not meta_path.exists():
            print(f"FAIL  {rel}: missing metadata sidecar {meta_path.name}")
            failed += 1
            continue

        try:
            meta = json.loads(meta_path.read_text())
            _, shape, _ = profile_from_metadata(meta)
        except Exception as exc:  # noqa: BLE001
            print(f"FAIL  {rel}: bad metadata ({exc})")
            failed += 1
            continue

        geotiff_exists = out_path.exists()
        if geotiff_exists and not args.overwrite:
            # Still validate so we can safely delete the memmap if requested.
            if args.dry_run:
                print(f"SKIP  {rel}: GeoTIFF exists")
            else:
                try:
                    validate_geotiff(out_path, shape)
                except Exception as exc:  # noqa: BLE001
                    print(f"FAIL  {rel}: existing GeoTIFF invalid ({exc}); use --overwrite")
                    failed += 1
                    continue
            skipped += 1
        else:
            if args.dry_run:
                action = "OVERWRITE" if geotiff_exists else "WRITE"
                print(f"{action:5} {rel} -> {out_path.name}")
            else:
                try:
                    write_geotiff(memmap_path, meta, out_path)
                    validate_geotiff(out_path, shape)
                except Exception as exc:  # noqa: BLE001
                    print(f"FAIL  {rel}: {exc}")
                    failed += 1
                    continue
                written += 1

        # Deletion only happens once we are certain a valid GeoTIFF is on disk.
        if args.delete_memmaps:
            if args.dry_run:
                print(f"      would delete {rel}")
            else:
                memmap_path.unlink()
                deleted += 1

    print(
        f"\nDone. written={written} skipped(existing)={skipped} "
        f"deleted_memmaps={deleted} failed={failed}"
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
