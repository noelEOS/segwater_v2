#!/usr/bin/env python3
"""Measure the S1 log10 export damage at CHIP level, across the whole manifest.

Earth Engine's server-side ``log10`` zeroed ~85% of one band on some scenes (see
docs/dataset_v3/S1_LOG10_EXPORT_DEFECT.md). The per-pair screens that found it
sampled one window per pair, which answers "which pairs are affected" but not
"how many chips are unusable" -- and the corpus is 1.36M chips, so those are
different questions.

The casualty has an exact, countable signature. Nodata is meant to be *fill*:
a pixel with no S1 observation at all, which is 0 in **both** bands. A pixel
that is 0 in one band and non-zero in the other is not fill -- the sensor saw
something there -- so it is either a log10 casualty or some other per-band
defect. Either way it is not valid data.

So per chip, per band, this counts:

* ``n_fill``      -- 0 in both bands. Legitimate absence of observation.
* ``n_orphan_vv`` -- VV is 0 where VH is not. Damage.
* ``n_orphan_vh`` -- VH is 0 where VV is not. Damage.

Every pixel of every chip is read at full resolution. No sampling: the point is
to bound the damage, and a sample cannot do that.

A pixel zeroed in *both* bands cannot be told apart from fill here, because
``0/0`` is what fill is. That is a property of the encoding rather than a gap in
this audit, and it can only be tested in Earth Engine against the unquantized
source -- which was done: 0.000% both-band overlap across 7 scenes, including
all three damaged ones.

Reads the whole S1 raster once per pair and slices the chips out of it in
memory, rather than issuing ~400 windowed reads per pair. The rasters are large
(up to ~400 MB) but this is far fewer syscalls and the memory is transient.
"""

from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import from_bounds

import paths

CHIP_PX = 224
CHIP_PIXELS = CHIP_PX * CHIP_PX


def s1_path(pair: str):
    matches = sorted(paths.RAW.glob("sentinel1_GRD/s1_q55s005_%s_*.tif" % pair))
    if len(matches) != 1:
        raise FileNotFoundError("expected one S1 raster for %s, found %d"
                                % (pair, len(matches)))
    return matches[0]


def audit_pair(pair: str, chip_ids: np.ndarray, boxes: np.ndarray) -> dict:
    """Count fill and orphan pixels for every chip of one pair."""
    try:
        with rasterio.open(s1_path(pair)) as src:
            if src.count != 2:
                raise ValueError("expected 2 bands, found %d" % src.count)
            vv = src.read(1)
            vh = src.read(2)
            transform = src.transform
            height, width = src.height, src.width

        zero_vv = vv == 0
        zero_vh = vh == 0
        fill = zero_vv & zero_vh
        orphan_vv = zero_vv & ~zero_vh
        orphan_vh = zero_vh & ~zero_vv

        n = len(chip_ids)
        out = {k: np.zeros(n, dtype=np.int32)
               for k in ("n_fill", "n_orphan_vv", "n_orphan_vh", "n_read")}
        for i, (west, south, east, north) in enumerate(boxes):
            window = from_bounds(west, south, east, north,
                                 transform=transform).round_offsets().round_lengths()
            col0, row0 = int(window.col_off), int(window.row_off)
            col1, row1 = col0 + int(window.width), row0 + int(window.height)
            # Clip to the raster; a chip reaching past the edge is counted only
            # over the pixels that exist, and n_read records how many that was.
            c0, r0 = max(0, col0), max(0, row0)
            c1, r1 = min(width, col1), min(height, row1)
            if c1 <= c0 or r1 <= r0:
                continue
            sl = (slice(r0, r1), slice(c0, c1))
            out["n_read"][i] = (r1 - r0) * (c1 - c0)
            out["n_fill"][i] = int(fill[sl].sum())
            out["n_orphan_vv"][i] = int(orphan_vv[sl].sum())
            out["n_orphan_vh"][i] = int(orphan_vh[sl].sum())

        return {"pair": pair, "chip_id": chip_ids, "error": "", **out}
    except Exception as error:  # noqa: BLE001 - one bad pair must not kill the run
        return {"pair": pair, "error": "%s: %s" % (type(error).__name__, str(error)[:200])}


def summarise(table: pd.DataFrame) -> None:
    total = len(table)
    orphan = table["n_orphan_vv"] + table["n_orphan_vh"]
    frac = orphan / table["n_read"].clip(lower=1)

    print("CHIPS AUDITED %d  (pairs %d)" % (total, table["pair_name"].nunique()))
    print()
    print("ORPHAN PIXELS -- 0 in one band, data in the other. Not fill; damage.")
    print("  chips with any orphan pixel   %9d  (%.3f%%)"
          % (int((orphan > 0).sum()), 100 * float((orphan > 0).mean())))
    print("  total orphan pixels           %9d" % int(orphan.sum()))
    print()
    print("  by share of the chip affected:")
    for lo, hi, label in [(0.0, 1e-9, "clean (0)"),
                          (1e-9, 0.001, "<0.1%"),
                          (0.001, 0.01, "0.1-1%"),
                          (0.01, 0.05, "1-5%"),
                          (0.05, 0.25, "5-25%"),
                          (0.25, 0.50, "25-50%"),
                          (0.50, 1.01, ">50%")]:
        mask = (frac >= lo) & (frac < hi) if lo else (frac <= hi)
        print("    %-12s %9d chips" % (label, int(mask.sum())))
    print()
    print("FILL PIXELS -- 0 in both bands. Legitimate absence of observation.")
    fill_frac = table["n_fill"] / table["n_read"].clip(lower=1)
    print("  chips with any fill           %9d  (%.2f%%)"
          % (int((table["n_fill"] > 0).sum()), 100 * float((table["n_fill"] > 0).mean())))
    print("  mean fill share               %9.4f" % float(fill_frac.mean()))
    print()
    affected = table.loc[orphan > 0, "pair_name"].value_counts()
    print("PAIRS CONTRIBUTING DAMAGED CHIPS: %d" % len(affected))
    for pair, n in affected.head(25).items():
        share = frac[table["pair_name"] == pair].mean()
        print("    %-12s %6d chips   mean orphan share %.4f" % (pair, n, share))
    if len(affected) > 25:
        print("    ... and %d more" % (len(affected) - 25))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    parser.add_argument("--limit", type=int, default=None, help="first N pairs only")
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()

    manifest = pd.read_parquet(
        paths.require(paths.MANIFESTS / "dataset_v3_manifest.parquet", "v3 manifest"),
        columns=["pair_name", "chip_id", "bbox_w", "bbox_s", "bbox_e", "bbox_n"])
    groups = list(manifest.groupby("pair_name", sort=True))
    if args.limit:
        groups = groups[:args.limit]

    frames, errors = [], []
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        futures = [
            pool.submit(audit_pair, pair, group["chip_id"].to_numpy(),
                        group[["bbox_w", "bbox_s", "bbox_e", "bbox_n"]].to_numpy())
            for pair, group in groups
        ]
        for done, future in enumerate(as_completed(futures), 1):
            result = future.result()
            if result["error"]:
                errors.append(result)
                continue
            frames.append(pd.DataFrame({
                "pair_name": result["pair"],
                "chip_id": result["chip_id"],
                "n_read": result["n_read"],
                "n_fill": result["n_fill"],
                "n_orphan_vv": result["n_orphan_vv"],
                "n_orphan_vh": result["n_orphan_vh"],
            }))
            if done % 250 == 0:
                print("  %d/%d pairs" % (done, len(groups)), flush=True)

    table = pd.concat(frames, ignore_index=True)
    if errors:
        print("\nERRORS on %d pairs, e.g. %s" % (len(errors), errors[0]["error"]))

    summarise(table)

    if not args.write:
        print("\n(dry run; pass --write to save)")
        return 0

    paths.ensure_out()
    target = paths.MANIFESTS / "s1_damage_audit.parquet"
    table.to_parquet(target, index=False, compression="zstd")
    print("\nwrote %s  (%d rows)" % (target, len(table)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
