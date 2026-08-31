#!/usr/bin/env python3
"""Score every chip already in the corpus against the new valid mask.

The chips in ``dataset_v3_base.parquet`` passed the *original* gate, which
counted SCL classes {3,7,8,9,10} and rejected above 15%. The new lineage's mask
is different -- the Level 2 label's 255 -- so a chip that passed then may fail
now, and the point of this pass is to find out which.

Scores are stored as a continuous fraction and nothing is marked rejected here.
A threshold is a decision, and decisions belong in ``apply_invalid.py``; keeping
them apart means the threshold can be swept against the real distribution
without re-reading 3,383 rasters.

Work is one task per pair, so each label is opened once for its ~436 chips
rather than once per chip, and chips are read in raster order so successive
windows hit the same tiles.
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import rasterio

import invalid_mask
import paths

# Verified 2026-08-31. Re-declared rather than imported from build_base, so a
# change there surfaces here as a conflict instead of propagating silently.
EXPECTED_CHIPS = 1_474_047
EXPECTED_PAIRS = 3_383

BASE_COLUMNS = ["pair_name", "chip_id", "bbox_w", "bbox_s", "bbox_e", "bbox_n"]


def load_base() -> pd.DataFrame:
    base = pd.read_parquet(
        paths.require(paths.MANIFESTS / "dataset_v3_base.parquet", "base table"),
        columns=BASE_COLUMNS,
    )
    if len(base) != EXPECTED_CHIPS or base["pair_name"].nunique() != EXPECTED_PAIRS:
        raise AssertionError(
            "base is %d chips / %d pairs, expected %d / %d"
            % (len(base), base["pair_name"].nunique(), EXPECTED_CHIPS, EXPECTED_PAIRS)
        )
    if base.duplicated(["pair_name", "chip_id"]).any():
        raise AssertionError("(pair_name, chip_id) is not unique in the base")
    return base


def score_pair(pair: str, chip_ids: np.ndarray, boxes: np.ndarray) -> dict:
    """Score one pair's chips. Errors are returned, not raised."""
    try:
        label = next((paths.LABELS_OUT / pair).glob("*.tif"))
        with rasterio.open(label) as src:
            if src.crs.to_epsg() != 4326:
                raise ValueError("label CRS is %s, expected EPSG:4326" % src.crs)
            order = invalid_mask.sort_key_for_locality(src, boxes)
            scored = invalid_mask.invalid_fractions(src, boxes[order])
        restore = np.argsort(order)
        return {
            "pair": pair,
            "chip_id": chip_ids,
            "frac": scored["frac"][restore],
            "npix": scored["npix"][restore],
            "boundless": scored["boundless"][restore],
            "covered": scored["covered"][restore],
            "max_residual": scored["max_residual"],
            "error": "",
        }
    except Exception as error:  # noqa: BLE001 - one bad pair must not kill the run
        return {"pair": pair, "error": "%s: %s" % (type(error).__name__, str(error)[:200])}


def summarise(table: pd.DataFrame, errors: list[dict], max_residual: float) -> None:
    covered = table["label_covers"]
    frac = table.loc[covered, "invalid_frac"]
    print("CHIPS SCORED")
    print("  chips                    %9d" % len(table))
    print("  pairs                    %9d" % table["pair_name"].nunique())
    print("  label covers them        %9d" % int(covered.sum()))
    print("  NOT covered              %9d" % int((~covered).sum()))
    print("  partly-filled reads      %9d" % int(table["invalid_boundless"].sum()))
    print("  max sub-pixel residual   %9.2e" % max_residual)
    if errors:
        print("  ERRORS                   %9d  e.g. %s" % (len(errors), errors[0]["error"]))
    print()
    print("INVALID FRACTION under the new mask")
    print("  mean %.4f | median %.4f | p90 %.4f | p99 %.4f"
          % (frac.mean(), frac.median(), frac.quantile(0.90), frac.quantile(0.99)))
    print()
    print("WOULD BE REJECTED, by threshold")
    for threshold in (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.50):
        n = int((frac > threshold).sum())
        print("  > %.2f    %9d  (%.2f%%)" % (threshold, n, 100 * n / max(len(frac), 1)))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    parser.add_argument("--limit", type=int, default=None,
                        help="score only the first N pairs (smoke run)")
    parser.add_argument("--write", action="store_true",
                        help="write the scores; without it, only summarise")
    args = parser.parse_args()

    base = load_base()
    groups = list(base.groupby("pair_name", sort=True))
    if args.limit:
        groups = groups[:args.limit]

    frames, errors, max_residual = [], [], 0.0
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        futures = [
            pool.submit(
                score_pair, pair,
                group["chip_id"].to_numpy(),
                group[["bbox_w", "bbox_s", "bbox_e", "bbox_n"]].to_numpy(),
            )
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
                "invalid_frac": result["frac"],
                "invalid_npix": result["npix"],
                "invalid_boundless": result["boundless"],
                "label_covers": result["covered"],
            }))
            max_residual = max(max_residual, result["max_residual"])
            if done % 250 == 0:
                print("  %d/%d pairs" % (done, len(groups)), flush=True)

    table = pd.concat(frames, ignore_index=True)

    if not args.limit and len(table) != EXPECTED_CHIPS:
        raise AssertionError("expected %d scored chips, got %d"
                             % (EXPECTED_CHIPS, len(table)))
    if max_residual >= invalid_mask.MAX_SUBPIXEL_RESIDUAL:
        raise AssertionError("sub-pixel residual %.3g exceeds the guard; the "
                             "label grid moved" % max_residual)
    scored_npix = table.loc[table["label_covers"], "invalid_npix"]
    if not (scored_npix == invalid_mask.CHIP_PIXELS).all():
        raise AssertionError("a covered chip did not read exactly %d pixels"
                             % invalid_mask.CHIP_PIXELS)
    if table.loc[table["label_covers"], "invalid_frac"].isna().any():
        raise AssertionError("a covered chip produced no invalid fraction")

    table["invalid_mask_rule"] = invalid_mask.MASK_RULE
    table["assessed_at"] = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")

    summarise(table, errors, max_residual)

    if not args.write:
        print("\n(dry run; pass --write to save)")
        return 0

    paths.ensure_out()
    target = paths.MANIFESTS / "chip_invalid_scores.parquet"
    table.to_parquet(target, index=False, compression="zstd")
    print("\nwrote %s  (%d rows, %.1f MiB)"
          % (target, len(table), target.stat().st_size / 2**20))
    return 0


if __name__ == "__main__":
    sys.exit(main())
