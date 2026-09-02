#!/usr/bin/env python3
"""Count water, land and invalid pixels in each chip's label.

Three columns: ``n_water``, ``n_land``, ``n_invalid``. They sum to 50,176 by
construction, which is asserted per chip -- an invariant is worth more than a
convenience column, and any fraction a consumer wants is one division away.

No fraction is stored on purpose. "Water fraction" is ambiguous between
``water / (water + land)`` and ``water / 50176``: the first is undefined for a
fully-invalid chip, the second makes a cloudy chip look water-poor when it is
merely unobserved. Storing the counts lets the consumer pick the denominator
their question needs, and makes which one they picked visible in their code.

**Each chip is read from the label it will actually be trained on.** A chip with
``v3_label_variant == 'b8_lt400'`` is read from ``b8_lt400_water_663/out/``, the
rest from ``out/``. Counting everything from the parent would misdescribe the
135,766 ``apply-nir`` chips, whose whole point is that their water class was
corrected.

The B8<400 correction only ever flips land to water -- verified: 0 invalid-state
changes and 0 water-to-land flips across the three pairs checked when the
variant was first examined -- so ``n_invalid`` is identical either way and only
the water/land split moves.
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
from rasterio.windows import from_bounds

import paths

CHIP_PIXELS = 224 * 224
LAND, WATER, INVALID = 0, 1, 255

EXPECTED_CHIPS = 1_357_354
EXPECTED_B8 = 135_766


def label_path(pair: str, variant: str):
    root = paths.LABELS_B8 / "out" if variant == "b8_lt400" else paths.LABELS_OUT
    matches = sorted((root / pair).glob("*.tif"))
    if len(matches) != 1:
        raise FileNotFoundError("expected one %s label for %s, found %d"
                                % (variant, pair, len(matches)))
    return matches[0]


def count_pair(pair: str, chips: pd.DataFrame) -> dict:
    """Class counts for every chip of one pair. Errors are returned, not raised."""
    try:
        n = len(chips)
        counts = {k: np.zeros(n, dtype=np.int32)
                  for k in ("n_water", "n_land", "n_invalid")}

        # A pair's chips can span both variants, so read whichever rasters this
        # pair actually needs -- at most two, each opened once.
        for variant, group in chips.groupby("v3_label_variant", sort=True):
            with rasterio.open(label_path(pair, variant)) as src:
                transform = src.transform
                data = src.read(1)
            for i, row in zip(group.index, group.itertuples()):
                window = from_bounds(row.bbox_w, row.bbox_s, row.bbox_e, row.bbox_n,
                                     transform=transform).round_offsets().round_lengths()
                r0, c0 = int(window.row_off), int(window.col_off)
                block = data[r0:r0 + int(window.height), c0:c0 + int(window.width)]
                if block.size != CHIP_PIXELS:
                    raise ValueError("%s chip %d read %d px, expected %d"
                                     % (pair, row.chip_id, block.size, CHIP_PIXELS))
                at = chips.index.get_loc(i)
                counts["n_water"][at] = int((block == WATER).sum())
                counts["n_land"][at] = int((block == LAND).sum())
                counts["n_invalid"][at] = int((block == INVALID).sum())

        total = counts["n_water"] + counts["n_land"] + counts["n_invalid"]
        if not (total == CHIP_PIXELS).all():
            raise AssertionError("%s: counts do not sum to %d" % (pair, CHIP_PIXELS))

        return {"pair": pair, "chip_id": chips["chip_id"].to_numpy(),
                "error": "", **counts}
    except Exception as error:  # noqa: BLE001 - one bad pair must not kill the run
        return {"pair": pair, "error": "%s: %s" % (type(error).__name__, str(error)[:200])}


def summarise(table: pd.DataFrame, manifest: pd.DataFrame) -> None:
    total = table["n_water"] + table["n_land"] + table["n_invalid"]
    valid = table["n_water"] + table["n_land"]
    print("LABEL CLASS COUNTS over %d chips" % len(table))
    print("  sum == %d on every chip: %s" % (CHIP_PIXELS, bool((total == CHIP_PIXELS).all())))
    print()
    for column in ("n_water", "n_land", "n_invalid"):
        share = table[column].sum() / total.sum()
        print("  %-10s total %15s  (%.2f%% of all pixels)"
              % (column, "{:,}".format(int(table[column].sum())), 100 * share))
    print()
    print("  per-chip water share of VALID pixels:")
    with np.errstate(invalid="ignore", divide="ignore"):
        frac = np.where(valid > 0, table["n_water"] / valid.replace(0, np.nan), np.nan)
    frac = pd.Series(frac)
    print("     median %.4f  p10 %.4f  p90 %.4f  | undefined (no valid px): %d"
          % (frac.median(), frac.quantile(0.10), frac.quantile(0.90),
             int(frac.isna().sum())))
    print()
    print("  chips that are all water: %s | all land: %s | all invalid: %s"
          % ("{:,}".format(int((table["n_water"] == CHIP_PIXELS).sum())),
             "{:,}".format(int((table["n_land"] == CHIP_PIXELS).sum())),
             "{:,}".format(int((table["n_invalid"] == CHIP_PIXELS).sum()))))

    # The invalid count must agree with what the mask stage already measured.
    joined = manifest.merge(table, on=["pair_name", "chip_id"], validate="1:1")
    expected = (joined["invalid_frac"] * CHIP_PIXELS).round().astype(int)
    delta = (expected - joined["n_invalid"]).abs()
    print()
    print("  n_invalid vs invalid_frac x %d: max |diff| %d px, disagreeing chips %d"
          % (CHIP_PIXELS, int(delta.max()), int((delta > 1).sum())))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    parser.add_argument("--limit", type=int, default=None, help="first N pairs only")
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()

    manifest = pd.read_parquet(
        paths.require(paths.MANIFESTS / "dataset_v3_manifest.parquet", "v3 manifest"),
        columns=["pair_name", "chip_id", "bbox_w", "bbox_s", "bbox_e", "bbox_n",
                 "v3_label_variant", "invalid_frac"])
    if len(manifest) != EXPECTED_CHIPS:
        raise AssertionError("manifest is %d chips, expected %d"
                             % (len(manifest), EXPECTED_CHIPS))
    b8 = int((manifest["v3_label_variant"] == "b8_lt400").sum())
    if not args.limit and b8 != EXPECTED_B8:
        raise AssertionError("expected %d b8_lt400 chips, found %d" % (EXPECTED_B8, b8))

    groups = list(manifest.groupby("pair_name", sort=True))
    if args.limit:
        groups = groups[:args.limit]
    print("counting %d chips across %d pairs (%d on the B8<400 variant)"
          % (sum(len(g) for _, g in groups), len(groups), b8))

    frames, errors = [], []
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        futures = [pool.submit(count_pair, pair, group) for pair, group in groups]
        for done, future in enumerate(as_completed(futures), 1):
            result = future.result()
            if result["error"]:
                errors.append(result)
                continue
            frames.append(pd.DataFrame({
                "pair_name": result["pair"], "chip_id": result["chip_id"],
                "n_water": result["n_water"], "n_land": result["n_land"],
                "n_invalid": result["n_invalid"],
            }))
            if done % 250 == 0:
                print("  %d/%d pairs" % (done, len(groups)), flush=True)

    table = pd.concat(frames, ignore_index=True)
    if errors:
        print("\nERRORS on %d pairs, e.g. %s" % (len(errors), errors[0]["error"]))

    summarise(table, manifest)

    if not args.write:
        print("\n(dry run; pass --write to save)")
        return 0

    paths.ensure_out()
    target = paths.MANIFESTS / "chip_label_counts.parquet"
    table.to_parquet(target, index=False, compression="zstd")
    print("\nwrote %s  (%d rows)" % (target, len(table)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
