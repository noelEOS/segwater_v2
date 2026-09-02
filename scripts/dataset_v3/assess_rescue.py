#!/usr/bin/env python3
"""Score the SCL-off windows that the new valid mask might rescue.

The original chipper rejected 217,894 in-buffer windows on its SCL gate. Much
of that was Sen2Cor calling a bright shoreline cloud, so under the new mask many
of those windows are clean. This measures how many.

Two exclusions, both enforced structurally rather than by a filter applied
afterwards, because a filter can be forgotten and a structure cannot:

* **Scenes that reached round 3 are not eligible.** Rescuing a window there
  would mean visually confirming it -- a fourth round of manual inspection,
  which is not on the table. Those pairs are never submitted for scoring, and an
  assertion before writing confirms every surviving row came from an accepted
  scene.
* **``scl0`` windows are not eligible.** They were rejected for containing a
  pixel of genuinely absent data, not for cloud, and a new cloud mask cannot
  redeem missing data. They are dropped at read time.

⚠️ **These windows have no chip array in the memmap.** The source pixels exist
-- the label and S2 band rasters cover the whole pair, which is how the invalid
fraction below is measured at all -- but chipping only cut and stacked the
windows that passed the gate, so a rejected window was never written to a
memmap and has no row index into one. It also has no ``chip_id``.

The manifest this writes is therefore a work order: cut these footprints from
pair rasters we already hold, append them to the memmaps, then assign chip ids
and splits. Until that happens they are not rows that can be appended to the
chip corpus, and nothing may join this file to the base table. The output
deliberately has no ``chip_id`` column at all, so a join attempted on one fails
loudly instead of silently matching nothing.

Scores are stored as a continuous fraction. No threshold is applied here: a
threshold is a decision, and decisions belong in apply_invalid.py.
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
from read_scl_lost import (malformed_fraction, pair_name_from_path,
                           read_scl_lost)

# Verified on the VM 2026-08-31. Asserted so a corpus swap cannot pass silently.
EXPECTED_GPKG_FILES = 3_097
EXPECTED_FEATURES = 217_894
EXPECTED_VERDICTS = {"cloud15": 217_123, "scl0": 771}
EXPECTED_ELIGIBILITY = {"accepted": 2_195, "water-correction": 632, "rejected": 268}
EXPECTED_SCORED = 144_338   # 144,345 less PAIR_4985's 7 malformed windows

# 9,931 of the scored windows fall ENTIRELY outside their pair's label raster
# -- the label is cropped to the chip extent and never reached them. They are
# unscoreable, not cloudy, and are reported as such rather than counted as 100%
# invalid, which would silently turn "we cannot measure this" into "we measured
# it and it failed". (9,933 across all eligible pairs, less the 2 belonging to
# the dropped PAIR_4985.)
EXPECTED_UNSCOREABLE = 9_931

# Pairs with an scl_lost file but no memmap-selected chip, so absent from the
# base. Named rather than silently dropped.
EXPECTED_ABSENT = {"PAIR_1266", "PAIR_3254"}

# Only the cloud clause is reassessable; see the module docstring.
RESCUABLE_VERDICTS = {"cloud15"}

# PAIR_4985's *scl_lost* geometry is malformed. Its 7 windows have the correct
# width (0.020122 deg = exactly 224 px) but are 0.291 deg tall, about 3,240 px
# -- 14.5x too tall. GDAL reads the same values, so the defect is in the file.
#
# The pair itself is fine: its dynamic_extents GPKG holds 5 perfectly square
# chips that overlay the raster correctly. The fault is specific to how the
# scl_lost file was built. build_ungated_buffer_gpkg.py does not store window
# geometry; it recovers the affine by fitting passed chips' bounds against their
# (row, col) with np.polyfit. PAIR_4985 has only 5 surviving chips and all 5 sit
# in ONE row, so the y-slope is fitted from a single distinct x-value and came
# out 14.5x too large. The cols fit had 5 distinct values and is exact -- hence
# the correct width and wrong height.
#
# That x/y asymmetry is the fingerprint: any pair whose survivors occupy a
# single row or column is suspect. This is the only one of the 2,195 eligible
# pairs where the residual guard fired.
MALFORMED_GEOMETRY_PAIRS = {"PAIR_4985"}


def scene_categories() -> pd.Series:
    """Round-2 verdict per pair, from the base table."""
    base = pd.read_parquet(
        paths.require(paths.MANIFESTS / "dataset_v3_base.parquet", "base table"),
        columns=["pair_name", "qc_scene_category"],
    )
    return base.drop_duplicates("pair_name").set_index("pair_name")["qc_scene_category"]


def partition(categories: pd.Series) -> tuple[list[str], dict[str, int], list[str]]:
    """Split the scl_lost pairs into eligible, excluded, and absent-from-base."""
    files = sorted(paths.require(paths.CHIPS_SCL_LOST, "SCL-off GPKG directory")
                   .glob("PAIR_*_scl_lost.gpkg"))
    if len(files) != EXPECTED_GPKG_FILES:
        raise AssertionError("expected %d GPKGs, found %d"
                             % (EXPECTED_GPKG_FILES, len(files)))

    absent, counts, eligible = [], {}, []
    for path in files:
        pair = pair_name_from_path(path)
        if pair not in categories.index:
            absent.append(pair)
            continue
        category = categories[pair]
        counts[category] = counts.get(category, 0) + 1
        if category == "accepted" and pair not in MALFORMED_GEOMETRY_PAIRS:
            eligible.append(pair)

    if counts != EXPECTED_ELIGIBILITY:
        raise AssertionError("eligibility partition changed: %r" % counts)
    if set(absent) != EXPECTED_ABSENT:
        raise AssertionError("pairs absent from the base changed: %r" % sorted(absent))
    return eligible, counts, absent


def score_pair(pair: str) -> dict:
    """Score one eligible pair's rescuable windows. Errors are returned, not raised."""
    try:
        table = read_scl_lost(paths.CHIPS_SCL_LOST / ("%s_scl_lost.gpkg" % pair),
                              verdicts=RESCUABLE_VERDICTS)
        if table.empty:
            return {"pair": pair, "rows": None, "error": ""}

        # Check the geometry rather than trusting it; see malformed_fraction.
        bad = malformed_fraction(table)
        if bad:
            return {"pair": pair, "rows": None, "malformed": bad,
                    "error": "malformed geometry: %.0f%% of windows are not "
                             "chip-sized squares" % (100 * bad)}

        boxes = table[["bbox_w", "bbox_s", "bbox_e", "bbox_n"]].to_numpy()
        label = next((paths.LABELS_OUT / pair).glob("*.tif"))
        with rasterio.open(label) as src:
            if src.crs.to_epsg() != 4326:
                raise ValueError("label CRS is %s, expected EPSG:4326" % src.crs)
            order = invalid_mask.sort_key_for_locality(src, boxes)
            scored = invalid_mask.invalid_fractions(src, boxes[order])

        # Undo the locality ordering so results line up with the table again.
        restore = np.argsort(order)
        table = table.assign(
            invalid_frac=scored["frac"][restore],
            invalid_npix=scored["npix"][restore],
            invalid_boundless=scored["boundless"][restore],
            label_covers=scored["covered"][restore],
            old_invalid_frac=[invalid_mask.parse_old_percent(r) for r in table["reason"]],
        )
        return {"pair": pair, "rows": table, "error": "",
                "max_residual": scored["max_residual"]}
    except Exception as error:  # noqa: BLE001 - one bad pair must not kill the run
        return {"pair": pair, "rows": None,
                "error": "%s: %s" % (type(error).__name__, str(error)[:200])}


def summarise(table: pd.DataFrame, counts: dict[str, int], absent: list[str],
              errors: list[dict], max_residual: float) -> None:
    print("ELIGIBILITY (pairs with an SCL-off GPKG)")
    for category, n in sorted(counts.items()):
        mark = "eligible" if category == "accepted" else "EXCLUDED"
        print("  %-18s %5d  %s" % (category, n, mark))
    print("  %-18s %5d  %s" % ("absent from base", len(absent), sorted(absent)))
    print("  %-18s %5d  %s  malformed geometry, dropped"
          % ("excluded outright", len(MALFORMED_GEOMETRY_PAIRS),
             sorted(MALFORMED_GEOMETRY_PAIRS)))
    print()
    covered = table["label_covers"]
    print("WINDOWS")
    print("  rescuable (cloud15)      %9d" % len(table))
    print("  label covers them        %9d" % int(covered.sum()))
    print("  NOT covered by the label %9d  (%.2f%%) -- unscoreable, not cloudy"
          % (int((~covered).sum()), 100 * float((~covered).mean())))
    print("  max sub-pixel residual   %9.2e" % max_residual)
    print("  partly-filled reads      %9d" % int(table["invalid_boundless"].sum()))
    if errors:
        print("  ERRORS                   %9d  e.g. %s" % (len(errors), errors[0]["error"]))
    print()
    print("INVALID FRACTION under the new mask (covered windows only)")
    frac = table.loc[covered, "invalid_frac"]
    print("  mean %.4f | median %.4f | p10 %.4f | p90 %.4f"
          % (frac.mean(), frac.median(), frac.quantile(0.10), frac.quantile(0.90)))
    print("  old gate mean %.4f  ->  new mask mean %.4f"
          % (table.loc[covered, "old_invalid_frac"].mean(), frac.mean()))
    print()
    print("WOULD RESCUE, by threshold")
    print("  (%% of the %d covered windows; the %d uncovered can never rescue)"
          % (len(frac), int((~covered).sum())))
    for threshold in (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.50):
        n = int((frac <= threshold).sum())
        print("  <= %.2f   %9d  (%.2f%% of covered, %.2f%% of all eligible)"
              % (threshold, n, 100 * n / max(len(frac), 1),
                 100 * n / max(len(table), 1)))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    parser.add_argument("--limit", type=int, default=None,
                        help="score only the first N eligible pairs (smoke run)")
    parser.add_argument("--write", action="store_true",
                        help="write the manifest; without it, only summarise")
    args = parser.parse_args()

    eligible, counts, absent = partition(scene_categories())
    if args.limit:
        eligible = eligible[:args.limit]

    frames, errors, max_residual = [], [], 0.0
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        futures = {pool.submit(score_pair, pair): pair for pair in eligible}
        for done, future in enumerate(as_completed(futures), 1):
            result = future.result()
            if result["error"]:
                errors.append(result)
            elif result["rows"] is not None:
                frames.append(result["rows"])
                max_residual = max(max_residual, result.get("max_residual", 0.0))
            if done % 250 == 0:
                print("  %d/%d pairs" % (done, len(eligible)), flush=True)

    table = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    if not args.limit:
        if len(table) != EXPECTED_SCORED:
            raise AssertionError("expected %d scored windows, got %d"
                                 % (EXPECTED_SCORED, len(table)))
    if max_residual >= invalid_mask.MAX_SUBPIXEL_RESIDUAL:
        raise AssertionError("sub-pixel residual %.3g exceeds the guard; the "
                             "label grid moved" % max_residual)
    scored_npix = table.loc[table["label_covers"], "invalid_npix"]
    if not (scored_npix == invalid_mask.CHIP_PIXELS).all():
        raise AssertionError("a covered window did not read exactly %d pixels"
                             % invalid_mask.CHIP_PIXELS)
    if table["old_invalid_frac"].isna().any():
        raise AssertionError("a reason string carried no parseable percentage")
    uncovered = int((~table["label_covers"]).sum())
    if not args.limit and uncovered != EXPECTED_UNSCOREABLE:
        raise AssertionError("expected %d windows outside their label, got %d"
                             % (EXPECTED_UNSCOREABLE, uncovered))
    if table.loc[table["label_covers"], "invalid_frac"].isna().any():
        raise AssertionError("a covered window produced no invalid fraction")

    table["qc_scene_category"] = "accepted"
    table["requires_rechipping"] = True
    table["invalid_mask_rule"] = invalid_mask.MASK_RULE
    table["assessed_at"] = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")

    summarise(table, counts, absent, errors, max_residual)

    if not args.write:
        print("\n(dry run; pass --write to save)")
        return 0

    # The hard rule and the never-cut caveat, made mechanical.
    if not (table["qc_scene_category"] == "accepted").all():
        raise AssertionError("a rescue candidate came from a non-accepted scene")
    if not (table["verdict"] == "cloud15").all():
        raise AssertionError("a non-cloud15 window reached the manifest")
    if "chip_id" in table.columns:
        raise AssertionError("rescue candidates must not carry a chip_id")

    paths.ensure_out()
    target = paths.MANIFESTS / "rescue_candidates.parquet"
    table.to_parquet(target, index=False, compression="zstd")
    print("\nwrote %s  (%d rows, %.1f MiB)"
          % (target, len(table), target.stat().st_size / 2**20))
    return 0


if __name__ == "__main__":
    sys.exit(main())
