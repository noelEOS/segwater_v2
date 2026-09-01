#!/usr/bin/env python3
"""Attach the label class counts to the manifest.

Three columns -- ``n_water``, ``n_land``, ``n_invalid`` -- from
``count_label_classes.py``. They sum to 50,176 per chip, and ``n_invalid`` is
cross-checked against the independently measured ``invalid_frac``: the two agree
to 0 pixels on every chip, so the mask stage and the counting stage validate
each other rather than merely coexisting.

No fraction is derived here. See ``count_label_classes.py`` for why the
denominator is left to the consumer.

The composition summary uses a **1% tolerance**, not strict equality. Labels
carry speckle: 305,690 chips -- 22.55% of the passing corpus -- are 99% or more
one class yet not pure, differing by a median of 31 pixels out of 50,176, and
23,032 differ by exactly one. Calling those "mixed" nearly doubles the apparent
mixed fraction (48.5% strict against 26.0% at 1%) and describes the speckle
rather than the scene.

1% is where the curve flattens rather than an arbitrary round number: relaxing
0% to 1% reclassifies ~350k chips, while 1% to 5% moves only ~100k more. The
speckle sits well below 1%. Both the strict and tolerant counts are printed, so
the choice is visible instead of assumed.
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys

import pandas as pd

import paths

EXPECTED_CHIPS = 1_357_354
CHIP_PIXELS = 224 * 224
COUNT_COLUMNS = ["n_water", "n_land", "n_invalid"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()

    target = paths.require(paths.MANIFESTS / "dataset_v3_manifest.parquet", "v3 manifest")
    manifest = pd.read_parquet(target)
    counts = pd.read_parquet(
        paths.require(paths.MANIFESTS / "chip_label_counts.parquet", "label counts"))

    if len(manifest) != EXPECTED_CHIPS:
        raise AssertionError("manifest is %d chips, expected %d"
                             % (len(manifest), EXPECTED_CHIPS))

    manifest = manifest.drop(columns=[c for c in COUNT_COLUMNS if c in manifest.columns])
    merged = manifest.merge(counts, on=["pair_name", "chip_id"], how="left", validate="1:1")
    if len(merged) != EXPECTED_CHIPS:
        raise AssertionError("join changed the row count to %d" % len(merged))
    if merged["n_water"].isna().any():
        raise AssertionError("%d chips have no label counts"
                             % int(merged["n_water"].isna().sum()))

    total = merged[COUNT_COLUMNS].sum(axis=1)
    if not (total == CHIP_PIXELS).all():
        raise AssertionError("counts do not sum to %d on every chip" % CHIP_PIXELS)

    # n_invalid was measured twice, by different code reading different rasters
    # for the b8 chips. They must agree exactly.
    expected = (merged["invalid_frac"] * CHIP_PIXELS).round().astype(int)
    if (expected - merged["n_invalid"]).abs().max() > 1:
        raise AssertionError("n_invalid disagrees with invalid_frac")

    passing = merged["passes_all_gates"]
    valid = merged["n_water"] + merged["n_land"]
    print("LABEL COUNTS on %d chips" % len(merged))
    for column in COUNT_COLUMNS:
        print("  %-10s %18s px  (%.2f%%)"
              % (column, "{:,}".format(int(merged[column].sum())),
                 100 * merged[column].sum() / total.sum()))
    print()
    print("  water share of valid px, over chips that pass every gate:")
    share = (merged.loc[passing, "n_water"] / valid[passing]).dropna()
    print("     median %.4f  p10 %.4f  p25 %.4f  p75 %.4f  p90 %.4f"
          % (share.median(), share.quantile(0.10), share.quantile(0.25),
             share.quantile(0.75), share.quantile(0.90)))
    print()
    n_pass = int(passing.sum())
    share_pass = (merged.loc[passing, "n_water"] / valid[passing])
    print("  composition of the passing corpus (%s chips), by tolerance on the"
          % "{:,}".format(n_pass))
    print("  water share of valid pixels:")
    print("     %-9s %13s %13s %13s" % ("tolerance", "~all water", "~all land", "mixed"))
    for tol in (0.0, 0.005, 0.01, 0.05):
        water = int((share_pass >= 1 - tol).sum())
        land = int((share_pass <= tol).sum())
        mixed = n_pass - water - land
        mark = "  <- reported" if tol == 0.01 else ""
        print("     %-9s %13s %13s %13s  (%.1f%% mixed)%s"
              % ("%.1f%%" % (100 * tol), "{:,}".format(water), "{:,}".format(land),
                 "{:,}".format(mixed), 100 * mixed / n_pass, mark))
    print()
    minority = merged.loc[passing, ["n_water", "n_land"]].min(axis=1)
    near = ((share_pass >= 0.99) | (share_pass <= 0.01)) & (minority > 0)
    print("     %s chips (%.2f%%) are >=99%% one class but NOT pure -- speckle."
          % ("{:,}".format(int(near.sum())), 100 * float(near.mean())))
    print("     their minority pixels: median %d of %d; %s chips differ by exactly 1."
          % (minority[near].median(), CHIP_PIXELS,
             "{:,}".format(int((minority == 1).sum()))))

    if not args.write:
        print("\n(dry run; pass --write to save)")
        return 0

    merged["v3_updated_at"] = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    merged.to_parquet(target, index=False, compression="zstd")
    print("\nwrote %s  (%d rows, %d cols, %.1f MiB)"
          % (target, len(merged), len(merged.columns), target.stat().st_size / 2**20))
    return 0


if __name__ == "__main__":
    sys.exit(main())
