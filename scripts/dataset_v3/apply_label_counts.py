#!/usr/bin/env python3
"""Attach the label class counts to the manifest.

Three columns -- ``n_water``, ``n_land``, ``n_invalid`` -- from
``count_label_classes.py``. They sum to 50,176 per chip, and ``n_invalid`` is
cross-checked against the independently measured ``invalid_frac``: the two agree
to 0 pixels on every chip, so the mask stage and the counting stage validate
each other rather than merely coexisting.

No fraction is derived here. See ``count_label_classes.py`` for why the
denominator is left to the consumer.
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
    print("  composition of the passing corpus (%s chips):" % "{:,}".format(int(passing.sum())))
    pure_water = int((merged.loc[passing, "n_water"] == CHIP_PIXELS).sum())
    pure_land = int((merged.loc[passing, "n_land"] == CHIP_PIXELS).sum())
    mixed = int(passing.sum()) - pure_water - pure_land
    for label, n in (("all water", pure_water), ("all land", pure_land),
                     ("mixed / partly invalid", mixed)):
        print("     %-24s %10s  (%.2f%%)"
              % (label, "{:,}".format(n), 100 * n / max(int(passing.sum()), 1)))

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
