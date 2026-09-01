#!/usr/bin/env python3
"""Carry the stored tide values onto the manifest for chips that already have them.

Tide is **per chip**, not per pair: it is FES2022b evaluated at the nearest
ocean-edge point to each chip's own centroid, so it varies across a pair
(3,365 of 3,385 pairs hold more than one distinct value, with a within-pair
range of median 6.4 cm). It therefore cannot be inherited from a pair the way
``pairbased_split`` was.

The 1,262,581 chips that already existed in the corpus have tide computed and
stored, so this joins it on ``(pair_name, chip_id)`` rather than recomputing.
The 93,121 recovered windows were never chipped and have no stored value; they
are left null here and must be computed from FES against their own centroids.

⚠️ This join is only safe because recovered chips now occupy a disjoint id
space (``chip_id >= 100000``). Under the previous scheme 27,750 of them carried
ids that also exist in the source corpus, and this join would have silently
attached another chip's tide to them. The build asserts the separation rather
than trusting it.

Sign convention, checked rather than assumed. A project note warns that the
parquet's ``s1_s2_tide_level_sat_delta_first_round`` is *signed* while the
pair-level CSV of the same name is ``|s1 - s2|``, and that comparisons to the
10 cm threshold must ``.abs()`` first. **In this parquet the column is already
an absolute value**: it matches ``|s1 - s2|`` to 1.8e-15, holds zero negatives
across all 1,649,079 rows, and spans 0.000 to 9.996 -- while the raw
``s1 - s2`` is negative in 735,999 of them. The warning may hold for another
parquet in the lineage; it does not hold for ``..._RESPLIT_PATHS_FIXED``.
Taking ``.abs()`` is harmless either way, so downstream code should still do it.
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys

import pandas as pd
import pyarrow.parquet as pq

import paths

EXPECTED_CHIPS = 1_357_354
EXPECTED_EXISTING = 1_264_188
EXPECTED_RECOVERED = 93_166
RECOVERED_ID_OFFSET = 100_000

SOURCE = ("Global_Sen12_Coast_2017_2024_CHIPS_DATABASE_with_splits_w_metadata"
          "_w_tides_w_path_passedQC_memmap_selected-V2-w_histograms-CONFIRMED"
          "-RESPLIT_PATHS_FIXED.parquet")

# The per-chip tide columns, and the pair-level first-round columns the
# retention gate actually used. Both are carried: the first-round set is what
# gated the corpus, the plain set is the later per-chip re-annotation.
TIDE_COLUMNS = [
    "s1_tide_level_sat", "s1_tide_level_sat_phase",
    "s2_tide_level_sat", "s2_tide_level_sat_phase",
    "s1_tide_level_sat_first_round", "s1_tide_level_sat_phase_first_round",
    "s2_tide_level_sat_first_round", "s2_tide_level_sat_phase_first_round",
    "s1_s2_tide_level_sat_delta_first_round",
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()

    target = paths.require(paths.MANIFESTS / "dataset_v3_manifest.parquet", "v3 manifest")
    manifest = pd.read_parquet(target)
    if len(manifest) != EXPECTED_CHIPS:
        raise AssertionError("manifest is %d chips, expected %d"
                             % (len(manifest), EXPECTED_CHIPS))

    source = pq.read_table(paths.require(paths.DB_SLIM / SOURCE, "source parquet"),
                           columns=["pair_name", "chip_id"] + TIDE_COLUMNS).to_pandas()
    if source.duplicated(["pair_name", "chip_id"]).any():
        raise AssertionError("(pair_name, chip_id) is not unique in the source")

    # The join is only meaningful because the id spaces are disjoint. Check it
    # here too: this is the file the collision would have corrupted.
    recovered = manifest["chip_origin"] == "recovered"
    if (manifest.loc[recovered, "chip_id"] < RECOVERED_ID_OFFSET).any():
        raise AssertionError("a recovered chip_id is below the offset; the tide "
                             "join would attach another chip's values")

    stale = [c for c in TIDE_COLUMNS if c in manifest.columns]
    manifest = manifest.drop(columns=stale)
    merged = manifest.merge(source, on=["pair_name", "chip_id"], how="left", validate="1:1")
    if len(merged) != EXPECTED_CHIPS:
        raise AssertionError("join changed the row count to %d" % len(merged))

    has_tide = merged["s1_tide_level_sat"].notna()
    print("TIDE over %d chips" % len(merged))
    print("  with a stored value      %9d  (%.2f%%)"
          % (int(has_tide.sum()), 100 * float(has_tide.mean())))
    print("  without                  %9d" % int((~has_tide).sum()))
    print()
    print("  by origin:")
    for origin in ("existing", "recovered"):
        m = merged["chip_origin"] == origin
        print("    %-10s %9d chips, %9d with tide (%.2f%%)"
              % (origin, int(m.sum()), int((m & has_tide).sum()),
                 100 * float(has_tide[m].mean()) if m.any() else 0.0))
    print()
    missing_existing = int(((merged["chip_origin"] == "existing") & ~has_tide).sum())
    print("  existing chips MISSING tide: %d" % missing_existing)
    if missing_existing:
        print("    (these had no stored value in the source either)")

    # Every recovered chip must be null: they have no stored value by
    # construction, and a non-null one would mean the join found something it
    # should not have.
    if merged.loc[recovered, "s1_tide_level_sat"].notna().any():
        raise AssertionError("a recovered chip picked up a stored tide value")

    print()
    print("  delta sign check (parquet column is SIGNED):")
    delta = merged["s1_s2_tide_level_sat_delta_first_round"].dropna()
    print("    min %.2f  max %.2f  | negative: %d of %d"
          % (delta.min(), delta.max(), int((delta < 0).sum()), len(delta)))
    print("    |delta| <= 10 cm: %d (%.2f%%)"
          % (int((delta.abs() <= 10).sum()), 100 * float((delta.abs() <= 10).mean())))

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
