#!/usr/bin/env python3
"""Put tide on every chip: stored values where they exist, computed where not.

Tide is **per chip**, not per pair: it is FES2022b evaluated at the nearest
ocean-edge point to each chip's own centroid, so it varies across a pair
(3,365 of 3,385 pairs hold more than one distinct value, with a within-pair
range of median 6.4 cm). It therefore cannot be inherited from a pair the way
``pairbased_split`` was.

The 1,264,188 chips that already existed in the corpus have tide stored from
the original build, so this joins it on ``(pair_name, chip_id)`` rather than
recomputing -- there is no reason to regenerate what is already there.

The 93,166 recovered windows were never chipped and have no stored value.
``compute_tide_recovered.py`` generates theirs from FES2022b at their own
centroids, and this folds that in. ``tide_source`` records which route each chip
took, so the two are never silently conflated.

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

import numpy as np
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

    stale = [c for c in TIDE_COLUMNS + ["tide_point_lat", "tide_point_lon",
                                        "tide_source"] if c in manifest.columns]
    manifest = manifest.drop(columns=stale)
    merged = manifest.merge(source, on=["pair_name", "chip_id"], how="left", validate="1:1")

    # Recovered chips were never in the source, so their tide is computed
    # separately by compute_tide_recovered.py using the chain verified in
    # verify_tide_reproduction.py (600 chips, max 0.097 cm level error,
    # 1200/1200 exact phase). Fold it in where present.
    computed_path = paths.MANIFESTS / "computed_chip_tide.parquet"
    if computed_path.exists():
        computed = pd.read_parquet(computed_path).set_index(["pair_name", "chip_id"])
        index = pd.MultiIndex.from_frame(merged[["pair_name", "chip_id"]])
        fill = computed.reindex(index)
        for column in ("s1_tide_level_sat", "s1_tide_level_sat_phase",
                       "s2_tide_level_sat", "s2_tide_level_sat_phase"):
            merged[column] = merged[column].combine_first(
                pd.Series(fill[column].to_numpy(), index=merged.index))
        merged["tide_point_lat"] = fill["tide_point_lat"].to_numpy()
        merged["tide_point_lon"] = fill["tide_point_lon"].to_numpy()
        # A chip took the computed route if the source join left it null.
        was_null = merged["s1_tide_level_sat"].isna() | (
            pd.MultiIndex.from_frame(merged[["pair_name", "chip_id"]]).isin(computed.index))
        merged["tide_source"] = np.where(was_null, "computed_fes2022b", "stored")
        print("folded in %d computed rows" % len(computed))
    if len(merged) != EXPECTED_CHIPS:
        raise AssertionError("join changed the row count to %d" % len(merged))

    has_tide = merged["s1_tide_level_sat"].notna()
    print("TIDE over %d chips" % len(merged))
    print("  with a tide value        %9d  (%.2f%%)"
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

    # Recovered chips must take their values from the computed table, never
    # from the source join -- a stored hit would mean the id spaces overlapped.
    if "tide_source" in merged.columns:
        if not merged.loc[recovered, "tide_source"].eq("computed_fes2022b").all():
            raise AssertionError("a recovered chip did not take the computed tide")
        if merged["s1_tide_level_sat"].isna().any():
            raise AssertionError("%d chips still have no tide level"
                                 % int(merged["s1_tide_level_sat"].isna().sum()))

    print()
    print("  delta sign check (see the docstring: absolute in THIS parquet):")
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
