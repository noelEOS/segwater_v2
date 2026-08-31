#!/usr/bin/env python3
"""Combine the surviving corpus chips and the recovered windows into one table.

This is the manifest the v3 memmaps are built from. The memmaps are rebuilt
from scratch, so the old row indices carry no authority; what matters is that
every chip can be located in its pair raster and that we can tell afterwards
where each one came from.

Two populations, one schema:

* **existing** -- chips already in the corpus that survived both the QC
  verdicts and the new invalid mask. Their old memmap rows are kept as
  ``legacy_chipbased_row`` / ``legacy_pairbased_row`` purely so the rebuild can
  be verified against the old arrays; nothing should index with them.
* **recovered** -- SCL-off windows the new mask clears. These were never cut,
  so they get a fresh ``chip_id`` continuing their pair's numbering and inherit
  their pair's split. They have no round-3 evidence, because rescue is only
  allowed from scenes that never reached round 3.

``chip_origin`` distinguishes them, and is the column to filter on when
verifying the rebuild: only ``existing`` chips have something to compare
against.
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys

import numpy as np
import pandas as pd

import paths

EXPECTED_EXISTING = 1_264_188
THRESHOLD = 0.15

# Columns a recovered window cannot have, and why.
PHASE3_COLUMNS = ["phase3_source_category", "phase3_scene_mode", "phase3_action",
                  "phase3_committed_at", "phase3_revision", "phase3_reviewed",
                  "phase3_scene_reject"]


def load_existing() -> pd.DataFrame:
    base = pd.read_parquet(
        paths.require(paths.MANIFESTS / "dataset_v3_base.parquet", "base table"))
    keep = base[base["v3_status"] == "include"].copy()
    if len(keep) != EXPECTED_EXISTING:
        raise AssertionError("expected %d included chips, found %d"
                             % (EXPECTED_EXISTING, len(keep)))
    keep = keep.rename(columns={"chipbased_row": "legacy_chipbased_row",
                                "pairbased_row": "legacy_pairbased_row"})
    keep["chip_origin"] = "existing"
    return keep


def load_recovered(existing: pd.DataFrame) -> pd.DataFrame:
    rescue = pd.read_parquet(
        paths.require(paths.MANIFESTS / "rescue_candidates.parquet", "rescue candidates"))
    got = rescue[rescue["label_covers"] & (rescue["invalid_frac"] <= THRESHOLD)].copy()

    # Fresh chip ids, continuing each pair's existing numbering so
    # (pair_name, chip_id) stays one uniform key across both populations.
    highest = existing.groupby("pair_name")["chip_id"].max()
    got = got.sort_values(["pair_name", "row", "col"]).reset_index(drop=True)
    start = got["pair_name"].map(highest).fillna(0).astype("int64")
    got["chip_id"] = (start + got.groupby("pair_name").cumcount() + 1).astype("int64")

    # Inherit the pair's split -- but only where a pair HAS one split.
    #
    # `pairbased_split` is assigned per pair and is uniform within it (verified:
    # 0 of 3,020 pairs span more than one), so inheriting is exactly right.
    #
    # `chipbased_split` is assigned per chip and is NOT uniform: 3,009 of 3,020
    # pairs span two or three splits. Taking the pair's mode would put nearly
    # every recovered chip in train simply because train is the majority
    # everywhere, which is an artefact of the aggregation and not a split
    # assignment. It is left null and must be assigned deliberately.
    by_pair = existing.groupby("pair_name")["pairbased_split"].agg(
        lambda s: s.mode().iat[0] if len(s.mode()) else pd.NA)
    got["pairbased_split"] = got["pair_name"].map(by_pair)
    got["chipbased_split"] = pd.NA

    spans = existing.groupby("pair_name")["pairbased_split"].nunique()
    if (spans > 1).any():
        raise AssertionError("%d pairs span more than one pairbased_split; "
                             "inheriting it is no longer safe" % int((spans > 1).sum()))

    # Scene-level QC context, known because rescue implies an accepted scene.
    got["qc_scene_category"] = "accepted"
    round1 = existing.drop_duplicates("pair_name").set_index("pair_name")
    got["qc_round1_category"] = got["pair_name"].map(round1["qc_round1_category"])
    got["scl11_released"] = got["pair_name"].map(round1["scl11_released"])
    for column in ("system:index_s1", "system:index_s2", "system:time_start_s1",
                   "intersects_gcl_coastline"):
        got[column] = got["pair_name"].map(round1[column])

    # No round-3 evidence exists for these, by construction.
    for column in PHASE3_COLUMNS:
        got[column] = pd.NA
    got["phase3_reviewed"] = False
    got["phase3_scene_reject"] = False

    got["v3_status"] = "include"
    got["v3_label_variant"] = "parent"
    got["v3_stage"] = "rescue"
    got["v3_reason"] = "SCL-off window cleared by the new mask (<=%.2f)" % THRESHOLD
    got["legacy_chipbased_row"] = pd.NA
    got["legacy_pairbased_row"] = pd.NA
    got["chip_origin"] = "recovered"
    return got


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()

    existing = load_existing()
    recovered = load_recovered(existing)

    columns = [c for c in existing.columns if c in set(existing.columns)]
    manifest = pd.concat(
        [existing[columns], recovered.reindex(columns=columns)], ignore_index=True)
    manifest["v3_updated_at"] = dt.datetime.now(
        dt.timezone.utc).isoformat(timespec="seconds")

    if manifest.duplicated(["pair_name", "chip_id"]).any():
        raise AssertionError("(pair_name, chip_id) is not unique in the manifest")
    if (manifest["v3_status"] != "include").any():
        raise AssertionError("the manifest must contain only included chips")
    if (manifest["invalid_frac"] > THRESHOLD).any():
        raise AssertionError("a chip over the invalid threshold reached the manifest")
    got = manifest["chip_origin"] == "recovered"
    if manifest.loc[got, "phase3_reviewed"].any():
        raise AssertionError("a recovered window carries round-3 evidence")
    if (manifest.loc[got, "qc_scene_category"] != "accepted").any():
        raise AssertionError("a recovered window came from a non-accepted scene")
    if manifest.loc[~got, "legacy_chipbased_row"].isna().any():
        raise AssertionError("an existing chip lost its legacy memmap row")
    if manifest["pairbased_split"].isna().any():
        raise AssertionError("a chip has no pairbased_split")
    # chipbased_split is deliberately null for recovered chips; see load_recovered.
    if manifest.loc[~got, "chipbased_split"].isna().any():
        raise AssertionError("an existing chip lost its chipbased_split")

    print("V3 MANIFEST")
    print("  chips                    %9d" % len(manifest))
    print("  pairs                    %9d" % manifest["pair_name"].nunique())
    print()
    print("  by origin: %s" % manifest["chip_origin"].value_counts().to_dict())
    print("  by variant: %s" % manifest["v3_label_variant"].value_counts().to_dict())
    print()
    print("  pair-based split:  %s"
          % manifest["pairbased_split"].value_counts().to_dict())
    print("  chip-based split:  %s  (null for %d recovered)"
          % (manifest["chipbased_split"].value_counts().to_dict(),
             int(manifest["chipbased_split"].isna().sum())))
    print()
    print("  invalid_frac  mean %.4f | median %.4f | max %.4f"
          % (manifest["invalid_frac"].mean(), manifest["invalid_frac"].median(),
             manifest["invalid_frac"].max()))
    print()
    print("  recovered chips need cutting from the pair rasters before the")
    print("  memmaps can be built; existing chips can be verified against the old ones.")

    if not args.write:
        print("\n(dry run; pass --write to save)")
        return 0

    target = paths.MANIFESTS / "dataset_v3_manifest.parquet"
    manifest.to_parquet(target, index=False, compression="zstd")
    print("\nwrote %s  (%d rows, %.1f MiB)"
          % (target, len(manifest), target.stat().st_size / 2**20))
    return 0


if __name__ == "__main__":
    sys.exit(main())
