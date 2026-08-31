#!/usr/bin/env python3
"""Attach the per-chip S1 failure mode to the manifest.

``audit_s1_damage.py`` counts pixels; this turns those counts into a readable
status so a consumer can filter on the failure mode without re-deriving it.

The distinction that matters is what a zero means. S1 nodata is encoded as
code 0, and the quantizer never emits 0 (it clamps to [1, 65535]), so every zero
was written by ``unmask(0)`` at submit. But two very different things produce
one:

* **fill** -- 0 in *both* bands. No S1 observation at that pixel, which is the
  ordinary footprint edge. Legitimate.
* **damage** -- 0 in *one* band with data in the other. The sensor observed
  something and the export lost it. See ``docs/dataset_v3/S1_LOG10_EXPORT_DEFECT.md``.

``s1_status`` is one of:

``clean``          neither; the chip is intact.
``fill``           has fill pixels, no damage. Usable; the fill is real absence.
``damaged_vh``     VH lost pixels VV kept.
``damaged_vv``     VV lost pixels VH kept.
``damaged_both``   both bands lost pixels somewhere in the chip.

A chip with both fill and damage is reported as damaged: the damage is what
decides whether it can be used, and the fill count is still available in the
audit table.
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys

import pandas as pd

import paths

EXPECTED_CHIPS = 1_357_354
CHIP_PIXELS = 224 * 224

# Verified 2026-09-01 on the shipped corpus. Asserted so a re-export that
# changes the picture cannot pass unnoticed.
EXPECTED_STATUS = {"clean": 1_355_702, "fill": 781, "damaged_vh": 871}


def classify(audit: pd.DataFrame) -> pd.DataFrame:
    has_fill = audit["n_fill"] > 0
    has_vv = audit["n_orphan_vv"] > 0
    has_vh = audit["n_orphan_vh"] > 0

    status = pd.Series("clean", index=audit.index, dtype="object")
    status[has_fill] = "fill"
    # Damage overrides fill: it is what decides usability.
    status[has_vv & ~has_vh] = "damaged_vv"
    status[has_vh & ~has_vv] = "damaged_vh"
    status[has_vv & has_vh] = "damaged_both"

    orphan = audit["n_orphan_vv"] + audit["n_orphan_vh"]
    return pd.DataFrame({
        "pair_name": audit["pair_name"],
        "chip_id": audit["chip_id"],
        "s1_status": status,
        "s1_fill_frac": (audit["n_fill"] / audit["n_read"].clip(lower=1)).astype("float32"),
        "s1_damage_frac": (orphan / audit["n_read"].clip(lower=1)).astype("float32"),
    })


def summarise(table: pd.DataFrame) -> None:
    print("S1 STATUS over %d chips" % len(table))
    counts = table["s1_status"].value_counts()
    for status, n in counts.items():
        print("  %-14s %9d  (%.4f%%)" % (status, n, 100 * n / len(table)))
    print()
    damaged = table["s1_status"].str.startswith("damaged")
    if damaged.any():
        print("DAMAGED chips by pair:")
        for pair, n in table.loc[damaged, "pair_name"].value_counts().items():
            share = table.loc[damaged & (table["pair_name"] == pair), "s1_damage_frac"].mean()
            print("  %-12s %6d chips   mean %.1f%% of the chip lost" % (pair, n, 100 * share))
    print()
    filled = table["s1_status"] == "fill"
    if filled.any():
        print("FILL chips: %d across %d pairs, mean %.2f%% of the chip"
              % (int(filled.sum()), table.loc[filled, "pair_name"].nunique(),
                 100 * table.loc[filled, "s1_fill_frac"].mean()))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()

    manifest_path = paths.require(paths.MANIFESTS / "dataset_v3_manifest.parquet",
                                  "v3 manifest")
    manifest = pd.read_parquet(manifest_path)
    audit = pd.read_parquet(
        paths.require(paths.MANIFESTS / "s1_damage_audit.parquet", "S1 damage audit"))

    if len(manifest) != EXPECTED_CHIPS:
        raise AssertionError("manifest is %d chips, expected %d"
                             % (len(manifest), EXPECTED_CHIPS))
    if (audit["n_read"] != CHIP_PIXELS).any():
        raise AssertionError("a chip did not read exactly %d S1 pixels" % CHIP_PIXELS)

    table = classify(audit)
    counts = table["s1_status"].value_counts().to_dict()
    if counts != EXPECTED_STATUS:
        raise AssertionError("S1 status counts changed: %r (expected %r)"
                             % (counts, EXPECTED_STATUS))

    summarise(table)

    # Re-running must not accumulate suffixed duplicates.
    carried = [c for c in table.columns if c not in ("pair_name", "chip_id")]
    manifest = manifest.drop(columns=[c for c in carried if c in manifest.columns])
    merged = manifest.merge(table, on=["pair_name", "chip_id"], how="left", validate="1:1")
    if merged["s1_status"].isna().any():
        raise AssertionError("%d chips have no S1 audit row"
                             % int(merged["s1_status"].isna().sum()))
    if len(merged) != EXPECTED_CHIPS:
        raise AssertionError("join changed the row count to %d" % len(merged))

    if not args.write:
        print("\n(dry run; pass --write to save)")
        return 0

    merged["v3_updated_at"] = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    merged.to_parquet(manifest_path, index=False, compression="zstd")
    print("\nwrote %s  (%d rows, %d cols, %.1f MiB)"
          % (manifest_path, len(merged), len(merged.columns),
             manifest_path.stat().st_size / 2**20))
    return 0


if __name__ == "__main__":
    sys.exit(main())
