#!/usr/bin/env python3
"""Write a manifest holding exactly the chips that are in the v3 memmaps.

The full manifest carries every chip the build ever considered, including those
that failed a gate and the 6,023 held out as an evaluation buffer. This one
holds only the 1,322,788 chips that were actually written, so a consumer cannot
index a row that does not exist.

**Membership comes from ``memmap_row_index_v3.parquet``, not from a predicate.**
That file is what the memmap build itself emitted, so it is the authority on
what was written. Reconstructing the set from ``split_v3.notna()`` would agree
today -- verified -- but it would be a second definition that could drift from
the arrays if a future build changed its selection. Deriving from the index
means this file is wrong only if the index is.

Each row carries its ``row`` position within its split's memmap, so
``(split_v3, row)`` addresses the array directly. The row order is the one the
build recorded: row i of ``{split}.memmap`` is the i-th row of the manifest
filtered to that split, in manifest positional order.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys

import pathlib

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

import paths

EXPECTED_ROWS = 1_322_788
EXPECTED_SPLITS = {"train": 925_123, "val": 265_952, "test": 131_713}
CHIP_PIXELS = 224 * 224


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--memmap-dir", default=None,
                        help="where the memmaps and build_manifest.json live "
                             "(default: ~/memmaps_v3)")
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()

    memmap_dir = (pathlib.Path(args.memmap_dir) if args.memmap_dir
                  else pathlib.Path.home() / "memmaps_v3")

    index = pd.read_parquet(
        paths.require(paths.MANIFESTS / "memmap_row_index_v3.parquet", "memmap row index"))
    manifest = pd.read_parquet(
        paths.require(paths.MANIFESTS / "dataset_v3_manifest.parquet", "v3 manifest"))

    if len(index) != EXPECTED_ROWS:
        raise AssertionError("row index is %d rows, expected %d"
                             % (len(index), EXPECTED_ROWS))

    # Join the full manifest onto the index, keyed on the index so the result is
    # exactly the written chips and nothing else. The index repeats some
    # manifest columns; drop its copies so the manifest stays the single source
    # for everything except membership, split and row.
    keep = ["pair_name", "chip_id", "split_v3", "row"]
    merged = index[keep].merge(
        manifest.drop(columns=[c for c in ("split_v3",) if c in manifest.columns]),
        on=["pair_name", "chip_id"], how="left", validate="1:1")
    if len(merged) != EXPECTED_ROWS:
        raise AssertionError("join changed the row count to %d" % len(merged))
    missing = merged["passes_all_gates"].isna()
    if missing.any():
        raise AssertionError("%d memmap chips are absent from the manifest"
                             % int(missing.sum()))

    # Everything written must have passed every gate.
    if not merged["passes_all_gates"].all():
        raise AssertionError("%d memmap chips do not pass all gates"
                             % int((~merged["passes_all_gates"]).sum()))

    counts = merged["split_v3"].value_counts().to_dict()
    if counts != EXPECTED_SPLITS:
        raise AssertionError("split sizes changed: %r" % counts)

    # (split, row) must address each array exactly once, contiguously from 0.
    for split, group in merged.groupby("split_v3"):
        rows = group["row"].to_numpy()
        if rows.min() != 0 or rows.max() != len(group) - 1 or len(set(rows)) != len(group):
            raise AssertionError("%s rows are not a contiguous 0..n-1 permutation" % split)

    # Cross-check the arrays themselves: file size must equal rows x chip bytes.
    build = json.loads((memmap_dir / "build_manifest.json").read_text()) \
        if (memmap_dir / "build_manifest.json").exists() else None
    if build:
        channels = len(build["channels"])
        itemsize = 2 if build["dtype"] == "float16" else 4
        for split, expected_n in EXPECTED_SPLITS.items():
            path = memmap_dir / ("%s.memmap" % split)
            if not path.exists():
                print("  NOTE %s not present at %s" % (split, memmap_dir))
                continue
            want = expected_n * channels * CHIP_PIXELS * itemsize
            got = path.stat().st_size
            if got != want:
                raise AssertionError("%s.memmap is %d bytes, expected %d for %d chips"
                                     % (split, got, want, expected_n))

    merged = merged.sort_values(["split_v3", "row"]).reset_index(drop=True)

    print("MEMMAP MANIFEST  %s chips, %d columns" % ("{:,}".format(len(merged)),
                                                     len(merged.columns)))
    print()
    for split in ("train", "val", "test"):
        s = merged[merged["split_v3"] == split]
        origin = s["chip_origin"].value_counts().to_dict()
        print("  %-6s %9s chips  rows 0..%s  %s"
              % (split, "{:,}".format(len(s)), "{:,}".format(len(s) - 1), origin))
    print()
    print("  pairs %s | label variant %s"
          % (merged["pair_name"].nunique(), merged["v3_label_variant"].value_counts().to_dict()))
    valid = merged["n_water"] + merged["n_land"]
    print("  water share of valid px: median %.4f"
          % (merged["n_water"] / valid).median())
    print("  every chip passes all gates: %s" % bool(merged["passes_all_gates"].all()))
    if build:
        print("  memmap byte sizes match the row counts: yes")

    if not args.write:
        print("\n(dry run; pass --write to save)")
        return 0

    table = pa.Table.from_pandas(merged, preserve_index=False)
    table = table.replace_schema_metadata({
        "segwater.memmap_manifest": json.dumps({
            "rows": len(merged),
            "splits": EXPECTED_SPLITS,
            "membership": "exactly the chips in memmap_row_index_v3.parquet, "
                          "which the memmap build emitted",
            "addressing": "row i of {split}.memmap is the row with "
                          "(split_v3 == split, row == i)",
            "memmap_dir": str(memmap_dir),
            "built": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        }),
    })
    target = paths.MANIFESTS / "dataset_v3_memmap_manifest.parquet"
    pq.write_table(table, target, compression="zstd")
    print("\nwrote %s  (%d rows, %.1f MiB)"
          % (target, len(merged), target.stat().st_size / 2**20))
    return 0


if __name__ == "__main__":
    sys.exit(main())
