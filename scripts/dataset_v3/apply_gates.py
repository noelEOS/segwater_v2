#!/usr/bin/env python3
"""Resolve every gate into one column: ``passes_all_gates``.

A chip is **dual-sensor**: its input is SAR and its label is optical. Each
source must clear its own gates, and the chip is ready for model development
only when both have.

**Label side** (Sentinel-2 derived):

1. **QC verdict** -- round 1 rejected whole scenes; round 3 rejected chips,
   corrected them with the B8<400 variant, or kept the original.
2. **Invalid mask** -- the share of the chip the new mask calls invalid must not
   exceed the threshold.

**Input side** (Sentinel-1):

3. **S1 completeness** -- every pixel must carry an observation. A chip fails on
   any zero, whether it is *fill* (0 in both bands, genuine absence at a
   footprint edge) or *damage* (0 in one band, lost to the Earth Engine
   ``log10`` export defect).

Fill and damage differ in provenance, not in fitness: either way the model would
read absence where it expects backscatter. The distinction is preserved in
``s1_status`` because it matters for deciding whether to re-export, but it does
not change whether a chip can be used.

Gates 1 and 2 are already resolved into ``v3_status`` and the manifest holds
only ``include`` chips, so in practice this column adds gate 3 and states the
conjunction in one place. That matters: a consumer filtering the manifest alone
would silently pick up chips with missing SAR input.

``passes_all_gates`` means **qualified**, not **cut**. 93,156 qualifying chips
are rescued windows that were never chipped and have no array in any memmap yet.
That is a different property and it already has a column -- ``chip_origin`` --
so the two are kept separate rather than collapsed into one flag that would be
wrong for either the corpus builder or the model loader.

1,652 chips fail the input gate: 871 ``damaged_vh`` and 781 ``fill``.
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys

import pandas as pd

import paths

EXPECTED_CHIPS = 1_357_354
# clean only. Fill and damage both fail the input gate; see the docstring.
EXPECTED_PASSING = 1_355_702



def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-fill", action="store_true",
                        help="let chips with fill pixels pass. Off by default: "
                             "fill is missing SAR input, so the model would read "
                             "absence where it expects backscatter")
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()

    target = paths.require(paths.MANIFESTS / "dataset_v3_manifest.parquet", "v3 manifest")
    manifest = pd.read_parquet(target)
    if len(manifest) != EXPECTED_CHIPS:
        raise AssertionError("manifest is %d chips, expected %d"
                             % (len(manifest), EXPECTED_CHIPS))
    for column in ("v3_status", "invalid_frac", "s1_status"):
        if column not in manifest.columns:
            raise AssertionError("%s missing; run the earlier stages first" % column)

    qc_ok = manifest["v3_status"] == "include"
    # The input gate: S1 must be complete. Fill and damage both mean a pixel
    # with no observation, so both fail.
    s1_ok = manifest["s1_status"] == "clean"
    if args.allow_fill:
        s1_ok |= manifest["s1_status"] == "fill"
    passes = qc_ok & s1_ok

    manifest["passes_all_gates"] = passes

    print("GATES over %d chips" % len(manifest))
    print("  LABEL side  QC verdict + invalid mask          %9d" % int(qc_ok.sum()))
    print("  INPUT side  S1 complete (no fill, no damage)   %9d" % int(s1_ok.sum()))
    print("  ---")
    print("  passes_all_gates                               %9d  (%.4f%%)"
          % (int(passes.sum()), 100 * float(passes.mean())))
    print()
    print("  failing, by reason:")
    for status, n in manifest.loc[~passes, "s1_status"].value_counts().items():
        print("    s1_status=%-14s %9d" % (status, n))
    print()
    print("  of the passing chips:")
    print("    already cut (in an old memmap)   %9d"
          % int((passes & (manifest["chip_origin"] == "existing")).sum()))
    print("    need cutting (rescued windows)   %9d"
          % int((passes & (manifest["chip_origin"] == "recovered")).sum()))
    print()
    print("    pair-based split: %s"
          % manifest.loc[passes, "pairbased_split"].value_counts().to_dict())
    print("    label variant:    %s"
          % manifest.loc[passes, "v3_label_variant"].value_counts().to_dict())

    # Every passing chip is s1_status == 'clean', which is exactly the set
    # build_histograms.py covered, so the histogram table and the passing set
    # coincide by construction.
    no_histogram = int((passes & (manifest["s1_status"] != "clean")).sum())
    if no_histogram:
        raise AssertionError("%d passing chips have no S1 histogram" % no_histogram)

    if not args.allow_fill and int(passes.sum()) != EXPECTED_PASSING:
        raise AssertionError("expected %d passing chips, got %d"
                             % (EXPECTED_PASSING, int(passes.sum())))

    if not args.write:
        print("\n(dry run; pass --write to save)")
        return 0

    manifest["v3_updated_at"] = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    manifest.to_parquet(target, index=False, compression="zstd")
    print("\nwrote %s  (%d rows, %d cols, %.1f MiB)"
          % (target, len(manifest), len(manifest.columns), target.stat().st_size / 2**20))
    return 0


if __name__ == "__main__":
    sys.exit(main())
