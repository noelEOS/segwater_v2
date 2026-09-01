#!/usr/bin/env python3
"""Resolve every gate into one column: ``passes_all_gates``.

A chip qualifies for the v3 corpus only if it clears all three gates, which
judge different things and are recorded separately as evidence:

1. **QC verdict** -- the label. Round 1 rejected whole scenes; round 3 rejected
   chips, corrected them with the B8<400 variant, or kept the original.
2. **Invalid mask** -- coverage. The share of the chip the new mask calls
   invalid must not exceed the threshold.
3. **S1 integrity** -- the input. The chip's Sentinel-1 pixels must not have
   been lost to the Earth Engine ``log10`` export defect.

Gates 1 and 2 are already resolved into ``v3_status`` and the manifest holds
only ``include`` chips, so in practice this column adds gate 3 and states the
conjunction in one place. That matters: a consumer that filtered the manifest
alone would silently pick up 871 chips whose VH is 88% missing.

``passes_all_gates`` means **qualified**, not **cut**. 93,156 qualifying chips
are rescued windows that were never chipped and have no array in any memmap yet.
That is a different property and it already has a column -- ``chip_origin`` --
so the two are kept separate rather than collapsed into one flag that would be
wrong for either the corpus builder or the model loader.

Fill is not a disqualifier. A chip with fill pixels has genuine absence of S1
observation at the footprint edge, which is real data about the world rather
than a defect; those pixels are ``0`` in both bands and the loader can mask them.
Only ``damaged_*`` chips fail, and only 871 do.
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys

import pandas as pd

import paths

EXPECTED_CHIPS = 1_357_354
# clean (1,355,702) + fill (781). Fill passes: see the module docstring.
EXPECTED_PASSING = 1_356_483
# Histograms were built over s1_status == 'clean' only, so the fill chips that
# pass the gates have none yet. Surfaced rather than left to be discovered.
EXPECTED_PASSING_WITHOUT_HISTOGRAM = 781


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--include-fill", action="store_true",
                        help="also fail chips that contain fill pixels; off by "
                             "default because fill is real absence, not damage")
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
    s1_ok = ~manifest["s1_status"].str.startswith("damaged")
    if args.include_fill:
        s1_ok &= manifest["s1_status"] != "fill"
    passes = qc_ok & s1_ok

    manifest["passes_all_gates"] = passes

    print("GATES over %d chips" % len(manifest))
    print("  QC verdict + invalid mask (v3_status=include)  %9d" % int(qc_ok.sum()))
    print("  S1 integrity (not damaged)                     %9d" % int(s1_ok.sum()))
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

    # The histogram build covered clean chips only, so passing fill chips have
    # no histogram. They are usable; they just need histogramming before any
    # statistic is computed over the full passing set.
    no_histogram = int((passes & (manifest["s1_status"] == "fill")).sum())
    if no_histogram:
        print()
        print("    NOTE %d passing chips have no S1 histogram yet (s1_status=fill;"
              % no_histogram)
        print("         build_histograms.py covered 'clean' only).")

    if not args.include_fill and int(passes.sum()) != EXPECTED_PASSING:
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
