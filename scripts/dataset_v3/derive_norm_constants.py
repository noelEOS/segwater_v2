#!/usr/bin/env python3
"""Normalization constants for the v3 memmaps, from the v3 TRAIN split.

Mean and population std of VV and VH in dB, computed over exactly the chips
that will populate ``train.memmap`` -- nothing more, nothing less. That
sentence is the whole reason this stage exists as its own script: the v2
constants were derived with a filter on the split column only, never on
``selected_for_memmap``, so they described 1,154,117 chips while the memmap
held 1,032,004 (a 10.6% excess; ~0.25-0.31 dB in the means). Harmless there --
a uniform affine offset -- but a defect all the same, and one assertion would
have caught it. Here that assertion is the centerpiece: the number of chips
entering the moments must equal the train split's size exactly.

**Recipe** (the established one, so the constants stay the same kind of
quantity): per-chip histogram -> moments over bin centers, merged Welford-style
(``WELFORD_MEAN_STD_MEMMAP_V2.py``). Because Welford merging is associative
and the bins are shared, merging chip-by-chip equals computing moments on the
summed histogram; the sum is what is computed, vectorized.

**Inputs.** ``chip_s1_histograms.parquet`` -- per-chip 320-bin exact bincount
histograms over the re-quantized dB codes, bin edges carried once in the
parquet schema metadata (never hard-coded here). The table covers all
``s1_status == 'clean'`` chips (1,355,702), a SUPERSET of the corpus: filter
the manifest to train first and join, never join-then-filter.

Two v2 caveats that do NOT apply to this lineage, stated so their absence
reads as a decision: (a) coverage is 1.0 by construction -- the histogram
build asserted ``sum(counts) + n_nodata == 50,176`` per chip and passing chips
have zero nodata -- so nothing is silently dropped at the range edges (v2 lost
0.21% of VV and 6.6% of VH); (b) pixels above +25 dB were folded into the top
bin at build time and are reported here via ``n_above_*``, not discarded.
No self-test against the v2 shipped constants is attempted: that baseline
lived in the SLIM_PARQUET histograms with different bins and different pixels
and is not reproducible from v3 inputs.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

import paths

CHIP_PIXELS = 224 * 224

# The previous lineage's constants (selection-clean by construction), for the
# reported delta only -- never adopted here.
MIXED80_CONSTANTS = {"vv": (-15.747683, 6.706104), "vh": (-23.800326, 7.709898)}

# Catastrophe bounds: loose enough for any plausible corpus, tight enough to
# catch a units or encoding error before it ships a plausible-looking number.
SANE = {"vv": {"mean": (-25.0, -5.0)}, "vh": {"mean": (-35.0, -12.0)},
        "std": (3.0, 12.0)}


def moments_from_hist(total_counts: np.ndarray, centres: np.ndarray) -> tuple[int, float, float]:
    """(n, mean, population std) from one aggregate histogram."""
    n = int(total_counts.sum())
    share = total_counts.astype(np.float64) / n
    mean = float((share * centres).sum())
    var = float((share * (centres - mean) ** 2).sum())
    return n, mean, var ** 0.5


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()

    manifest = pd.read_parquet(
        paths.require(paths.MANIFESTS / "dataset_v3_manifest.parquet", "v3 manifest"),
        columns=["pair_name", "chip_id", "split_v3"])
    if "split_v3" not in manifest.columns or manifest["split_v3"].isna().all():
        raise AssertionError("split_v3 missing or empty; run apply_splits.py first")
    train = manifest[manifest["split_v3"] == "train"][["pair_name", "chip_id"]]
    n_train = len(train)
    print("train split: %s chips" % "{:,}".format(n_train))

    hist_path = paths.require(paths.MANIFESTS / "chip_s1_histograms.parquet",
                              "S1 histograms; run build_histograms.py first")
    table = pq.read_table(hist_path)
    meta = json.loads(table.schema.metadata[b"segwater.histogram"])
    if meta["n_bins"] != 320 or meta["bin_width_db"] != 0.25 or meta["codes_per_bin"] != 5:
        raise AssertionError("histogram grid changed: %r -- re-derive this stage "
                             "against the new metadata deliberately" % meta)
    edges = np.asarray(meta["edges_db"], dtype=np.float64)
    centres = (edges[:-1] + edges[1:]) / 2.0

    hist = table.to_pandas()
    # Filter-then-join: the histogram table is a superset (all clean chips).
    joined = train.merge(hist, on=["pair_name", "chip_id"], validate="1:1", how="left")
    if joined["vv_counts"].isna().any():
        raise AssertionError("%d train chips have no S1 histogram"
                             % int(joined["vv_counts"].isna().sum()))
    # THE v2-lesson assertion: the population entering the moments is exactly
    # the train split, chip for chip.
    if len(joined) != n_train:
        raise AssertionError("moment population %d != train split %d"
                             % (len(joined), n_train))

    result = {}
    print()
    for band in ("vv", "vh"):
        counts = np.vstack(joined["%s_counts" % band].to_numpy()).astype(np.int64)
        total = counts.sum(axis=0)
        n, mean, std = moments_from_hist(total, centres)
        if n != n_train * CHIP_PIXELS:
            raise AssertionError("%s: %d pixels entered the moments, expected %d "
                                 "(coverage must be exactly 1.0 in this lineage)"
                                 % (band.upper(), n, n_train * CHIP_PIXELS))
        lo, hi = SANE[band]["mean"]
        if not (lo <= mean <= hi) or not (SANE["std"][0] <= std <= SANE["std"][1]):
            raise AssertionError("%s moments outside sanity bounds: mean %.3f std %.3f"
                                 % (band.upper(), mean, std))
        n_above = int(joined["n_above_%s" % band].sum())
        ref_mean, ref_std = MIXED80_CONSTANTS[band]
        print("  %s  n %s px   mean %.6f dB   std %.6f dB" % (band.upper(), "{:,}".format(n), mean, std))
        print("      above +25 dB folded into top bin: %s px (%.5f%%)"
              % ("{:,}".format(n_above), 100 * n_above / n))
        print("      vs mixed80 lineage (%.6f/%.6f): mean %+.4f dB (%.4f sigma), std %+.4f dB"
              % (ref_mean, ref_std, mean - ref_mean, (mean - ref_mean) / ref_std,
                 std - ref_std))
        result[band] = {"n": n, "mean": mean, "std": std, "n_above_top": n_above}

    if not args.write:
        print("\n(dry run; pass --write to save)")
        return 0

    payload = {
        "vv": {"mean": result["vv"]["mean"], "std": result["vv"]["std"], "n": result["vv"]["n"]},
        "vh": {"mean": result["vh"]["mean"], "std": result["vh"]["std"], "n": result["vh"]["n"]},
        "_meta": {
            "population": "dataset_v3 manifest, split_v3 == 'train' (exactly; asserted)",
            "n_chips": n_train,
            "recipe": "per-chip exact bincount histograms -> moments over 0.25 dB bin centres "
                      "(Welford-merge-equivalent); coverage 1.0; +25 dB fold reported in n_above_top",
            "source": str(hist_path),
            "script": "scripts/dataset_v3/derive_norm_constants.py",
            "encoding": meta["encoding"],
            "derived_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        },
    }
    paths.ensure_out()
    target = paths.MANIFESTS / "norm_constants_v3.json"
    target.write_text(json.dumps(payload, indent=2) + "\n")
    print("\nwrote %s" % target)
    return 0


if __name__ == "__main__":
    sys.exit(main())
