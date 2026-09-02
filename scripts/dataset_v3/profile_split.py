#!/usr/bin/env python3
"""Per-chip and per-pair composition profile of one v3 split.

Training on the v3 memmaps makes the **test split** the instrument every later
result is read off. Before a number is quoted from it, what the split *is* has
to be measurable: its composition, its coverage gaps, and a per-chip covariate
table that a prediction can be joined against (error vs water share, stratum,
tide, radiometry, origin). This stage builds that table. It measures; it does
not decide and it does not report -- ``report_profile.py`` reads what this
writes and turns it into the markdown.

**Population.** Exactly the chips in ``dataset_v3_memmap_manifest.parquet``
filtered to one ``split_v3``, which is exactly the chips in that split's
memmap. Not the full manifest: a row in this profile always addresses a row
that exists in the array, so a consumer can carry ``row`` straight into an
inference loop. The pinned ``EXPECTED_SPLITS`` guards both halves of that --
chips *and* pairs -- because a chip count alone would not have caught the
error this table exists to prevent (a split's pair count was quoted wrong for
three splits at once and survived, since nothing asserted it).

**Joins are all filter-then-join.** Two of the three side tables are supersets
of the memmap manifest: the S1 histograms cover every ``s1_status == 'clean'``
chip (1,355,702) and the eval distances cover the whole passing corpus
(1,328,811). Filtering to the split first and joining ``validate="1:1"`` means
a superset can never silently fan out; joining first and filtering after would
give the same rows today and a different answer the day a side table grows.
Every join re-asserts the row count and the id-space split, because
``chip_id`` alone is not unique -- it repeats across pairs, and recovered chips
live at ``chip_id >= 100_000`` by construction.

**Radiometry comes from the exact per-chip histograms**, not from a re-read of
the arrays: 320 bins of 0.25 dB from -55 to +25 dB, with the edges carried in
the parquet's ``segwater.histogram`` schema metadata and read from there, never
hard-coded. Per-chip mean and population std are moments over bin centres (the
same recipe as ``derive_norm_constants.py``, so the numbers are the same kind
of quantity as the shipped normalization constants). Percentiles are the
**left-continuous binned inverse CDF**: the lowest bin centre whose cumulative
count reaches q*n. On 0.25 dB bins that is the resolution of any percentile
here, and it is why p5/p50/p95 land on quarter-dB values.

``floor_share`` is the share of a chip's pixels in the lowest bin -- the
quantization floor the v3 encoding puts at -55 dB -- and ``n_above`` is the
count folded into the top bin at +25 dB at build time. Both are reported, not
discarded, so a domain-shift check can see the clipping instead of inferring it.

**Tide delta is computed live** as ``|s1_tide_level_sat - s2_tide_level_sat|``.
The manifest's ``s1_s2_tide_level_sat_delta_first_round`` column belongs to the
first-round lineage and is not the delta the current gate was applied on; the
census export computes the live difference for the same reason.

Two columns are deliberately absent. No ``water_fraction`` over all 50,176
pixels: "water fraction" is ambiguous between that and the share of *valid*
pixels, and ``count_label_classes.py`` already refused to store a fraction for
this reason. What is stored is ``water_share = n_water / (n_water + n_land)``
under that name, plus the counts, so the denominator a consumer used is visible
in their code. And no class column at a second threshold: ``ws_class`` is at
t = 1% and the threshold is in the column's docstring and the report, because
the distribution is U-shaped and the mixed share swings 47.8 / 25.2 / 17.8% as
t moves 0 -> 1% -> 5%. A second column would invite quoting one without saying
which.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

import paths

# Chips and pairs per split, both pinned. Verified against the memmap manifest
# and `qc/splits_report_v3.md`, which agree.
EXPECTED_SPLITS = {
    "train": {"chips": 925_123, "pairs": 2_121},
    "val": {"chips": 265_952, "pairs": 575},
    "test": {"chips": 131_713, "pairs": 316},
}
EXPECTED_ROWS = 1_322_788

MIXED_THRESHOLD = 0.01
CHIP_PIXELS = 224 * 224

# Disjoint id spaces: recovered chips were given ids above 100,000 so a join
# back to a DATABASE/ parquet cannot collide with the original numbering.
RECOVERED_ID_FLOOR = 100_000
EXISTING_ID_CEIL = 4_634

# Histogram grid, asserted against the parquet metadata rather than trusted.
EXPECTED_BINS = 320
EXPECTED_BIN_WIDTH_DB = 0.25

PAIR_SUMMARY_CSV = (paths.DB_SLIM
                    / "pair_summary_with_splits-ready_for_memmap_v2_creation.csv")

PROFILE_COLUMNS = [
    "pair_name", "chip_id", "split_v3", "row", "chip_origin",
    "bbox_w", "bbox_s", "bbox_e", "bbox_n",
    "v3_label_variant", "phase3_reviewed", "phase3_action",
    "intersects_gcl_coastline",
    "n_water", "n_land", "n_invalid", "invalid_frac",
    "s1_tide_level_sat", "s2_tide_level_sat",
    "s1_tide_level_sat_phase", "s2_tide_level_sat_phase", "tide_source",
    "system:time_start_s1",
]


def check_id_spaces(frame: pd.DataFrame, where: str) -> None:
    """Recovered ids sit above the floor, existing ids below the ceiling."""
    recovered = frame["chip_origin"] == "recovered"
    bad_recovered = int((recovered & (frame["chip_id"] < RECOVERED_ID_FLOOR)).sum())
    bad_existing = int((~recovered & (frame["chip_id"] > EXISTING_ID_CEIL)).sum())
    if bad_recovered or bad_existing:
        raise AssertionError(
            "%s: id spaces overlap -- %d recovered chips below %d, %d existing "
            "above %d" % (where, bad_recovered, RECOVERED_ID_FLOOR,
                          bad_existing, EXISTING_ID_CEIL))


def join_1to1(left: pd.DataFrame, right: pd.DataFrame, what: str,
              required: str) -> pd.DataFrame:
    """Left-join a superset side table, asserting shape and completeness."""
    before = len(left)
    joined = left.merge(right, on=["pair_name", "chip_id"], how="left",
                        validate="1:1")
    if len(joined) != before:
        raise AssertionError("%s join changed the row count %d -> %d"
                             % (what, before, len(joined)))
    missing = int(joined[required].isna().sum())
    if missing:
        raise AssertionError("%d chips have no %s" % (missing, what))
    check_id_spaces(joined, "after %s join" % what)
    return joined


def histogram_stats(counts: np.ndarray, centres: np.ndarray) -> dict:
    """Per-chip moments and binned percentiles from one band's histograms.

    ``counts`` is (n_chips, n_bins). Percentiles are the left-continuous binned
    inverse CDF: the lowest bin centre whose cumulative count reaches q*n.
    """
    counts = counts.astype(np.float64)
    n = counts.sum(axis=1)
    if not np.all(n == CHIP_PIXELS):
        raise AssertionError("histogram counts do not sum to %d on %d chips"
                             % (CHIP_PIXELS, int((n != CHIP_PIXELS).sum())))
    share = counts / n[:, None]
    mean = share @ centres
    var = share @ (centres ** 2) - mean ** 2
    std = np.sqrt(np.maximum(var, 0.0))

    cum = np.cumsum(counts, axis=1)
    out = {"mean_db": mean, "std_db": std,
           "floor_share": counts[:, 0] / n}
    for q in (0.05, 0.50, 0.95):
        idx = np.argmax(cum >= q * n[:, None], axis=1)
        out["p%d_db" % round(100 * q)] = centres[idx]
    return out


def band_stats(hist_path, keys: pd.DataFrame, band: str,
               centres: np.ndarray, block: int) -> pd.DataFrame:
    """Read ONE band of the histogram table and reduce it to per-chip stats.

    One band at a time on purpose: the counts materialize to 1.6 GiB per band
    across the whole table, and there is no reason to hold both at once.
    """
    table = pq.read_table(hist_path, columns=["pair_name", "chip_id",
                                              "%s_counts" % band,
                                              "n_above_%s" % band])
    frame = table.to_pandas()
    del table
    # Filter-then-reduce: the histogram table is a superset of the split.
    frame = keys.merge(frame, on=["pair_name", "chip_id"], how="left",
                       validate="1:1")
    if frame["%s_counts" % band].isna().any():
        raise AssertionError("%d chips have no %s histogram"
                             % (int(frame["%s_counts" % band].isna().sum()),
                                band.upper()))

    pieces = []
    counts_column = frame["%s_counts" % band].to_numpy()
    for start in range(0, len(frame), block):
        chunk = np.vstack(counts_column[start:start + block])
        pieces.append(pd.DataFrame(histogram_stats(chunk, centres)))
    stats = pd.concat(pieces, ignore_index=True)
    stats.columns = ["%s_%s" % (band, c) for c in stats.columns]
    stats["n_above_%s" % band] = frame["n_above_%s" % band].to_numpy()
    return stats


def build_pair_profile(chips: pd.DataFrame) -> pd.DataFrame:
    """Per-pair aggregates. One row per pair; pairs never span splits."""
    grouped = chips.groupby("pair_name", sort=True)
    pairs = grouped.agg(
        n_chips=("chip_id", "size"),
        n_mixed=("is_mixed", "sum"),
        n_recovered=("is_recovered", "sum"),
        ws_median=("water_share", "median"),
        ws_p10=("water_share", lambda s: s.quantile(0.10)),
        ws_p90=("water_share", lambda s: s.quantile(0.90)),
        invalid_frac_median=("invalid_frac", "median"),
        lat=("lat", "mean"),
        lon=("lon", "mean"),
        year=("year", "median"),
        month=("month", "median"),
        s1_tide_median=("s1_tide_level_sat", "median"),
        tide_delta_median=("tide_delta_abs", "median"),
        vv_median_db=("vv_p50_db", "median"),
        vh_median_db=("vh_p50_db", "median"),
        eval_dist_km=("eval_dist_km", "median"),
    )
    pairs["mixed_share"] = pairs["n_mixed"] / pairs["n_chips"]
    pairs["n_existing"] = pairs["n_chips"] - pairs["n_recovered"]
    for column in ("split_v3", "stratum_coarse", "gcl_class", "climate",
                   "nearest_eval_site", "v3_label_variant"):
        pairs[column] = grouped[column].agg(
            lambda s: s.mode().iat[0] if not s.mode().empty else None)
    pairs["year"] = pairs["year"].round().astype(int)
    pairs["month"] = pairs["month"].round().astype(int)
    return pairs.reset_index()


def summarise(chips: pd.DataFrame, pairs: pd.DataFrame, split: str) -> None:
    print()
    print("PROFILE  %s  %s chips / %d pairs"
          % (split, "{:,}".format(len(chips)), len(pairs)))
    print()
    classes = chips["ws_class"].value_counts(normalize=True)
    print("  water-share class at t = %g%%:  %s"
          % (100 * MIXED_THRESHOLD,
             "  ".join("%s %.2f%%" % (k, 100 * v)
                       for k, v in classes.sort_index().items())))
    valid = chips["n_water"] + chips["n_land"]
    print("  pixel-level water prior: %.4f  |  water share median %.4f"
          % (chips["n_water"].sum() / valid.sum(), chips["water_share"].median()))
    print("  invalid_frac: mean %.4f  p95 %.4f  max %.4f"
          % (chips["invalid_frac"].mean(), chips["invalid_frac"].quantile(0.95),
             chips["invalid_frac"].max()))
    print("  origin: %s  |  label variant: %s"
          % (chips["chip_origin"].value_counts().to_dict(),
             chips["v3_label_variant"].value_counts().to_dict()))
    print("  strata occupied: %d of 25  |  years %d-%d"
          % (chips["stratum_coarse"].nunique(), chips["year"].min(),
             chips["year"].max()))
    for band in ("vv", "vh"):
        print("  %s dB: mean %.3f  p5 %.3f  p50 %.3f  p95 %.3f  floor share %.5f"
              % (band.upper(), chips["%s_mean_db" % band].mean(),
                 chips["%s_p5_db" % band].median(),
                 chips["%s_p50_db" % band].median(),
                 chips["%s_p95_db" % band].median(),
                 chips["%s_floor_share" % band].mean()))
    print("  eval distance km: min %.3f  median %.1f"
          % (chips["eval_dist_km"].min(), chips["eval_dist_km"].median()))
    fragile = pairs[pairs["n_mixed"] < 5]
    print("  pairs with <5 mixed chips: %d of %d" % (len(fragile), len(pairs)))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="test",
                       choices=sorted(EXPECTED_SPLITS), help="split to profile")
    parser.add_argument("--block", type=int, default=200_000,
                        help="chips per histogram reduction block")
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    split = args.split
    pinned = EXPECTED_SPLITS[split]

    manifest = pd.read_parquet(
        paths.require(paths.MANIFESTS / "dataset_v3_memmap_manifest.parquet",
                      "v3 memmap manifest"),
        columns=PROFILE_COLUMNS)
    if len(manifest) != EXPECTED_ROWS:
        raise AssertionError("memmap manifest is %d rows, expected %d"
                             % (len(manifest), EXPECTED_ROWS))

    chips = manifest[manifest["split_v3"] == split].reset_index(drop=True)
    if len(chips) != pinned["chips"]:
        raise AssertionError("%s has %d chips, expected %d"
                             % (split, len(chips), pinned["chips"]))
    if chips["pair_name"].nunique() != pinned["pairs"]:
        raise AssertionError("%s has %d pairs, expected %d"
                             % (split, chips["pair_name"].nunique(), pinned["pairs"]))
    check_id_spaces(chips, "split filter")

    # `row` addresses the split's memmap; it must be a full permutation.
    rows = chips["row"].to_numpy()
    if rows.min() != 0 or rows.max() != len(chips) - 1 or len(np.unique(rows)) != len(chips):
        raise AssertionError("%s rows are not a 0..n-1 permutation" % split)
    print("%s: %s chips / %d pairs, rows 0..%s"
          % (split, "{:,}".format(len(chips)), chips["pair_name"].nunique(),
             "{:,}".format(len(chips) - 1)))

    # --- composition ---------------------------------------------------------
    total = chips["n_water"] + chips["n_land"] + chips["n_invalid"]
    if not (total == CHIP_PIXELS).all():
        raise AssertionError("label counts do not sum to %d on every chip"
                             % CHIP_PIXELS)
    valid = chips["n_water"] + chips["n_land"]
    if (valid <= 0).any():
        raise AssertionError("%d chips have no valid pixels" % int((valid <= 0).sum()))
    chips["water_share"] = chips["n_water"] / valid
    chips["ws_class"] = np.select(
        [chips["water_share"] <= MIXED_THRESHOLD,
         chips["water_share"] >= 1 - MIXED_THRESHOLD],
        ["pure_land", "pure_water"], "mixed")
    chips["is_mixed"] = chips["ws_class"] == "mixed"
    chips["is_recovered"] = chips["chip_origin"] == "recovered"
    # invalid_frac is an independent measurement of the same quantity.
    delta = (chips["invalid_frac"] * CHIP_PIXELS - chips["n_invalid"]).abs()
    if (delta > 1).any():
        raise AssertionError("invalid_frac disagrees with n_invalid on %d chips"
                             % int((delta > 1).sum()))

    # --- geometry, time, tide ------------------------------------------------
    chips["lat"] = (chips["bbox_s"] + chips["bbox_n"]) / 2
    chips["lon"] = (chips["bbox_w"] + chips["bbox_e"]) / 2
    chips["latband"] = (chips["lat"] // 10 * 10).astype(int)
    # 20 pairs carry fractional seconds; a single-format parse silently NaTs
    # 11,588 chips. format="mixed" is not optional here.
    stamp = pd.to_datetime(chips["system:time_start_s1"], format="mixed", utc=True)
    if stamp.isna().any():
        raise AssertionError("%d chips have an unparseable S1 timestamp"
                             % int(stamp.isna().sum()))
    chips["year"] = stamp.dt.year
    chips["month"] = stamp.dt.month
    chips = chips.drop(columns=["system:time_start_s1"])
    chips["tide_delta_abs"] = (chips["s1_tide_level_sat"]
                               - chips["s2_tide_level_sat"]).abs()
    for column in ("s1_tide_level_sat", "s2_tide_level_sat", "tide_delta_abs"):
        if chips[column].isna().any():
            raise AssertionError("%d chips have no %s"
                                 % (int(chips[column].isna().sum()), column))

    # --- stratum -------------------------------------------------------------
    strata = pd.read_csv(paths.require(PAIR_SUMMARY_CSV, "pair summary"),
                         usecols=["pair_name", "stratum_coarse"])
    before = len(chips)
    chips = chips.merge(strata, on="pair_name", how="left", validate="m:1")
    if len(chips) != before:
        raise AssertionError("stratum join changed the row count")
    if chips["stratum_coarse"].isna().any():
        raise AssertionError("%d chips have no stratum"
                             % int(chips["stratum_coarse"].isna().sum()))
    chips["gcl_class"] = chips["stratum_coarse"].str.split("_").str[0]
    chips["climate"] = chips["stratum_coarse"].str.split("_").str[1]

    # --- eval distance -------------------------------------------------------
    eval_dist = pd.read_parquet(
        paths.require(paths.MANIFESTS / "chip_eval_distance.parquet",
                      "chip eval distance"))
    chips = join_1to1(chips, eval_dist, "eval distance", "eval_dist_km")

    # --- radiometry ----------------------------------------------------------
    hist_path = paths.require(paths.MANIFESTS / "chip_s1_histograms.parquet",
                              "S1 histograms")
    meta = json.loads(pq.ParquetFile(hist_path).schema_arrow
                      .metadata[b"segwater.histogram"])
    if meta["n_bins"] != EXPECTED_BINS or meta["bin_width_db"] != EXPECTED_BIN_WIDTH_DB:
        raise AssertionError("histogram grid changed: %r -- re-derive this stage "
                             "against the new metadata deliberately" % meta)
    edges = np.asarray(meta["edges_db"], dtype=np.float64)
    centres = (edges[:-1] + edges[1:]) / 2.0

    keys = chips[["pair_name", "chip_id"]]
    for band in ("vv", "vh"):
        stats = band_stats(hist_path, keys, band, centres, args.block)
        if len(stats) != len(chips):
            raise AssertionError("%s stats are %d rows, expected %d"
                                 % (band.upper(), len(stats), len(chips)))
        chips = pd.concat([chips, stats], axis=1)
        print("  %s radiometry reduced" % band.upper())

    if len(chips) != pinned["chips"]:
        raise AssertionError("chip profile drifted to %d rows" % len(chips))
    check_id_spaces(chips, "final")

    pairs = build_pair_profile(chips)
    if len(pairs) != pinned["pairs"]:
        raise AssertionError("pair profile is %d rows, expected %d"
                             % (len(pairs), pinned["pairs"]))
    if int(pairs["n_chips"].sum()) != pinned["chips"]:
        raise AssertionError("pair profile chip mass %d != %d"
                             % (int(pairs["n_chips"].sum()), pinned["chips"]))

    summarise(chips, pairs, split)

    if not args.write:
        print("\n(dry run; pass --write to save)")
        return 0

    paths.ensure_out()
    stamp_now = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    provenance = json.dumps({
        "split": split,
        "population": "dataset_v3_memmap_manifest.parquet, split_v3 == '%s' "
                      "(exactly; chips and pairs asserted)" % split,
        "n_chips": len(chips), "n_pairs": len(pairs),
        "mixed_threshold": MIXED_THRESHOLD,
        "water_share": "n_water / (n_water + n_land)",
        "percentiles": "left-continuous binned inverse CDF over 0.25 dB bins",
        "tide_delta": "|s1_tide_level_sat - s2_tide_level_sat|, computed live",
        "script": "scripts/dataset_v3/profile_split.py",
        "built": stamp_now,
    })
    for frame, name in ((chips, "chip"), (pairs, "pair")):
        table = pa.Table.from_pandas(frame, preserve_index=False)
        table = table.replace_schema_metadata({"segwater.profile": provenance})
        target = paths.MANIFESTS / ("%s_%s_profile.parquet" % (split, name))
        pq.write_table(table, target, compression="zstd")
        print("\nwrote %s  (%s rows, %.1f MiB)"
              % (target, "{:,}".format(len(frame)),
                 target.stat().st_size / 2**20))
    return 0


if __name__ == "__main__":
    sys.exit(main())
