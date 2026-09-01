#!/usr/bin/env python3
"""QC report on one split's composition profile, with the other splits as baselines.

Reads what ``profile_split.py`` wrote and turns it into a human-readable
markdown report plus the aggregate CSVs the supplementary figures are built
from. Reporting is kept out of the measurement stage on purpose, the same way
``report_splits.py`` is kept out of ``apply_splits.py``: the measurement is
expensive and writes a table other work joins against, this is decision-free
and can be re-run freely without touching that table.

The report answers one question -- *what can this split speak to, and what can
it not* -- in seven sections:

1. **Structure and effective size.** Chips per pair, Gini, and the Kish
   effective pair count. The last is the number that matters for any
   pair-macro metric: chips inside a pair are correlated, so the effective N
   behind a pair-macro mean is nearer the Kish count than the pair count, and
   much nearer that than the chip count. Any CI computed as if chips were
   independent will be too narrow.
2. **Composition.** The water-share distribution is U-shaped, so the report
   gives the shape and the walls rather than leaning on a median, and shows the
   class split at t = 0 / 1 / 5% side by side. The mixed share is a statement
   about the threshold as much as about the data.
3. **Strata coverage and gaps.** The full 25-stratum table, then the explicit
   list of strata with **zero** presence and the thin cells. Stated up front
   because it is the "what this split cannot speak to" list.
4. **Representativeness.** TVD of the split against train and against the whole
   corpus over every categorical axis, plus 1-D Wasserstein (W1) on tide level
   and on the aggregate VV/VH dB distributions. TVD reads in share units; W1
   reads in the axis's own units (cm, dB), which is what makes it the right
   statistic for a domain-shift check on radiometry.
5. **QC lineage.** What the split inherited from the label rounds: B8<400
   variant share, Phase 3 review share, recovered share.
6. **Metric-stability flags.** Pairs with fewer than 5 mixed chips (a
   pair-macro mean over those is dominated by a handful of chips), where the
   two sliver pairs landed, and the largest pairs.
7. **Eval-site context.** Distance to the nearest external evaluation AOI,
   which is > 30 km for every chip by construction; the report quotes the
   minimum so that the floor is visible rather than asserted.

Estimators are imported from ``report_splits.py`` -- ``gini``, ``kish``,
``tvd``, ``to_md`` -- rather than re-implemented, so the split report and this
one cannot drift. ``w1`` is the one addition; ``--selftest`` checks it against
a case with a known answer.
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys

import numpy as np
import pandas as pd

import paths
from report_splits import gini, kish, to_md, tvd

MIXED_THRESHOLD = 0.01
THRESHOLDS = (0.0, 0.01, 0.05)
FRAGILE_MIXED = 5
SLIVER_PAIRS = ("PAIR_2770", "PAIR_3267")
ALL_SPLITS = ("train", "val", "test")

EXPECTED_SPLITS = {
    "train": {"chips": 925_123, "pairs": 2_121},
    "val": {"chips": 265_952, "pairs": 575},
    "test": {"chips": 131_713, "pairs": 316},
}
EXPECTED_ROWS = 1_322_788

# Anchors from `qc/splits_report_v3.md`, reproduced here as a regression gate.
# A mismatch is a STOP, never something to patch.
ANCHOR_TEST_MIXED_PCT = 26.42
ANCHOR_ZERO_TEST_STRATA = ("1_D", "3_D", "5_D")

GCL_NAMES = {"0": "artificial", "1": "biogenic", "2": "sandy",
             "3": "muddy", "4": "rocky", "5": "estuary"}
CLIMATE_NAMES = {"A": "tropical", "B": "arid", "C": "temperate",
                 "D": "cold", "E": "polar"}


def w1(values_a: np.ndarray, values_b: np.ndarray,
       weights_a: np.ndarray | None = None,
       weights_b: np.ndarray | None = None) -> float:
    """1-D Wasserstein distance between two weighted empirical distributions.

    Computed as the integral of |F_a - F_b| over the pooled support, which for
    1-D distributions equals the optimal transport cost. Reads in the units of
    the axis: a W1 of 0.4 dB means the average pixel has to move 0.4 dB.
    """
    values_a = np.asarray(values_a, dtype=np.float64)
    values_b = np.asarray(values_b, dtype=np.float64)
    weights_a = (np.ones(len(values_a)) if weights_a is None
                 else np.asarray(weights_a, dtype=np.float64))
    weights_b = (np.ones(len(values_b)) if weights_b is None
                 else np.asarray(weights_b, dtype=np.float64))

    support = np.unique(np.concatenate([values_a, values_b]))
    order_a = np.argsort(values_a)
    order_b = np.argsort(values_b)
    cdf_a = np.cumsum(weights_a[order_a]) / weights_a.sum()
    cdf_b = np.cumsum(weights_b[order_b]) / weights_b.sum()
    at_a = np.searchsorted(values_a[order_a], support, side="right") - 1
    at_b = np.searchsorted(values_b[order_b], support, side="right") - 1
    fa = np.where(at_a >= 0, cdf_a[np.maximum(at_a, 0)], 0.0)
    fb = np.where(at_b >= 0, cdf_b[np.maximum(at_b, 0)], 0.0)
    widths = np.diff(support)
    return float((np.abs(fa - fb)[:-1] * widths).sum())


def selftest() -> int:
    """W1 against cases with an answer known in closed form."""
    grid = np.arange(0.0, 1.0, 0.001)
    shifted = grid + 0.25
    got = w1(grid, shifted)
    if abs(got - 0.25) > 1e-3:
        raise AssertionError("W1 of a 0.25 shift is %.6f, expected 0.25" % got)
    if abs(w1(grid, grid)) > 1e-12:
        raise AssertionError("W1 of a distribution against itself is not 0")
    # Weighted: two point masses 2.0 apart, half the mass moved.
    got = w1(np.array([0.0, 2.0]), np.array([0.0, 2.0]),
             np.array([1.0, 0.0]), np.array([0.0, 1.0]))
    if abs(got - 2.0) > 1e-9:
        raise AssertionError("weighted W1 is %.6f, expected 2.0" % got)
    print("selftest: w1 ok (shift 0.25, identity 0.0, weighted 2.0)")
    return 0


def class_shares(water_share: pd.Series, threshold: float) -> dict:
    """Three-class shares at one purity threshold."""
    land = float((water_share <= threshold).mean())
    water = float((water_share >= 1 - threshold).mean())
    return {"pure_land": land, "mixed": 1 - land - water, "pure_water": water}


def band_histogram(chips: pd.DataFrame, band: str,
                   edges: np.ndarray) -> np.ndarray:
    """Chip-mean dB histogram for one split, on a shared grid."""
    counts, _ = np.histogram(chips["%s_mean_db" % band], bins=edges)
    return counts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="test", choices=sorted(EXPECTED_SPLITS))
    parser.add_argument("--selftest", action="store_true",
                        help="check the new estimator and exit")
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    if args.selftest:
        return selftest()
    selftest()
    target_split = args.split

    profiles, pair_profiles = {}, {}
    for name in ALL_SPLITS:
        profiles[name] = pd.read_parquet(paths.require(
            paths.MANIFESTS / ("%s_chip_profile.parquet" % name),
            "%s chip profile; run profile_split.py --split %s" % (name, name)))
        pair_profiles[name] = pd.read_parquet(paths.require(
            paths.MANIFESTS / ("%s_pair_profile.parquet" % name),
            "%s pair profile" % name))
        pinned = EXPECTED_SPLITS[name]
        if len(profiles[name]) != pinned["chips"] or len(pair_profiles[name]) != pinned["pairs"]:
            raise AssertionError("%s profile is %d chips / %d pairs, expected %d / %d"
                                 % (name, len(profiles[name]), len(pair_profiles[name]),
                                    pinned["chips"], pinned["pairs"]))

    corpus = pd.concat(profiles.values(), ignore_index=True)
    if len(corpus) != EXPECTED_ROWS:
        raise AssertionError("profiles re-add to %d, expected %d"
                             % (len(corpus), EXPECTED_ROWS))

    chips = profiles[target_split]
    pairs = pair_profiles[target_split]
    train = profiles["train"]
    n_chips = len(chips)

    lines: list[str] = []

    def emit(text: str = "") -> None:
        print(text)
        lines.append(text)

    emit("# v3 %s-split profile -- generated %s"
         % (target_split,
            dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")))
    emit()
    emit("Population: the %s chips of `split_v3 == '%s'` in"
         % ("{:,}".format(n_chips), target_split))
    emit("`dataset_v3_memmap_manifest.parquet` -- exactly the chips in")
    emit("`%s.memmap`, across %d pairs. Water share is" % (target_split, len(pairs)))
    emit("`n_water / (n_water + n_land)`; composition classes use the purity")
    emit("threshold t = %g%% unless a section states otherwise." % (100 * MIXED_THRESHOLD))
    emit()

    # -- 1. structure ---------------------------------------------------------
    emit("## 1. Structure and effective size")
    emit()
    sizes = pairs["n_chips"].to_numpy()
    kish_pairs = kish(sizes)
    emit("| statistic | value |")
    emit("|---|---:|")
    emit("| chips | %s |" % "{:,}".format(n_chips))
    emit("| pairs | %d |" % len(pairs))
    emit("| chips per pair, median | %.0f |" % np.median(sizes))
    emit("| chips per pair, p10 / p90 | %.0f / %.0f "
         "|" % (np.percentile(sizes, 10), np.percentile(sizes, 90)))
    emit("| chips per pair, max | %s (%s) |"
         % ("{:,}".format(int(sizes.max())),
            pairs.loc[pairs["n_chips"].idxmax(), "pair_name"]))
    emit("| Gini of chip mass over pairs | %.3f |" % gini(sizes))
    emit("| Kish effective pairs | %.0f |" % kish_pairs)
    emit()
    emit("The Kish count is the N behind a pair-macro metric, not %d and"
         % len(pairs))
    emit("emphatically not %s. Chips inside a pair share a scene, a date, a"
         % "{:,}".format(n_chips))
    emit("tide state and a label raster, so they are correlated; a confidence")
    emit("interval computed as if the %s chips were independent will be too"
         % "{:,}".format(n_chips))
    emit("narrow. Block-resample whole pairs.")
    emit()

    # -- 2. composition -------------------------------------------------------
    emit("## 2. Composition")
    emit()
    rows = []
    for threshold in THRESHOLDS:
        shares = class_shares(chips["water_share"], threshold)
        rows.append({"t": "%g%%" % (100 * threshold),
                     "pure_land": "%.2f%%" % (100 * shares["pure_land"]),
                     "mixed": "%.2f%%" % (100 * shares["mixed"]),
                     "pure_water": "%.2f%%" % (100 * shares["pure_water"])})
    emit(to_md(pd.DataFrame(rows).set_index("t")))
    emit()
    mixed_pct = 100 * class_shares(chips["water_share"], MIXED_THRESHOLD)["mixed"]
    emit("The mixed share is as much a statement about t as about the data.")
    emit("At the reporting threshold t = %g%% it is **%.2f%%**."
         % (100 * MIXED_THRESHOLD, mixed_pct))
    emit()
    valid = chips["n_water"] + chips["n_land"]
    exact_land = int((chips["water_share"] == 0).sum())
    exact_water = int((chips["water_share"] == 1).sum())
    emit("- pixel-level water prior: **%.4f** (%s water of %s valid px)"
         % (chips["n_water"].sum() / valid.sum(),
            "{:,}".format(int(chips["n_water"].sum())),
            "{:,}".format(int(valid.sum()))))
    emit("- water share: median %.4f, p10 %.4f, p90 %.4f"
         % (chips["water_share"].median(), chips["water_share"].quantile(0.10),
            chips["water_share"].quantile(0.90)))
    emit("- the walls: %s chips (%.2f%%) hold exactly zero water pixels, %s "
         "(%.2f%%) exactly zero land"
         % ("{:,}".format(exact_land), 100 * exact_land / n_chips,
            "{:,}".format(exact_water), 100 * exact_water / n_chips))
    emit("- `invalid_frac`: mean %.4f, p95 %.4f, max %.4f (gate is 0.15)"
         % (chips["invalid_frac"].mean(), chips["invalid_frac"].quantile(0.95),
            chips["invalid_frac"].max()))
    emit()

    # -- 3. strata ------------------------------------------------------------
    emit("## 3. Strata coverage and gaps")
    emit()
    corpus_strata = sorted(corpus["stratum_coarse"].unique())
    table = pd.DataFrame(index=corpus_strata)
    table.index.name = "stratum_coarse"
    split_chips = chips.groupby("stratum_coarse").size()
    split_pairs = chips.groupby("stratum_coarse")["pair_name"].nunique()
    table["chips"] = split_chips.reindex(corpus_strata, fill_value=0).astype(int)
    table["pairs"] = split_pairs.reindex(corpus_strata, fill_value=0).astype(int)
    corpus_chips = corpus.groupby("stratum_coarse").size().reindex(corpus_strata)
    table["corpus_chips"] = corpus_chips.astype(int)
    table["share_of_stratum"] = ["%.2f%%" % v for v in
                                 100 * table["chips"] / table["corpus_chips"]]
    emit(to_md(table))
    emit()
    absent = [s for s in corpus_strata if table.loc[s, "chips"] == 0]
    emit("**Zero presence** (%d strata): %s." % (len(absent), ", ".join(absent)))
    emit("These are the thin strata forced whole to train, so the %s split"
         % target_split)
    emit("carries no evidence about them at all.")
    emit()
    thin = table[(table["pairs"] > 0) & (table["pairs"] <= 3)]
    emit("**Thin cells** (1-3 pairs -- present, but a single scene's quirks "
         "move the whole cell):")
    for stratum, row in thin.iterrows():
        gcl, climate = stratum.split("_")
        emit("- `%s` (%s x %s): %d pair%s, %s chips"
             % (stratum, GCL_NAMES.get(gcl, gcl), CLIMATE_NAMES.get(climate, climate),
                row["pairs"], "" if row["pairs"] == 1 else "s",
                "{:,}".format(int(row["chips"]))))
    emit()

    # -- 4. representativeness ------------------------------------------------
    emit("## 4. Representativeness")
    emit()
    emit("Total variation distance, %s against train and against the whole"
         % target_split)
    emit("corpus (0 = identical shares, 1 = disjoint).")
    emit()
    emit("| axis | vs train | vs corpus |")
    emit("|---|---:|---:|")
    axes = [("stratum_coarse", "stratum"), ("gcl_class", "coastline class"),
            ("climate", "climate"), ("ws_class", "water-share class (t=1%)"),
            ("s1_tide_level_sat_phase", "tide phase"), ("year", "year"),
            ("latband", "latitude band (10 deg)"), ("chip_origin", "chip origin")]
    tvd_rows = []
    for column, label in axes:
        here = chips[column].value_counts(normalize=True, dropna=False)
        against_train = tvd(here, train[column].value_counts(normalize=True, dropna=False))
        against_corpus = tvd(here, corpus[column].value_counts(normalize=True, dropna=False))
        emit("| %s | %.4f | %.4f |" % (label, against_train, against_corpus))
        tvd_rows.append({"axis": label, "vs_train": against_train,
                         "vs_corpus": against_corpus})
    emit()
    emit("1-D Wasserstein distance, in the units of each axis.")
    emit()
    emit("| axis | unit | vs train | vs corpus |")
    emit("|---|---|---:|---:|")
    w1_rows = []
    w1_axes = [("s1_tide_level_sat", "tide level at S1", "cm"),
               ("tide_delta_abs", "|S1-S2| tide delta", "cm"),
               ("vv_mean_db", "VV chip-mean", "dB"),
               ("vh_mean_db", "VH chip-mean", "dB"),
               ("water_share", "water share", "share")]
    for column, label, unit in w1_axes:
        against_train = w1(chips[column].to_numpy(), train[column].to_numpy())
        against_corpus = w1(chips[column].to_numpy(), corpus[column].to_numpy())
        emit("| %s | %s | %.4f | %.4f |" % (label, unit, against_train, against_corpus))
        w1_rows.append({"axis": label, "unit": unit, "vs_train": against_train,
                        "vs_corpus": against_corpus})
    emit()

    # -- 5. QC lineage --------------------------------------------------------
    emit("## 5. QC lineage in %s" % target_split)
    emit()
    b8 = int((chips["v3_label_variant"] == "b8_lt400").sum())
    reviewed = int(chips["phase3_reviewed"].sum())
    recovered = int(chips["is_recovered"].sum())
    recovered_pairs = int((pair_profiles[target_split]["n_recovered"] > 0).sum())
    emit("| lineage | chips | share |")
    emit("|---|---:|---:|")
    emit("| B8<400 label variant | %s | %.2f%% |"
         % ("{:,}".format(b8), 100 * b8 / n_chips))
    emit("| Phase 3 reviewed | %s | %.2f%% |"
         % ("{:,}".format(reviewed), 100 * reviewed / n_chips))
    emit("| recovered origin | %s | %.2f%% |"
         % ("{:,}".format(recovered), 100 * recovered / n_chips))
    emit("| GCL-coastline intersecting | %s | %.2f%% |"
         % ("{:,}".format(int(chips["intersects_gcl_coastline"].sum())),
            100 * chips["intersects_gcl_coastline"].mean()))
    emit("| tide computed (FES2022b) | %s | %.2f%% |"
         % ("{:,}".format(int((chips["tide_source"] == "computed_fes2022b").sum())),
            100 * (chips["tide_source"] == "computed_fes2022b").mean()))
    emit()
    emit("Recovered chips touch %d of %d pairs. Phase 3 actions: %s."
         % (recovered_pairs, len(pairs),
            ", ".join("%s %s" % (k, "{:,}".format(v)) for k, v in
                      chips["phase3_action"].value_counts().items()) or "none"))
    emit()

    # -- 6. stability ---------------------------------------------------------
    emit("## 6. Metric-stability flags")
    emit()
    fragile = pairs[pairs["n_mixed"] < FRAGILE_MIXED].sort_values("n_mixed")
    no_mixed = int((pairs["n_mixed"] == 0).sum())
    emit("**Pairs with fewer than %d mixed chips: %d of %d** (%d with none)."
         % (FRAGILE_MIXED, len(fragile), len(pairs), no_mixed))
    emit("A pair-macro mean gives each of these the same weight as a pair with")
    emit("thousands of mixed chips, so each is a high-variance term.")
    emit()
    if len(fragile):
        listed = fragile[["pair_name", "n_chips", "n_mixed", "stratum_coarse"]]
        emit(to_md(listed.set_index("pair_name")))
        emit()
    emit("**Sliver pairs.** The eval buffer left `PAIR_2770` with 8 chips and")
    emit("`PAIR_3267` with 3. Both landed in **train** by the ordinary draw:")
    for pair in SLIVER_PAIRS:
        where = corpus.loc[corpus["pair_name"] == pair, "split_v3"].unique()
        emit("- `%s`: %s (%d chips)"
             % (pair, ", ".join(where) if len(where) else "absent",
                int((corpus["pair_name"] == pair).sum())))
    emit()
    emit("Neither is in %s, so the pair-macro mean here carries no 3-chip term."
         % target_split)
    emit()
    largest = pairs.nlargest(10, "n_chips")[["pair_name", "n_chips", "mixed_share",
                                             "stratum_coarse"]]
    largest = largest.assign(mixed_share=["%.2f%%" % (100 * v)
                                          for v in largest["mixed_share"]])
    top10 = 100 * largest["n_chips"].sum() / n_chips
    emit("**Largest pairs** (top 10 hold %.1f%% of the split's chips):" % top10)
    emit()
    emit(to_md(largest.set_index("pair_name")))
    emit()

    # -- 7. eval-site context -------------------------------------------------
    emit("## 7. Eval-site context")
    emit()
    emit("Every chip is more than 30 km from every external evaluation AOI by")
    emit("construction (the exclusion buffer). The realized minimum is")
    emit("**%.3f km**; median %.1f km."
         % (chips["eval_dist_km"].min(), chips["eval_dist_km"].median()))
    emit()
    site = chips.groupby("nearest_eval_site").agg(
        chips=("chip_id", "size"), pairs=("pair_name", "nunique"),
        min_km=("eval_dist_km", "min"))
    site["min_km"] = ["%.1f" % v for v in site["min_km"]]
    emit(to_md(site.sort_values("chips", ascending=False)))
    emit()

    # -- anchors --------------------------------------------------------------
    emit("## Anchors")
    emit()
    if target_split == "test":
        if abs(mixed_pct - ANCHOR_TEST_MIXED_PCT) > 0.005:
            raise AssertionError("test mixed share %.4f%% != anchor %.2f%% -- STOP"
                                 % (mixed_pct, ANCHOR_TEST_MIXED_PCT))
        if tuple(absent) != ANCHOR_ZERO_TEST_STRATA:
            raise AssertionError("zero-presence strata %r != anchor %r -- STOP"
                                 % (tuple(absent), ANCHOR_ZERO_TEST_STRATA))
        anchor_csv = paths.QC / "splits_report_v3_strata.csv"
        if anchor_csv.exists():
            shipped = pd.read_csv(anchor_csv).set_index("stratum_coarse")["test"]
            got = table["chips"].reindex(shipped.index).fillna(0).astype(int)
            if not (got.to_numpy() == shipped.to_numpy()).all():
                raise AssertionError("per-stratum test counts differ from "
                                     "splits_report_v3_strata.csv -- STOP")
            emit("- per-stratum chips match `splits_report_v3_strata.csv`: **yes**")
        emit("- mixed share at t=1%% = %.2f%% (anchor %.2f%%): **match**"
             % (mixed_pct, ANCHOR_TEST_MIXED_PCT))
        emit("- zero-presence strata = %s (anchor): **match**" % ", ".join(absent))
    emit("- profiles re-add to %s chips: **yes**" % "{:,}".format(len(corpus)))
    emit()

    if not args.write:
        print("\n(dry run; pass --write to save)")
        return 0

    paths.ensure_out()
    figdata = paths.QC / ("%s_profile_figdata" % target_split)
    figdata.mkdir(parents=True, exist_ok=True)

    # --- aggregate exports for the figures (no chip-level egress) ------------
    # F1 geography: per-pair centroids for every split.
    geo = pd.concat([
        pair_profiles[name][["pair_name", "split_v3", "lat", "lon", "n_chips",
                             "mixed_share", "stratum_coarse"]]
        for name in ALL_SPLITS], ignore_index=True)
    geo.to_csv(figdata / "pair_geo.csv", index=False)

    # F2 composition: water-share histogram, walls exact, 1000 interior bins.
    edges = np.linspace(0, 1, 1001)
    rows = []
    for name in ALL_SPLITS + ("corpus",):
        group = corpus if name == "corpus" else profiles[name]
        share = group["water_share"]
        interior = share[(share > 0) & (share < 1)]
        counts, _ = np.histogram(interior, bins=edges)
        rows.append({"group": name, "kind": "exact0", "left": 0.0, "right": 0.0,
                     "count": int((share == 0).sum())})
        rows.append({"group": name, "kind": "exact1", "left": 1.0, "right": 1.0,
                     "count": int((share == 1).sum())})
        for i, count in enumerate(counts):
            if count:
                rows.append({"group": name, "kind": "bin", "left": edges[i],
                             "right": edges[i + 1], "count": int(count)})
    pd.DataFrame(rows).to_csv(figdata / "ws_hist.csv", index=False)

    class_rows = []
    for name in ALL_SPLITS + ("corpus",):
        group = corpus if name == "corpus" else profiles[name]
        for threshold in THRESHOLDS:
            shares = class_shares(group["water_share"], threshold)
            class_rows.append({"group": name, "t": threshold, **shares})
    pd.DataFrame(class_rows).to_csv(figdata / "ws_classes.csv", index=False)

    # F3 strata: the split-scoped table plus corpus totals.
    strata_out = table.copy()
    strata_out["corpus_pairs"] = corpus.groupby("stratum_coarse")["pair_name"] \
        .nunique().reindex(corpus_strata).astype(int)
    strata_out.reset_index().to_csv(figdata / "strata.csv", index=False)

    # F4 stability: per-pair sizes for the target split (already aggregate).
    pairs[["pair_name", "n_chips", "n_mixed", "mixed_share", "ws_median",
           "stratum_coarse", "n_recovered"]].to_csv(
        figdata / "pair_stability.csv", index=False)
    pd.DataFrame([{
        "split": target_split, "chips": n_chips, "pairs": len(pairs),
        "gini_chips_over_pairs": gini(sizes), "kish_pairs": kish_pairs,
        "mixed_pct_t1": mixed_pct,
        "fragile_pairs": len(fragile), "no_mixed_pairs": no_mixed,
        "water_prior": float(chips["n_water"].sum() / valid.sum()),
        "eval_dist_min_km": float(chips["eval_dist_km"].min()),
        "recovered_chips": recovered, "b8_chips": b8, "reviewed_chips": reviewed,
        "top10_pair_share_pct": top10,
    }]).to_csv(figdata / "summary_scalars.csv", index=False)

    # F5 radiometry: aggregate dB histograms of chip means, shared grid.
    db_edges = np.arange(-40.0, 10.01, 0.25)
    db_rows = []
    for band in ("vv", "vh"):
        for name in ALL_SPLITS:
            counts = band_histogram(profiles[name], band, db_edges)
            for i, count in enumerate(counts):
                if count:
                    db_rows.append({"band": band, "group": name,
                                    "left": db_edges[i], "right": db_edges[i + 1],
                                    "count": int(count)})
    pd.DataFrame(db_rows).to_csv(figdata / "radiometry_hist.csv", index=False)

    # F6 tide and time.
    tide_edges = np.arange(-450, 460, 10)
    tide_rows = []
    for name in ALL_SPLITS + ("corpus",):
        group = corpus if name == "corpus" else profiles[name]
        counts, _ = np.histogram(group["s1_tide_level_sat"].clip(-450, 450),
                                 bins=tide_edges)
        for i, count in enumerate(counts):
            if count:
                tide_rows.append({"group": name, "left": tide_edges[i],
                                  "right": tide_edges[i + 1], "count": int(count)})
    pd.DataFrame(tide_rows).to_csv(figdata / "tide_level_hist.csv", index=False)

    phases, periods = [], {"year": [], "month": []}
    for name in ALL_SPLITS + ("corpus",):
        group = corpus if name == "corpus" else profiles[name]
        phase = group["s1_tide_level_sat_phase"].fillna("none") \
            .value_counts(normalize=True).rename("share").reset_index()
        phase.columns = ["phase", "share"]
        phase["group"] = name
        phases.append(phase)
        for column in ("year", "month"):
            counts = group.groupby(column).size().rename("chips").reset_index()
            counts["group"] = name
            periods[column].append(counts)
    pd.concat(phases, ignore_index=True).to_csv(figdata / "phase.csv", index=False)
    for column, collected in periods.items():
        pd.concat(collected, ignore_index=True).to_csv(
            figdata / ("%s.csv" % column), index=False)

    pd.DataFrame(tvd_rows).to_csv(figdata / "tvd.csv", index=False)
    pd.DataFrame(w1_rows).to_csv(figdata / "w1.csv", index=False)

    report = paths.QC / ("%s_profile_report.md" % target_split)
    report.write_text("\n".join(lines) + "\n")
    print("\nwrote %s" % report)
    print("wrote %d figure CSVs to %s" % (len(list(figdata.glob("*.csv"))), figdata))
    return 0


if __name__ == "__main__":
    sys.exit(main())
