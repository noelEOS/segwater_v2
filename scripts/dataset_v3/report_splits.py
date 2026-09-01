#!/usr/bin/env python3
"""QC report on the v3 split assignment: composition, balance, disclosures.

Reads the manifest after ``apply_splits.py`` and writes a human-readable
markdown report plus a machine-readable CSV of the stratum x split table to
``paths.QC``. Reporting is kept out of ``apply_splits.py`` on purpose: the
``apply_*`` stages are fast decisions, and this can be re-run freely without
touching the manifest.

Contents:

* per split: chips, pairs, strata occupied, realized share vs 70/20/10, Gini
  of chip mass over pairs and the Kish effective pair count (estimators as in
  ``scripts/analysis/split_distribution_comparison.py``);
* the full stratum x split chip table;
* composition marginals (coastline class, climate) per split with total
  variation distance against the corpus;
* water-share composition per split (n_water / (n_water + n_land), classes at
  the 1% purity threshold -- always state the threshold: the corpus is
  U-shaped and "mixed" swings 47.8% -> 25.2% -> 17.8% as the threshold moves
  0 -> 1% -> 5%);
* disclosure blocks: the thin strata forced to train, the sliver pairs left by
  the eval buffer, the buffer-touched pairs, the 32 pins;
* pair movement vs the old ``pairbased_split`` (a sanity check that the new
  draw is not accidentally the old assignment).
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys

import numpy as np
import pandas as pd

import paths

EXPECTED_ASSIGNED = 1_322_788
TARGETS = {"train": 0.70, "val": 0.20, "test": 0.10}
MIXED_THRESHOLD = 0.01

PAIR_SUMMARY_CSV = (paths.DB_SLIM
                    / "pair_summary_with_splits-ready_for_memmap_v2_creation.csv")


def gini(sizes: np.ndarray) -> float:
    s = np.sort(sizes.astype(np.float64))
    n = len(s)
    return float((2 * np.arange(1, n + 1) - n - 1).dot(s) / (n * s.sum()))


def kish(sizes: np.ndarray) -> float:
    s = sizes.astype(np.float64)
    return float(s.sum() ** 2 / (s ** 2).sum())


def tvd(p: pd.Series, q: pd.Series) -> float:
    keys = p.index.union(q.index)
    return float(0.5 * (p.reindex(keys, fill_value=0.0)
                        - q.reindex(keys, fill_value=0.0)).abs().sum())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()

    manifest = pd.read_parquet(
        paths.require(paths.MANIFESTS / "dataset_v3_manifest.parquet", "v3 manifest"))
    for column in ("split_v3", "excluded_eval_buffer", "split_v3_reason"):
        if column not in manifest.columns:
            raise AssertionError("%s missing; run apply_splits.py first" % column)

    strata = pd.read_csv(paths.require(PAIR_SUMMARY_CSV, "pair summary"),
                         usecols=["pair_name", "stratum_coarse"])
    m = manifest[manifest["split_v3"].notna()].merge(
        strata, on="pair_name", how="left", validate="m:1")
    if len(m) != EXPECTED_ASSIGNED:
        raise AssertionError("assigned chips %d, expected %d"
                             % (len(m), EXPECTED_ASSIGNED))
    m["ws"] = m["n_water"] / (m["n_water"] + m["n_land"])
    m["cls"] = np.select([m["ws"] <= MIXED_THRESHOLD, m["ws"] >= 1 - MIXED_THRESHOLD],
                         ["pure_land", "pure_water"], "mixed")
    gcl = m["stratum_coarse"].str.split("_").str[0]
    climate = m["stratum_coarse"].str.split("_").str[1]
    m["gcl_class"] = gcl
    m["climate"] = climate

    lines: list[str] = []

    def emit(text: str = "") -> None:
        print(text)
        lines.append(text)

    emit("# v3 split report -- generated %s"
         % dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"))
    emit("")
    emit("Corpus: %s assigned chips across %d pairs; %s excluded by the eval"
         % ("{:,}".format(len(m)), m["pair_name"].nunique(),
            "{:,}".format(int(manifest["excluded_eval_buffer"].sum()))))
    emit("buffer; mixed threshold for composition classes: %g%%."
         % (100 * MIXED_THRESHOLD))
    emit("")

    emit("## Per split")
    emit("")
    emit("| split | chips | share | target | pairs | strata | Gini | Kish n_e | mixed%% |")
    emit("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    total = len(m)
    for name in ("train", "val", "test"):
        sub = m[m["split_v3"] == name]
        sizes = sub.groupby("pair_name").size().to_numpy()
        emit("| %s | %s | %.2f%% | %.0f%% | %d | %d | %.3f | %.0f | %.1f%% |"
             % (name, "{:,}".format(len(sub)), 100 * len(sub) / total,
                100 * TARGETS[name], len(sizes), sub["stratum_coarse"].nunique(),
                gini(sizes), kish(sizes), 100 * float((sub["cls"] == "mixed").mean())))
    emit("")

    emit("## Stratum x split (chips)")
    emit("")
    pivot = m.pivot_table(index="stratum_coarse", columns="split_v3",
                          values="chip_id", aggfunc="size", fill_value=0)
    pivot = pivot.reindex(columns=["train", "val", "test"], fill_value=0)
    pivot["total"] = pivot.sum(axis=1)
    emit(pivot.to_markdown())
    emit("")

    emit("## Composition vs corpus (total variation distance)")
    emit("")
    for axis in ("gcl_class", "climate", "cls"):
        corpus = m[axis].value_counts(normalize=True)
        parts = []
        for name in ("train", "val", "test"):
            d = tvd(m.loc[m["split_v3"] == name, axis].value_counts(normalize=True),
                    corpus)
            parts.append("%s %.4f" % (name, d))
        emit("- `%s`: %s" % (axis, " | ".join(parts)))
    emit("")

    emit("## Water-share composition per split (threshold %g%%)" % (100 * MIXED_THRESHOLD))
    emit("")
    comp = m.pivot_table(index="split_v3", columns="cls", values="chip_id",
                         aggfunc="size", fill_value=0)
    comp = comp.div(comp.sum(axis=1), axis=0)
    emit((100 * comp).round(2).to_markdown())
    emit("")

    emit("## Disclosures")
    emit("")
    thin = m[m["split_v3_reason"] == "thin_stratum_train"]
    emit("**Thin strata forced to train** (%d pairs, %s chips): %s"
         % (thin["pair_name"].nunique(), "{:,}".format(len(thin)),
            ", ".join(sorted(thin["stratum_coarse"].unique())) or "none"))
    emit("")
    pins = m[m["split_v3_reason"] == "pinned_train"]
    emit("**Pinned to train** (%d pairs, %s chips): %s"
         % (pins["pair_name"].nunique(), "{:,}".format(len(pins)),
            ", ".join(sorted(pins["pair_name"].unique()))))
    emit("")
    touched = manifest.loc[manifest["excluded_eval_buffer"], "pair_name"].unique()
    kept = m[m["pair_name"].isin(touched)].groupby("pair_name").size()
    emit("**Eval-buffer-touched pairs** (%d): excluded chips removed; remaining"
         % len(touched))
    emit("chips per touched pair (absent = pair fully excluded):")
    emit("")
    for pair in sorted(touched):
        n_excl = int((manifest["excluded_eval_buffer"]
                      & (manifest["pair_name"] == pair)).sum())
        n_kept = int(kept.get(pair, 0))
        note = "  <-- SLIVER" if 0 < n_kept < 10 else ("  <-- EMPTIED" if n_kept == 0 else "")
        emit("- %s: excluded %d, kept %d%s" % (pair, n_excl, n_kept, note))
    emit("")

    if "pairbased_split" in m.columns:
        old = m.groupby("pair_name")["pairbased_split"].first()
        new = m.groupby("pair_name")["split_v3"].first()
        both = old.notna()
        moved = int((old[both] != new[both]).sum())
        emit("**Movement vs the old pair-based split**: %d of %d pairs changed split."
             % (moved, int(both.sum())))
        emit("")

    if not args.write:
        print("\n(dry run; pass --write to save)")
        return 0

    paths.ensure_out()
    report = paths.QC / "splits_report_v3.md"
    report.write_text("\n".join(lines) + "\n")
    table_target = paths.QC / "splits_report_v3_strata.csv"
    pivot.to_csv(table_target)
    print("\nwrote %s and %s" % (report, table_target))
    return 0


if __name__ == "__main__":
    sys.exit(main())
