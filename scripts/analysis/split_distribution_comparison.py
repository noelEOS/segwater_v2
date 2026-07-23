#!/usr/bin/env python
"""
split_distribution_comparison.py — characterize distribution differences between
the two training-dataset variants: the shipped CHIP-level split (`split`, leaky)
vs the PAIR-based split (`split_pair_based`, scene-pure).

Both variants are the SAME 1,474,047 selected chips; only the split-label column
differs. This script compares, per split (default train+val), the two schemes on:

  1. diversity   — distinct pairs, chips-per-pair concentration (Gini, Kish
                   effective N, top-k pair share), Lorenz curves
  2. composition — main_gcl_class / broad_climate / stratum_coarse marginals
                   (chip-weighted) + TVD & JSD divergence across schemes
  3. tide        — s1/s2 tide level + phase + s1–s2 delta distributions
  4. temporal    — S1 acquisition year/month marginals
  5. spatial     — chip-centroid latitude bands, hemisphere shares, and the
                   "lost pairs" table (pairs chip-train saw but pair-train lost)
  6. radiometry  — per-split aggregate VV/VH histograms (200 fixed bins) with
                   TVD / JSD / Wasserstein-1 + pair-level bootstrap CIs

Outputs (all numbers quotable ONLY from these CSVs; see HEADLINE_NUMBERS.md):
  docs/roadmap_for_publication/dataset_build_evidence/split_distribution_comparison/
    tables/*.csv   figures/*.{png,pdf}   HEADLINE_NUMBERS.md

Usage (eda env has pandas/pyarrow):
  /opt/homebrew/Caskroom/miniforge/base/envs/eda/bin/python \
      scripts/analysis/split_distribution_comparison.py            # full run
  ... --hist-sample 200        # fast smoke test (radiometry on 200 pairs/scheme-split)
  ... --bootstrap-reps 0       # skip bootstrap CIs

Provenance constants asserted against docs/roadmap_for_publication/DATASET_BUILD_LEDGER.md.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import pyarrow.parquet as pq

# ----------------------------------------------------------------------------- paths
SLIM = "/Volumes/noel_wd_black_sn850x/segwater-database/DATABASE/SLIM_PARQUET"
DEFAULT_PARQUET = os.path.join(
    SLIM,
    "Global_Sen12_Coast_2017_2024_CHIPS_DATABASE_with_splits_w_metadata_w_tides"
    "_w_path_passedQC_memmap_selected-V2-w_histograms-CONFIRMED-RESPLIT_PATHS_FIXED.parquet",
)
DEFAULT_PAIR_CSV = os.path.join(SLIM, "pair_summary_with_splits-ready_for_memmap_v2_creation.csv")
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_OUT = os.path.join(
    REPO, "docs", "roadmap_for_publication", "dataset_build_evidence", "split_distribution_comparison"
)

# ------------------------------------------------------------------- ledger constants
# Source: docs/roadmap_for_publication/DATASET_BUILD_LEDGER.md (asserted, fail-fast)
TOTAL_SELECTED = 1_474_047
CHIP_SPLIT_COUNTS = {"train": 1_032_526, "val": 294_643, "test": 146_878}
PAIR_SPLIT_COUNTS = {"train": 1_032_004, "val": 291_771, "test": 150_272}
PAIRS_SELECTED = 3_383
CHIP_TRAIN_PAIRS = 3_376
PAIR_SPLIT_PAIRS = {"train": 2_310, "val": 695, "test": 378}

SCHEMES = {"chip": "split", "pair": "split_pair_based"}
SCHEME_LABEL = {"chip": "chip-level (shipped)", "pair": "pair-based"}

GCL = {0: "Artificial", 1: "Biogenic", 2: "Sandy", 3: "Muddy", 4: "Rocky", 5: "Estuary"}
CLIM = {"A": "A Tropical", "B": "B Arid", "C": "C Temperate", "D": "D Cold", "E": "E Polar"}

# dataviz skill categorical palette (light mode, validated): scheme colors
COL_CHIP = "#2a78d6"   # slot 1 blue  = chip-level (shipped)
COL_PAIR = "#eb6834"   # slot 2 orange = pair-based
COL_AQUA = "#1baf7a"   # slot 3 (kept pairs in map)
SCHEME_COL = {"chip": COL_CHIP, "pair": COL_PAIR}
INK, INK2, MUTED, GRID, BASE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"

mpl.rcParams.update({
    "font.size": 11, "axes.edgecolor": BASE, "axes.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "ytick.color": INK2, "xtick.color": INK2, "axes.labelcolor": "#2a2a28",
    "text.color": "#1a1a19", "figure.facecolor": "white", "axes.facecolor": "white",
})


# ----------------------------------------------------------------------------- helpers
def tvd(p: np.ndarray, q: np.ndarray) -> float:
    """Total variation distance between two aligned probability vectors."""
    return float(0.5 * np.abs(p - q).sum())


def jsd_bits(p: np.ndarray, q: np.ndarray) -> float:
    """Jensen–Shannon divergence in bits (base 2). Finite even with zero cells."""
    m = 0.5 * (p + q)
    def kl(a, b):
        mask = a > 0
        return float((a[mask] * np.log2(a[mask] / b[mask])).sum())
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def wasserstein1(mids: np.ndarray, p: np.ndarray, q: np.ndarray) -> float:
    """W1 (earth-mover) between histograms on shared ordered bin midpoints."""
    cdf_diff = np.cumsum(p - q)[:-1]
    widths = np.diff(mids)
    return float(np.abs(cdf_diff * widths).sum())


def norm_probs(counts: pd.Series, index) -> np.ndarray:
    v = counts.reindex(index).fillna(0).to_numpy(dtype=float)
    return v / v.sum()


def gini(x: np.ndarray) -> float:
    x = np.sort(np.asarray(x, dtype=float))
    n = len(x)
    return float((2 * np.arange(1, n + 1) - n - 1).dot(x) / (n * x.sum()))


def kish_effective_n(chips_per_pair: np.ndarray) -> float:
    """Kish effective number of *pairs* given unequal chip mass per pair."""
    x = np.asarray(chips_per_pair, dtype=float)
    return float(x.sum() ** 2 / (x ** 2).sum())


def save_fig(fig, outdir: str, name: str):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(outdir, "figures", f"{name}.{ext}"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote figures/{name}.png|.pdf")


def write_table(df: pd.DataFrame, outdir: str, name: str):
    path = os.path.join(outdir, "tables", f"{name}.csv")
    df.to_csv(path, index=False)
    print(f"  wrote tables/{name}.csv  ({len(df)} rows)")


# ----------------------------------------------------------------------------- loading
def resolve_inputs(args):
    for label, path in (("parquet", args.parquet), ("pair CSV", args.pair_csv)):
        if not os.path.exists(path):
            sys.exit(
                f"ERROR: {label} not found: {path}\n"
                "Is the black SSD (/Volumes/noel_wd_black_sn850x) mounted?"
            )


def load_metadata(parquet_path: str) -> pd.DataFrame:
    cols = [
        "pair_name", "split", "split_pair_based", "selected_for_memmap",
        "CHIP_BBOX", "system:time_start_s1",
        "s1_tide_level_sat", "s1_tide_level_sat_phase",
        "s2_tide_level_sat", "s2_tide_level_sat_phase",
        "s1_s2_tide_level_sat_delta_first_round",
    ]
    t0 = time.time()
    df = pq.read_table(parquet_path, columns=cols).to_pandas()
    sel = df["selected_for_memmap"].fillna(False).astype(bool)
    df = df.loc[sel].reset_index(drop=True)
    print(f"loaded metadata: {len(df):,} selected rows in {time.time()-t0:.1f}s")
    return df


def assert_ledger_counts(df: pd.DataFrame):
    assert len(df) == TOTAL_SELECTED, f"selected count {len(df):,} != {TOTAL_SELECTED:,}"
    for split, n in CHIP_SPLIT_COUNTS.items():
        got = int((df["split"] == split).sum())
        assert got == n, f"chip split {split}: {got:,} != {n:,}"
    for split, n in PAIR_SPLIT_COUNTS.items():
        got = int((df["split_pair_based"] == split).sum())
        assert got == n, f"pair split {split}: {got:,} != {n:,}"
    n_pairs = df["pair_name"].nunique()
    assert n_pairs == PAIRS_SELECTED, f"distinct pairs {n_pairs} != {PAIRS_SELECTED}"
    assert df.groupby("pair_name")["split_pair_based"].nunique().max() == 1, \
        "pair-based split is not pair-pure"
    got = df.loc[df["split"] == "train", "pair_name"].nunique()
    assert got == CHIP_TRAIN_PAIRS, f"chip-train pairs {got} != {CHIP_TRAIN_PAIRS}"
    for split, n in PAIR_SPLIT_PAIRS.items():
        got = df.loc[df["split_pair_based"] == split, "pair_name"].nunique()
        assert got == n, f"pair-based {split} pairs {got} != {n}"
    print("ledger assertions passed (DATASET_BUILD_LEDGER.md counts reproduced)")


def join_pair_covariates(df: pd.DataFrame, pair_csv: str) -> pd.DataFrame:
    pairs = pd.read_csv(pair_csv, usecols=["pair_name", "main_gcl_class", "broad_climate", "stratum_coarse"])
    out = df.merge(pairs, on="pair_name", how="left", validate="many_to_one")
    n_missing = int(out["main_gcl_class"].isna().sum())
    assert n_missing == 0, f"{n_missing} selected chips have no pair-covariate match"
    return out


# ----------------------------------------------------------------- family 1: diversity
def diversity_family(df: pd.DataFrame, splits, outdir) -> pd.DataFrame:
    rows, quant_rows, lorenz = [], [], {}
    for scheme, col in SCHEMES.items():
        for split in splits:
            cpp = df.loc[df[col] == split].groupby("pair_name").size().to_numpy()
            cpp_sorted = np.sort(cpp)[::-1]
            total = int(cpp.sum())
            rows.append({
                "scheme": scheme, "split": split, "n_chips": total, "n_pairs": len(cpp),
                "chips_per_pair_mean": cpp.mean(), "chips_per_pair_median": float(np.median(cpp)),
                "chips_per_pair_max": int(cpp.max()), "gini_chips_over_pairs": gini(cpp),
                "kish_effective_n_pairs": kish_effective_n(cpp),
                "top1pct_pairs_chip_share": cpp_sorted[: max(1, len(cpp) // 100)].sum() / total,
                "top10_pairs_chip_share": cpp_sorted[:10].sum() / total,
            })
            for q in (0.10, 0.25, 0.50, 0.75, 0.90, 0.99):
                quant_rows.append({"scheme": scheme, "split": split, "quantile": q,
                                   "chips_per_pair": float(np.quantile(cpp, q))})
            cum = np.cumsum(np.sort(cpp)) / total
            lorenz[(scheme, split)] = np.insert(cum, 0, 0.0)
    div = pd.DataFrame(rows)
    write_table(div, outdir, "diversity_by_split")
    write_table(pd.DataFrame(quant_rows), outdir, "chips_per_pair_quantiles")

    fig, ax = plt.subplots(figsize=(5.6, 5.0))
    for (scheme, split), cum in lorenz.items():
        if split != "train":
            continue
        x = np.linspace(0, 1, len(cum))
        ax.plot(x, cum, color=SCHEME_COL[scheme], lw=2,
                label=f"{SCHEME_LABEL[scheme]} train")
    ax.plot([0, 1], [0, 1], color=MUTED, lw=0.9, ls=":")
    ax.set_xlabel("cumulative share of pairs (sorted by size)")
    ax.set_ylabel("cumulative share of chips")
    ax.set_title("Chip mass over pairs — train sets", loc="left", fontsize=12, fontweight="bold", pad=10)
    ax.grid(color=GRID, lw=0.7, zorder=0)
    ax.legend(frameon=False, fontsize=10, loc="upper left")
    save_fig(fig, outdir, "fig_chips_per_pair_lorenz")
    return div


# --------------------------------------------------------------- family 2: composition
def composition_family(df: pd.DataFrame, splits, outdir):
    marg_rows, div_rows = [], []
    covs = {"main_gcl_class": sorted(df["main_gcl_class"].unique()),
            "broad_climate": sorted(df["broad_climate"].unique()),
            "stratum_coarse": sorted(df["stratum_coarse"].unique())}
    probs = {}
    for cov, cats in covs.items():
        for scheme, col in SCHEMES.items():
            for split in splits:
                counts = df.loc[df[col] == split, cov].value_counts()
                p = norm_probs(counts, cats)
                probs[(cov, scheme, split)] = p
                for c, pi, ni in zip(cats, p, counts.reindex(cats).fillna(0).astype(int)):
                    marg_rows.append({"covariate": cov, "category": c, "scheme": scheme,
                                      "split": split, "n_chips": int(ni), "share": pi})
        for split in splits:
            p, q = probs[(cov, "chip", split)], probs[(cov, "pair", split)]
            div_rows.append({"covariate": cov, "split": split,
                             "tvd": tvd(p, q), "jsd_bits": jsd_bits(p, q),
                             "n_categories": len(cats)})
    write_table(pd.DataFrame(marg_rows), outdir, "composition_marginals")
    write_table(pd.DataFrame(div_rows), outdir, "composition_divergence")

    # figure: class + climate train marginals, both schemes
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0))
    for ax, cov, names, order in (
        (axes[0], "main_gcl_class", GCL, [2, 4, 0, 1, 3, 5]),
        (axes[1], "broad_climate", CLIM, ["C", "A", "B", "D", "E"]),
    ):
        cats = covs[cov]
        y = np.arange(len(order))[::-1]
        h = 0.36
        for k, (scheme, off) in enumerate((("chip", h / 2), ("pair", -h / 2))):
            p = probs[(cov, scheme, "train")]
            vals = [100 * p[cats.index(c)] for c in order]
            ax.barh(y + off, vals, height=h, color=SCHEME_COL[scheme], zorder=3,
                    label=SCHEME_LABEL[scheme] if ax is axes[0] else None)
        ax.set_yticks(y)
        ax.set_yticklabels([names[c] for c in order])
        ax.set_xlabel("% of train chips")
        ax.grid(axis="x", color=GRID, lw=0.7, zorder=0)
    axes[0].set_title("Train composition by coastline type", loc="left",
                      fontsize=12, fontweight="bold", pad=10)
    axes[1].set_title("…by Köppen climate group", loc="left", fontsize=12,
                      fontweight="bold", pad=10)
    axes[0].legend(frameon=False, fontsize=10, loc="lower right")
    fig.tight_layout()
    save_fig(fig, outdir, "fig_composition_marginals")


# ---------------------------------------------------------------------- family 3: tide
def tide_family(df: pd.DataFrame, splits, outdir):
    dist_rows, div_rows = [], []
    # tide units: centimetres (FES2022b `pure_tide`, cm rel. MSL — DATASET_segwater.md)
    num_cols = {"s1_tide_level_sat": "s1_tide_level_cm",
                "s2_tide_level_sat": "s2_tide_level_cm",
                "s1_s2_tide_level_sat_delta_first_round": "s1_s2_tide_delta_cm"}
    ecdf_data = {}
    for raw, label in num_cols.items():
        x_all = df[raw].to_numpy()
        finite = np.isfinite(x_all)
        lo, hi = np.quantile(x_all[finite], [0.001, 0.999])
        edges = np.linspace(lo, hi, 61)
        mids = 0.5 * (edges[:-1] + edges[1:])
        probs = {}
        for scheme, col in SCHEMES.items():
            for split in splits:
                m = (df[col] == split).to_numpy() & finite
                x = np.clip(x_all[m], lo, hi)
                n_null = int(((df[col] == split).to_numpy() & ~finite).sum())
                hist, _ = np.histogram(x, bins=edges)
                p = hist / hist.sum()
                probs[(scheme, split)] = p
                dist_rows.append({"variable": label, "scheme": scheme, "split": split,
                                  "n": int(m.sum()), "n_null_dropped": n_null,
                                  "mean": float(x.mean()), "std": float(x.std(ddof=1)),
                                  "p05": float(np.quantile(x, 0.05)),
                                  "p50": float(np.quantile(x, 0.50)),
                                  "p95": float(np.quantile(x, 0.95))})
                if raw == "s1_tide_level_sat" and split == "train":
                    ecdf_data[scheme] = np.sort(x)
        for split in splits:
            p, q = probs[("chip", split)], probs[("pair", split)]
            div_rows.append({"variable": label, "split": split, "tvd": tvd(p, q),
                             "jsd_bits": jsd_bits(p, q),
                             "wasserstein1_cm": wasserstein1(mids, p, q)})
    # phase marginals
    for raw, label in (("s1_tide_level_sat_phase", "s1_phase"), ("s2_tide_level_sat_phase", "s2_phase")):
        cats = sorted(df[raw].dropna().unique())
        for scheme, col in SCHEMES.items():
            for split in splits:
                counts = df.loc[df[col] == split, raw].value_counts()
                p = norm_probs(counts, cats)
                for c, pi in zip(cats, p):
                    dist_rows.append({"variable": f"{label}:{c}", "scheme": scheme, "split": split,
                                      "n": int(counts.reindex([c]).fillna(0).iloc[0]),
                                      "n_null_dropped": int(df.loc[df[col] == split, raw].isna().sum()),
                                      "mean": pi, "std": np.nan, "p05": np.nan, "p50": np.nan, "p95": np.nan})
        for split in splits:
            p = norm_probs(df.loc[df["split"] == split, raw].value_counts(), cats)
            q = norm_probs(df.loc[df["split_pair_based"] == split, raw].value_counts(), cats)
            div_rows.append({"variable": label, "split": split, "tvd": tvd(p, q),
                             "jsd_bits": jsd_bits(p, q), "wasserstein1_cm": np.nan})
    write_table(pd.DataFrame(dist_rows), outdir, "tide_distributions")
    write_table(pd.DataFrame(div_rows), outdir, "tide_divergence")

    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    for scheme, x in ecdf_data.items():
        step = max(1, len(x) // 4000)
        ax.plot(x[::step], np.linspace(0, 1, len(x))[::step], color=SCHEME_COL[scheme],
                lw=2, label=f"{SCHEME_LABEL[scheme]} train")
    ax.set_xlabel("S1 tide level at acquisition (cm, FES2022b rel. MSL)")
    ax.set_ylabel("ECDF")
    ax.set_title("S1 tide-level distribution — train sets", loc="left",
                 fontsize=12, fontweight="bold", pad=10)
    ax.grid(color=GRID, lw=0.7, zorder=0)
    ax.legend(frameon=False, fontsize=10, loc="lower right")
    save_fig(fig, outdir, "fig_tide_level_ecdf")


# ------------------------------------------------------------------ family 4: temporal
def temporal_family(df: pd.DataFrame, splits, outdir):
    ts = pd.to_datetime(df["system:time_start_s1"], utc=True, format="mixed")
    df = df.assign(_year=ts.dt.year, _month=ts.dt.month)
    rows, div_rows = [], []
    for var, cats in (("_year", sorted(df["_year"].unique())), ("_month", list(range(1, 13)))):
        probs = {}
        for scheme, col in SCHEMES.items():
            for split in splits:
                counts = df.loc[df[col] == split, var].value_counts()
                p = norm_probs(counts, cats)
                probs[(scheme, split)] = p
                for c, pi, ni in zip(cats, p, counts.reindex(cats).fillna(0).astype(int)):
                    rows.append({"variable": var.strip("_"), "category": int(c), "scheme": scheme,
                                 "split": split, "n_chips": int(ni), "share": pi})
        for split in splits:
            p, q = probs[("chip", split)], probs[("pair", split)]
            div_rows.append({"variable": var.strip("_"), "split": split,
                             "tvd": tvd(p, q), "jsd_bits": jsd_bits(p, q)})
    write_table(pd.DataFrame(rows), outdir, "temporal_distributions")
    write_table(pd.DataFrame(div_rows), outdir, "temporal_divergence")

    tmp = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    w = 0.38
    x = np.arange(1, 13)
    for scheme, off in (("chip", -w / 2), ("pair", w / 2)):
        sub = tmp[(tmp.variable == "month") & (tmp.scheme == scheme) & (tmp.split == "train")]
        vals = sub.set_index("category")["share"].reindex(range(1, 13)).fillna(0) * 100
        ax.bar(x + off, vals, width=w, color=SCHEME_COL[scheme], zorder=3,
               label=f"{SCHEME_LABEL[scheme]} train", edgecolor="white", linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"])
    ax.set_ylabel("% of train chips")
    ax.set_title("S1 acquisition month — train sets", loc="left", fontsize=12,
                 fontweight="bold", pad=10)
    ax.grid(axis="y", color=GRID, lw=0.7, zorder=0)
    ax.legend(frameon=False, fontsize=10)
    save_fig(fig, outdir, "fig_temporal_month")


# ------------------------------------------------------------------- family 5: spatial
def spatial_family(df: pd.DataFrame, splits, outdir):
    bbox = np.stack([np.fromstring(s.strip("[]"), sep=",") for s in df["CHIP_BBOX"]])
    lon = 0.5 * (bbox[:, 0] + bbox[:, 2])
    lat = 0.5 * (bbox[:, 1] + bbox[:, 3])
    df = df.assign(_lon=lon, _lat=lat)

    band_edges = np.arange(-90, 91, 10)
    band_labels = [f"{lo:+03d}..{lo+10:+03d}" for lo in band_edges[:-1]]
    rows, div_rows = [], []
    probs = {}
    for scheme, col in SCHEMES.items():
        for split in splits:
            x = df.loc[df[col] == split, "_lat"].to_numpy()
            hist, _ = np.histogram(x, bins=band_edges)
            p = hist / hist.sum()
            probs[(scheme, split)] = p
            for lab, ni, pi in zip(band_labels, hist, p):
                rows.append({"latitude_band_deg": lab, "scheme": scheme, "split": split,
                             "n_chips": int(ni), "share": pi})
            rows.append({"latitude_band_deg": "share_north_hemisphere", "scheme": scheme,
                         "split": split, "n_chips": int((x >= 0).sum()),
                         "share": float((x >= 0).mean())})
    for split in splits:
        p, q = probs[("chip", split)], probs[("pair", split)]
        div_rows.append({"variable": "latitude_band_10deg", "split": split,
                         "tvd": tvd(p, q), "jsd_bits": jsd_bits(p, q)})
    write_table(pd.DataFrame(rows), outdir, "spatial_coverage")
    write_table(pd.DataFrame(div_rows), outdir, "spatial_divergence")

    # lost-pairs table: pairs present in chip-level train, absent from pair-based train
    per_pair = (df.groupby("pair_name")
                  .agg(n_chips=("pair_name", "size"), lon=("_lon", "mean"), lat=("_lat", "mean"),
                       split_pair_based=("split_pair_based", "first"),
                       main_gcl_class=("main_gcl_class", "first"),
                       broad_climate=("broad_climate", "first"))
                  .reset_index())
    chip_train_pairs = set(df.loc[df["split"] == "train", "pair_name"].unique())
    per_pair["in_chip_train"] = per_pair["pair_name"].isin(chip_train_pairs)
    per_pair["in_pair_train"] = per_pair["split_pair_based"] == "train"
    lost = per_pair[per_pair["in_chip_train"] & ~per_pair["in_pair_train"]].copy()
    write_table(lost.drop(columns=["in_chip_train", "in_pair_train"]), outdir, "spatial_lost_pairs")

    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    kept = per_pair[per_pair["in_pair_train"]]
    ax.scatter(kept["lon"], kept["lat"], s=6, color=COL_AQUA, alpha=0.65, lw=0,
               label=f"kept in pair-based train (n={len(kept):,})", zorder=3)
    ax.scatter(lost["lon"], lost["lat"], s=8, color=COL_PAIR, alpha=0.8, lw=0,
               label=f"lost from train → pair-based val/test (n={len(lost):,})", zorder=4)
    ax.set_xlim(-180, 180)
    ax.set_ylim(-65, 80)
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    ax.set_title("Scene (pair) centroids: geographic coverage lost by the pair-based train split",
                 loc="left", fontsize=12, fontweight="bold", pad=10)
    ax.grid(color=GRID, lw=0.7, zorder=0)
    ax.legend(frameon=False, fontsize=10, loc="lower left")
    save_fig(fig, outdir, "fig_spatial_map")

    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    y = np.arange(len(band_labels))
    h = 0.38
    for scheme, off in (("chip", h / 2), ("pair", -h / 2)):
        ax.barh(y + off, 100 * probs[(scheme, "train")], height=h,
                color=SCHEME_COL[scheme], zorder=3, label=f"{SCHEME_LABEL[scheme]} train")
    keep = [i for i in range(len(band_labels))
            if probs[("chip", "train")][i] > 0 or probs[("pair", "train")][i] > 0]
    ax.set_yticks(y[keep])
    ax.set_yticklabels([band_labels[i] for i in keep], fontsize=9)
    ax.set_ylim(min(keep) - 0.7, max(keep) + 0.7)
    ax.set_xlabel("% of train chips")
    ax.set_title("Latitude-band composition — train sets", loc="left", fontsize=12,
                 fontweight="bold", pad=10)
    ax.grid(axis="x", color=GRID, lw=0.7, zorder=0)
    ax.legend(frameon=False, fontsize=10, loc="lower right")
    save_fig(fig, outdir, "fig_spatial_latitude_bands")


# ---------------------------------------------------------------- family 6: radiometry
def parse_counts(s: str) -> np.ndarray:
    return np.fromstring(s.strip("[]"), sep=",", dtype=float).astype(np.int64)


def radiometry_family(parquet_path: str, splits, outdir, hist_sample: int,
                      bootstrap_reps: int, seed: int):
    """Stream the parquet, accumulate per-(band, scheme, split, pair) histogram counts."""
    rng = np.random.default_rng(seed)
    pf = pq.ParquetFile(parquet_path)
    bands = ("vv", "vh")
    edges = {}
    pair_acc: dict = {}   # (band, scheme, split) -> {pair: int64[200]}
    rowsum_stats = {"n_rows": 0, "sum_min": None, "sum_max": None}

    sample_pairs = None
    if hist_sample > 0:
        meta = pq.read_table(parquet_path, columns=["pair_name", "selected_for_memmap"]).to_pandas()
        sel_pairs = meta.loc[meta["selected_for_memmap"].fillna(False).astype(bool), "pair_name"].unique()
        sample_pairs = set(rng.choice(sel_pairs, size=min(hist_sample, len(sel_pairs)), replace=False))
        print(f"radiometry: sampling {len(sample_pairs)} pairs (smoke test)")

    cols = ["pair_name", "split", "split_pair_based", "selected_for_memmap",
            "raw_vv_bins", "raw_vv_counts", "raw_vh_bins", "raw_vh_counts"]
    t0 = time.time()
    n_batches = 0
    for batch in pf.iter_batches(batch_size=100_000, columns=cols):
        d = batch.to_pandas()
        d = d.loc[d["selected_for_memmap"].fillna(False).astype(bool)]
        if sample_pairs is not None:
            d = d.loc[d["pair_name"].isin(sample_pairs)]
        if len(d) == 0:
            continue
        if not edges:
            for band in bands:
                edges[band] = np.fromstring(d[f"raw_{band}_bins"].iloc[0].strip("[]"), sep=",")
                # spot-check bin edges are globally shared (verified in build ledger session)
                for s in d[f"raw_{band}_bins"].iloc[:: max(1, len(d) // 20)]:
                    assert np.allclose(np.fromstring(s.strip("[]"), sep=","), edges[band]), \
                        f"non-uniform {band} bin edges"
        for band in bands:
            counts = np.stack([parse_counts(s) for s in d[f"raw_{band}_counts"]])
            assert counts.shape[1] == len(edges[band]) - 1, f"{band} count length mismatch"
            if band == "vv":
                rs = counts.sum(axis=1)
                rowsum_stats["n_rows"] += len(rs)
                lo, hi = int(rs.min()), int(rs.max())
                rowsum_stats["sum_min"] = lo if rowsum_stats["sum_min"] is None else min(lo, rowsum_stats["sum_min"])
                rowsum_stats["sum_max"] = hi if rowsum_stats["sum_max"] is None else max(hi, rowsum_stats["sum_max"])
            for scheme, col in SCHEMES.items():
                for split in splits:
                    m = (d[col] == split).to_numpy()
                    if not m.any():
                        continue
                    key = (band, scheme, split)
                    acc = pair_acc.setdefault(key, {})
                    sub = d.loc[m, "pair_name"]
                    csub = counts[m]
                    for pair, idx in sub.groupby(sub).indices.items():
                        acc[pair] = acc.get(pair, 0) + csub[idx].sum(axis=0)
        n_batches += 1
        print(f"  radiometry batch {n_batches}: {len(d):,} rows kept "
              f"({time.time()-t0:.0f}s elapsed)", flush=True)

    print(f"radiometry pass done in {time.time()-t0:.0f}s; "
          f"VV row-sum range [{rowsum_stats['sum_min']}, {rowsum_stats['sum_max']}] "
          f"over {rowsum_stats['n_rows']:,} rows (224x224 = 50,176)")

    # densities + divergences (+ optional pair bootstrap)
    hist_rows, div_rows = [], []
    for band in bands:
        mids = 0.5 * (edges[band][:-1] + edges[band][1:])
        dens = {}
        for scheme in SCHEMES:
            for split in splits:
                stack = np.stack(list(pair_acc[(band, scheme, split)].values()))
                total = stack.sum(axis=0).astype(float)
                p = total / total.sum()
                assert abs(p.sum() - 1) < 1e-9
                dens[(scheme, split)] = (p, stack)
                for m_, pi in zip(mids, p):
                    hist_rows.append({"band": band, "scheme": scheme, "split": split,
                                      "bin_mid": m_, "density": pi})
        for split in splits:
            p, sp = dens[("chip", split)]
            q, sq = dens[("pair", split)]
            row = {"band": band, "split": split, "tvd": tvd(p, q),
                   "jsd_bits": jsd_bits(p, q), "wasserstein1": wasserstein1(mids, p, q),
                   "n_pairs_chip": len(sp), "n_pairs_pair": len(sq)}
            if bootstrap_reps > 0:
                stats = {"tvd": [], "jsd_bits": [], "wasserstein1": []}
                for _ in range(bootstrap_reps):
                    bp = sp[rng.integers(0, len(sp), len(sp))].sum(axis=0).astype(float)
                    bq = sq[rng.integers(0, len(sq), len(sq))].sum(axis=0).astype(float)
                    bp, bq = bp / bp.sum(), bq / bq.sum()
                    stats["tvd"].append(tvd(bp, bq))
                    stats["jsd_bits"].append(jsd_bits(bp, bq))
                    stats["wasserstein1"].append(wasserstein1(mids, bp, bq))
                for k, v in stats.items():
                    row[f"{k}_boot_lo95"] = float(np.quantile(v, 0.025))
                    row[f"{k}_boot_hi95"] = float(np.quantile(v, 0.975))
                row["bootstrap_reps"] = bootstrap_reps
                row["bootstrap_unit"] = "pair"
            div_rows.append(row)
    hist_df = pd.DataFrame(hist_rows)
    write_table(hist_df[hist_df.band == "vv"].drop(columns="band"), outdir, "radiometry_vv_hist")
    write_table(hist_df[hist_df.band == "vh"].drop(columns="band"), outdir, "radiometry_vh_hist")
    write_table(pd.DataFrame(div_rows), outdir, "radiometry_divergence")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0), sharey=False)
    for ax, band in zip(axes, bands):
        sub = hist_df[hist_df.band == band]
        for scheme in SCHEMES:
            for split, ls in (("train", "-"), ("val", "--")):
                if split not in splits:
                    continue
                s = sub[(sub.scheme == scheme) & (sub.split == split)]
                ax.plot(s["bin_mid"], s["density"], color=SCHEME_COL[scheme], ls=ls,
                        lw=2 if split == "train" else 1.3,
                        label=f"{SCHEME_LABEL[scheme]} {split}")
        ax.set_xlabel(f"{band.upper()} backscatter (dB, raw pre-normalization bins)")
        ax.set_ylabel("density")
        ax.grid(color=GRID, lw=0.7, zorder=0)
        ax.set_title(f"{band.upper()} aggregate distribution", loc="left",
                     fontsize=12, fontweight="bold", pad=10)
    axes[0].legend(frameon=False, fontsize=9)
    fig.tight_layout()
    save_fig(fig, outdir, "fig_radiometry_vv_vh")
    return pd.DataFrame(div_rows)


# ------------------------------------------------------------------------- headline md
def write_headline(outdir, div_df, rad_div, splits, hist_sample):
    t = {n: pd.read_csv(os.path.join(outdir, "tables", f"{n}.csv"))
         for n in ("diversity_by_split", "composition_divergence", "tide_divergence",
                   "temporal_divergence", "spatial_divergence", "spatial_lost_pairs")}
    d = t["diversity_by_split"].set_index(["scheme", "split"])
    lines = [
        "# Split-distribution comparison — headline numbers",
        "",
        "Chip-level (`split`, shipped/leaky) vs pair-based (`split_pair_based`, scene-pure).",
        "Same 1,474,047 selected chips in both; only the split labels differ.",
        "Generated by `scripts/analysis/split_distribution_comparison.py`. Every number",
        "below is computed in, and quotable from, the cited CSV under `tables/`.",
        "",
    ]
    if hist_sample:
        lines += [f"**⚠ SMOKE-TEST RUN**: radiometry restricted to {hist_sample} sampled "
                  "pairs — radiometry numbers NOT quotable.", ""]
    lines += [
        "## Scene diversity (tables/diversity_by_split.csv)",
        "",
        f"- Train pairs: {int(d.loc[('chip','train'),'n_pairs']):,} (chip-level) → "
        f"{int(d.loc[('pair','train'),'n_pairs']):,} (pair-based) = "
        f"{100*(1 - d.loc[('pair','train'),'n_pairs']/d.loc[('chip','train'),'n_pairs']):.1f}% fewer distinct scenes "
        f"at ~equal chip count ({int(d.loc[('chip','train'),'n_chips']):,} vs {int(d.loc[('pair','train'),'n_chips']):,}).",
        f"- Kish effective number of pairs (train): {d.loc[('chip','train'),'kish_effective_n_pairs']:.0f} → "
        f"{d.loc[('pair','train'),'kish_effective_n_pairs']:.0f}.",
        f"- Gini of chip mass over pairs (train): {d.loc[('chip','train'),'gini_chips_over_pairs']:.3f} → "
        f"{d.loc[('pair','train'),'gini_chips_over_pairs']:.3f}.",
        "",
        "## Composition divergence, chip-level vs pair-based within each split "
        "(tables/composition_divergence.csv)",
        "",
    ]
    for _, r in t["composition_divergence"].iterrows():
        lines.append(f"- {r['covariate']} ({r['split']}): TVD = {r['tvd']:.4f}, "
                     f"JSD = {r['jsd_bits']:.5f} bits over {int(r['n_categories'])} categories.")
    lines += ["", "## Tide (tables/tide_divergence.csv)", ""]
    for _, r in t["tide_divergence"].iterrows():
        w = "" if pd.isna(r["wasserstein1_cm"]) else f", W1 = {r['wasserstein1_cm']:.2f} cm"
        lines.append(f"- {r['variable']} ({r['split']}): TVD = {r['tvd']:.4f}{w}.")
    lines += ["", "## Temporal (tables/temporal_divergence.csv)", ""]
    for _, r in t["temporal_divergence"].iterrows():
        lines.append(f"- {r['variable']} ({r['split']}): TVD = {r['tvd']:.4f}.")
    lines += ["", "## Spatial (tables/spatial_divergence.csv, tables/spatial_lost_pairs.csv)", ""]
    for _, r in t["spatial_divergence"].iterrows():
        lines.append(f"- 10° latitude bands ({r['split']}): TVD = {r['tvd']:.4f}.")
    lines.append(f"- Pairs in chip-level train but NOT in pair-based train: "
                 f"{len(t['spatial_lost_pairs']):,} (tables/spatial_lost_pairs.csv; these scenes "
                 f"moved wholly to pair-based val/test).")
    lines += ["", "## Radiometry — aggregate VV/VH input distributions "
              "(tables/radiometry_divergence.csv)", ""]
    for _, r in rad_div.iterrows():
        ci = ""
        if "tvd_boot_lo95" in r and not pd.isna(r.get("tvd_boot_lo95", np.nan)):
            ci = (f" [pair-bootstrap 95% CI: TVD {r['tvd_boot_lo95']:.4f}–{r['tvd_boot_hi95']:.4f}, "
                  f"{int(r['bootstrap_reps'])} reps]")
        lines.append(f"- {r['band'].upper()} ({r['split']}): TVD = {r['tvd']:.4f}, "
                     f"JSD = {r['jsd_bits']:.5f} bits, W1 = {r['wasserstein1']:.4f} dB{ci}.")
    lines += [
        "",
        "## Notes",
        "",
        "- No per-chip water-fraction / label-balance column exists in the build parquet;",
        "  VV/VH radiometry is the input-distribution proxy. No label statistic is computed.",
        "- Divergences compare the SAME split name across the two schemes (e.g. chip-train",
        "  vs pair-train). TVD = total variation distance (max probability mass that must",
        "  move); JSD = Jensen–Shannon divergence, base 2; W1 = Wasserstein-1 on bin midpoints.",
        "- Tide levels are centimetres (FES2022b `pure_tide`, cm rel. MSL — DATASET_segwater.md).",
        "  Tide rows drop NaN tide levels; dropped counts are in tables/tide_distributions.csv",
        "  (`n_null_dropped`).",
        "- Radiometry bins are raw dB (pre z-score normalization) — the memmap stores z-scored",
        "  values, but `raw_vv_bins`/`raw_vh_bins` are the raw-histogram edges.",
        "- Bootstrap CIs of a non-negative distance are noise-inflated: a point estimate BELOW",
        "  its pair-bootstrap CI means the observed divergence is smaller than the",
        "  pair-resampling noise floor, i.e. indistinguishable from zero at the pair level.",
    ]
    path = os.path.join(outdir, "HEADLINE_NUMBERS.md")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  wrote HEADLINE_NUMBERS.md")


# ------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--parquet", default=DEFAULT_PARQUET)
    ap.add_argument("--pair-csv", default=DEFAULT_PAIR_CSV)
    ap.add_argument("--output-dir", default=DEFAULT_OUT)
    ap.add_argument("--splits", nargs="+", default=["train", "val"],
                    choices=["train", "val", "test"])
    ap.add_argument("--hist-sample", type=int, default=0,
                    help="if >0, radiometry pass uses this many random pairs (smoke test)")
    ap.add_argument("--bootstrap-reps", type=int, default=200,
                    help="pair-level bootstrap reps for radiometry divergence CIs (0 = skip)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--skip-radiometry", action="store_true")
    args = ap.parse_args()

    resolve_inputs(args)
    for sub in ("tables", "figures"):
        os.makedirs(os.path.join(args.output_dir, sub), exist_ok=True)

    df = load_metadata(args.parquet)
    assert_ledger_counts(df)
    df = join_pair_covariates(df, args.pair_csv)

    print("\n[1/6] diversity")
    div_df = diversity_family(df, args.splits, args.output_dir)
    print("\n[2/6] composition")
    composition_family(df, args.splits, args.output_dir)
    print("\n[3/6] tide")
    tide_family(df, args.splits, args.output_dir)
    print("\n[4/6] temporal")
    temporal_family(df, args.splits, args.output_dir)
    print("\n[5/6] spatial")
    spatial_family(df, args.splits, args.output_dir)
    if args.skip_radiometry:
        print("\n[6/6] radiometry SKIPPED")
        rad_div = pd.DataFrame(columns=["band", "split", "tvd", "jsd_bits", "wasserstein1"])
    else:
        print("\n[6/6] radiometry (streaming pass over histograms)")
        rad_div = radiometry_family(args.parquet, args.splits, args.output_dir,
                                    args.hist_sample, args.bootstrap_reps, args.seed)
    write_headline(args.output_dir, div_df, rad_div, args.splits, args.hist_sample)
    print("\nDONE:", args.output_dir)


if __name__ == "__main__":
    main()
