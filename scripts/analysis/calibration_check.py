#!/usr/bin/env python3
"""
Calibration check (reliability diagrams + ECE) for Semarang water-probability rasters.

For one or more models, each evaluated on the six Semarang dates, this computes
per-scene and pooled reliability diagrams and calibration metrics from the RAW
predicted probabilities — no temperature scaling, no bootstrap.

Design
------
- Pixel access and masking REUSE the repo's overlap utilities
  (scripts/inference_overlap_utils.py): the geospatial overlap of
  reference / prediction / valid-mask is computed exactly as in
  scripts/evaluate_indonesia_inference_run_aucroc.py, and the SAME valid mask
  (reference-nodata exclusion AND external valid mask == valid_mask_value) is
  applied identically to prediction and label. We do not write a new loader; we
  replicate the loader's masking per tile so the pass can stream.
- Scale: each scene is streamed in row-block windows; a full raster is never
  held in memory. Within each tile we accumulate into fixed accumulators.

Single streaming pass per scene accumulates:
  - a FINE fixed histogram over [0, 1] with 1000 equal-width bins, storing per
    fine bin: pixel count, positive-count (sum of binary labels), and sum of
    predicted probability;
  - scalar moments for an exact Brier score: N, sum(p), sum(p^2), sum(y),
    sum(p*y).

From the fine histogram (no raster re-read) we derive two coarse 15-bin
binnings: an equal-mass (quantile) binning (PRIMARY, plotted) and an
equal-width binning (SECONDARY). Reliability and ECE come from the SAME binned
accumulators.

The pooled result aggregates the per-scene fine histograms and scalar moments;
it is labelled explicitly as repeated measures of a single AOI (Semarang),
NOT an independent-sample estimate.

Usage
-----
    python scripts/analysis/calibration_check.py \\
        --run-dir   <.../semarang_probability_aucroc__20260617T092846Z> \\
        [--models   Upernet_Swin_Base_224_native224_weighted ...] \\
        [--models-contains native224]            # select by substring \\
        [--fine-bins 1000] [--coarse-bins 15] \\
        [--block-rows 512]

By default --models-contains native224 selects all seven native224_weighted
models in the run manifest. Outputs go to <run-dir>/calibration/<model>/.
"""

from __future__ import annotations

import argparse
import json
import sys
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window

# --- repo imports: scripts/ holds inference_overlap_utils.py and the evaluation pkg
_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPTS = _REPO_ROOT / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from inference_overlap_utils import (  # noqa: E402
    read_profile,
    assert_same_crs,
    assert_same_resolution,
    intersection_bounds,
    rounded_window_from_bounds,
)

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.colors as mcolors  # noqa: E402
import matplotlib.colorbar  # noqa: E402

# Colormap for the log10(bin-count) encoding on per-scene reliability diagrams.
COUNT_CMAP = "viridis"


# --------------------------------------------------------------------------- #
# Streaming accumulator
# --------------------------------------------------------------------------- #

class FineAccumulator:
    """Fixed fine histogram over [0, 1] plus exact-Brier scalar moments."""

    def __init__(self, fine_bins: int):
        self.fine_bins = fine_bins
        self.edges = np.linspace(0.0, 1.0, fine_bins + 1)
        self.count = np.zeros(fine_bins, dtype=np.int64)
        self.pos = np.zeros(fine_bins, dtype=np.int64)
        self.sum_p = np.zeros(fine_bins, dtype=np.float64)
        # scalar moments
        self.n = 0
        self.s_p = 0.0
        self.s_p2 = 0.0
        self.s_y = 0
        self.s_py = 0.0

    def update(self, p: np.ndarray, y: np.ndarray) -> None:
        """p: float probabilities in [0,1]; y: binary {0,1} labels. Same length."""
        if p.size == 0:
            return
        # clip into [0,1] so out-of-range scores still land in the edge bins
        p = np.clip(p, 0.0, 1.0)
        # bin index in [0, fine_bins-1]
        idx = np.minimum((p * self.fine_bins).astype(np.int64), self.fine_bins - 1)
        self.count += np.bincount(idx, minlength=self.fine_bins)
        self.pos += np.bincount(idx, weights=y, minlength=self.fine_bins).astype(np.int64)
        self.sum_p += np.bincount(idx, weights=p, minlength=self.fine_bins)
        self.n += int(p.size)
        self.s_p += float(p.sum())
        self.s_p2 += float(np.dot(p, p))
        self.s_y += int(y.sum())
        self.s_py += float(np.dot(p, y))

    def merge(self, other: "FineAccumulator") -> None:
        assert self.fine_bins == other.fine_bins
        self.count += other.count
        self.pos += other.pos
        self.sum_p += other.sum_p
        self.n += other.n
        self.s_p += other.s_p
        self.s_p2 += other.s_p2
        self.s_y += other.s_y
        self.s_py += other.s_py

    def brier(self) -> float:
        # mean((p - y)^2) = (sum p^2 - 2 sum p*y + sum y) / N   (y in {0,1} => y^2=y)
        if self.n == 0:
            return float("nan")
        return (self.s_p2 - 2.0 * self.s_py + self.s_y) / self.n

    def prevalence(self) -> float:
        return (self.s_y / self.n) if self.n else float("nan")


# --------------------------------------------------------------------------- #
# Coarse binnings derived from the fine histogram (no raster re-read)
# --------------------------------------------------------------------------- #

def _bin_edges_quantile(acc: FineAccumulator, n_coarse: int) -> np.ndarray:
    """Equal-mass edges on the fine-bin grid: each coarse bin ~ N/n_coarse pixels.

    Edges are snapped to fine-bin boundaries (the finest resolution available
    without re-reading rasters). For water-segmentation probabilities, mass piles
    up near 0 (and 1): if one fine bin holds more than N/n_coarse pixels, several
    equal-mass targets land on the same fine boundary and the duplicate edges are
    collapsed, so FEWER than n_coarse distinct bins result. This is expected, not
    a defect; the realized bin count is recorded in metrics as
    n_coarse_bins_quantile.
    """
    fine_edges = acc.edges
    cum = np.cumsum(acc.count)
    n = acc.n
    if n == 0:
        return np.linspace(0.0, 1.0, n_coarse + 1)
    targets = np.linspace(0, n, n_coarse + 1)[1:-1]
    # for each target mass, find the fine-bin boundary whose cumulative count first reaches it
    inner = np.searchsorted(cum, targets, side="left") + 1
    inner = np.clip(inner, 1, acc.fine_bins)
    edges = np.concatenate(([0], inner, [acc.fine_bins])).astype(int)
    edges = np.unique(edges)  # collapse duplicates (heavy mass points)
    return fine_edges[edges]


def _bin_edges_fixed(n_coarse: int) -> np.ndarray:
    return np.linspace(0.0, 1.0, n_coarse + 1)


def _coarse_table(acc: FineAccumulator, coarse_edges: np.ndarray) -> pd.DataFrame:
    """Aggregate the fine histogram into coarse bins defined by coarse_edges.

    coarse_edges are a subset of the fine edges (quantile) or arbitrary fixed
    edges; we map each fine bin to its coarse bin by its left edge.
    """
    fine_left = acc.edges[:-1]
    # coarse bin index for each fine bin (right-open; last bin closed)
    cidx = np.searchsorted(coarse_edges, fine_left, side="right") - 1
    cidx = np.clip(cidx, 0, len(coarse_edges) - 2)
    n_coarse = len(coarse_edges) - 1

    count = np.bincount(cidx, weights=acc.count, minlength=n_coarse)
    pos = np.bincount(cidx, weights=acc.pos, minlength=n_coarse)
    sum_p = np.bincount(cidx, weights=acc.sum_p, minlength=n_coarse)

    with np.errstate(invalid="ignore", divide="ignore"):
        mean_pred = np.where(count > 0, sum_p / count, np.nan)
        obs_freq = np.where(count > 0, pos / count, np.nan)

    return pd.DataFrame({
        "bin_left": coarse_edges[:-1],
        "bin_right": coarse_edges[1:],
        "count": count.astype(np.int64),
        "mean_pred": mean_pred,
        "obs_freq": obs_freq,
    })


def _ece_mce(table: pd.DataFrame, n_total: int) -> tuple[float, float]:
    t = table[table["count"] > 0]
    if t.empty or n_total == 0:
        return float("nan"), float("nan")
    gap = np.abs(t["obs_freq"].to_numpy() - t["mean_pred"].to_numpy())
    weights = t["count"].to_numpy() / n_total
    ece = float(np.sum(weights * gap))
    mce = float(np.max(gap))
    return ece, mce


def _fine_table(acc: FineAccumulator) -> pd.DataFrame:
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_pred = np.where(acc.count > 0, acc.sum_p / acc.count, np.nan)
        obs_freq = np.where(acc.count > 0, acc.pos / acc.count, np.nan)
    return pd.DataFrame({
        "binning": "fine",
        "bin_left": acc.edges[:-1],
        "bin_right": acc.edges[1:],
        "count": acc.count,
        "mean_pred": mean_pred,
        "obs_freq": obs_freq,
    })


# --------------------------------------------------------------------------- #
# Streaming pass over one scene (reusing repo overlap + masking logic)
# --------------------------------------------------------------------------- #

def stream_scene(
    reference_path: str,
    prediction_path: str,
    valid_mask_path: str | None,
    valid_mask_value: int,
    reference_water_values: list[int],
    reference_nodata_values: list[int],
    resolution_atol: float,
    fine_bins: int,
    block_rows: int,
    log,
) -> tuple[FineAccumulator, dict]:
    """One streaming pass; returns the fine accumulator and per-scene diagnostics.

    Mirrors _load_overlap_reference_and_probability exactly: same overlap bounds,
    same rounded windows, same valid_mask = (~nodata) & (external == value),
    applied identically to label and prediction.
    """
    ref_profile = read_profile(reference_path)
    pred_profile = read_profile(prediction_path)
    mask_profile = read_profile(valid_mask_path) if valid_mask_path else None

    assert_same_crs(ref_profile, pred_profile, "prediction")
    assert_same_resolution(ref_profile, pred_profile, resolution_atol, "prediction")

    bounds_inputs = [ref_profile["bounds"], pred_profile["bounds"]]
    if mask_profile:
        assert_same_crs(ref_profile, mask_profile, "valid_mask")
        assert_same_resolution(ref_profile, mask_profile, resolution_atol, "valid_mask")
        bounds_inputs.append(mask_profile["bounds"])

    overlap_bounds = intersection_bounds(*bounds_inputs)

    acc = FineAccumulator(fine_bins)

    diag = {
        "overlap_pixels": 0,
        "valid_after_reference_nodata": 0,
        "valid_after_external_mask": 0,
        "valid_pixels_after_all_masks": 0,
        "masking_consistent": True,
    }

    with rasterio.open(reference_path) as src_ref, \
         rasterio.open(prediction_path) as src_pred:

        ref_window = rounded_window_from_bounds(overlap_bounds, src_ref.transform)
        pred_window = rounded_window_from_bounds(overlap_bounds, src_pred.transform)

        src_mask = rasterio.open(valid_mask_path) if valid_mask_path else None
        mask_window = (
            rounded_window_from_bounds(overlap_bounds, src_mask.transform)
            if src_mask else None
        )

        n_rows = int(ref_window.height)
        n_cols = int(ref_window.width)

        # shape sanity across the three windows (same as loader's shape checks)
        if (int(pred_window.height), int(pred_window.width)) != (n_rows, n_cols):
            raise ValueError(
                f"Overlap shape mismatch ref{(n_rows, n_cols)} vs "
                f"pred{(int(pred_window.height), int(pred_window.width))}"
            )
        if src_mask and (int(mask_window.height), int(mask_window.width)) != (n_rows, n_cols):
            raise ValueError("Overlap shape mismatch for valid mask.")

        for r0 in range(0, n_rows, block_rows):
            h = min(block_rows, n_rows - r0)

            rw = Window(ref_window.col_off, ref_window.row_off + r0, n_cols, h)
            pw = Window(pred_window.col_off, pred_window.row_off + r0, n_cols, h)
            reference = src_ref.read(1, window=rw)
            probability = src_pred.read(1, window=pw)

            external = None
            if src_mask:
                mw = Window(mask_window.col_off, mask_window.row_off + r0, n_cols, h)
                external = src_mask.read(1, window=mw)

            # --- identical valid mask for label AND prediction (loader lines 118-122)
            valid = np.ones(reference.shape, dtype=bool)
            if reference_nodata_values:
                not_nodata = ~np.isin(reference, reference_nodata_values)
                valid &= not_nodata
            if external is not None:
                ext_ok = external == valid_mask_value
                valid &= ext_ok

            diag["overlap_pixels"] += reference.size
            diag["valid_after_reference_nodata"] += (
                int(not_nodata.sum()) if reference_nodata_values else reference.size
            )
            diag["valid_after_external_mask"] += (
                int(ext_ok.sum()) if external is not None else reference.size
            )

            if not valid.any():
                continue

            y = np.isin(reference[valid], reference_water_values).astype(np.float64)
            p = probability[valid].astype(np.float64)
            # the SAME boolean `valid` indexes both arrays -> identical mask
            acc.update(p, y)
            diag["valid_pixels_after_all_masks"] += int(valid.sum())

        if src_mask:
            src_mask.close()

    log(f"    overlap={diag['overlap_pixels']:,}  "
        f"after_nodata={diag['valid_after_reference_nodata']:,}  "
        f"after_external_mask={diag['valid_after_external_mask']:,}  "
        f"valid_all_masks={diag['valid_pixels_after_all_masks']:,}")
    if acc.n != diag["valid_pixels_after_all_masks"]:
        diag["masking_consistent"] = False
        log("    WARNING: accumulated N != valid_pixels_after_all_masks")
    return acc, diag


# --------------------------------------------------------------------------- #
# Per-scene / pooled metrics + figures
# --------------------------------------------------------------------------- #

def metrics_from_acc(acc: FineAccumulator, coarse_bins: int):
    q_edges = _bin_edges_quantile(acc, coarse_bins)
    f_edges = _bin_edges_fixed(coarse_bins)
    q_tab = _coarse_table(acc, q_edges)
    f_tab = _coarse_table(acc, f_edges)
    ece_q, mce_q = _ece_mce(q_tab, acc.n)
    ece_f, mce_f = _ece_mce(f_tab, acc.n)
    metrics = {
        "n_valid": acc.n,
        "prevalence": acc.prevalence(),
        "brier": acc.brier(),
        "ece_quantile": ece_q,
        "mce_quantile": mce_q,
        "ece_fixed_width": ece_f,
        "mce_fixed_width": mce_f,
        "n_coarse_bins_quantile": int(len(q_edges) - 1),
        "n_coarse_bins_fixed_width": int(len(f_edges) - 1),
        "fine_bins": acc.fine_bins,
    }
    return metrics, q_tab, f_tab


def bins_csv(acc: FineAccumulator, q_tab: pd.DataFrame, f_tab: pd.DataFrame) -> pd.DataFrame:
    fine = _fine_table(acc)
    q = q_tab.assign(binning="quantile")
    f = f_tab.assign(binning="fixed_width")
    cols = ["binning", "bin_left", "bin_right", "count", "mean_pred", "obs_freq"]
    return pd.concat([fine[cols], q[cols], f[cols]], ignore_index=True)


def _fmt_count(n: int) -> str:
    """Compact human-readable pixel count, e.g. 1348463 -> '1.3M', 190495 -> '190k'."""
    n = int(n)
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.0f}k"
    return str(n)


def reliability_plot(q_tab: pd.DataFrame, metrics: dict, title: str, out_png: Path,
                     ax=None, annotate_counts: bool = True):
    own = ax is None
    if own:
        fig, ax = plt.subplots(figsize=(4.2, 4.2))
    t = q_tab[q_tab["count"] > 0]
    counts = t["count"].to_numpy()
    mean_pred = t["mean_pred"].to_numpy()
    obs_freq = t["obs_freq"].to_numpy()

    ax.plot([0, 1], [0, 1], ls="--", lw=1.0, color="#888888", label="perfect")
    # thin connecting line for the curve shape
    ax.plot(mean_pred, obs_freq, lw=1.2, color="#1f4e79", zorder=2)
    # marker AREA proportional to log10(bin pixel count) so a reader can see
    # whether a resolved point rests on millions of pixels or only thousands
    log_counts = np.log10(np.maximum(counts, 1))
    sizes = 12.0 + 70.0 * (log_counts - log_counts.min()) / max(np.ptp(log_counts), 1e-9)
    ax.scatter(mean_pred, obs_freq, s=sizes, color="#1f4e79", edgecolor="white",
               linewidth=0.5, zorder=3, label="quantile bins (area $\\propto$ log count)")

    if annotate_counts:
        # Declutter: where saturated quantile bins crowd (typically near p=0),
        # annotating every point overlaps illegibly. Skip a point's label if a
        # previously-labelled point is within `min_sep` in BOTH axes; the
        # marker-area cue still encodes count for the skipped points.
        min_sep = 0.045
        placed = []  # (x, y) of labels already drawn
        order = np.argsort(-counts)  # label the largest-count bins first
        for i in order:
            x, y, c = mean_pred[i], obs_freq[i], counts[i]
            if any(abs(x - px) < min_sep and abs(y - py) < min_sep for px, py in placed):
                continue
            ax.annotate(_fmt_count(c), (x, y), textcoords="offset points",
                        xytext=(5, 4), fontsize=6, color="#333333", zorder=4)
            placed.append((x, y))

    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_xlabel("mean predicted probability")
    ax.set_ylabel("observed water frequency")
    ax.set_title(title, fontsize=9)
    ax.text(0.04, 0.96,
            f"ECE={metrics['ece_quantile']:.4f}\nprev={metrics['prevalence']:.4f}\n"
            f"n={_fmt_count(metrics['n_valid'])}",
            transform=ax.transAxes, ha="left", va="top", fontsize=8,
            bbox=dict(boxstyle="round", fc="white", ec="#cccccc", alpha=0.8))
    ax.grid(alpha=0.3)
    if own:
        ax.legend(fontsize=6.5, loc="lower right")
        fig.tight_layout()
        fig.savefig(out_png, dpi=200)
        for ext_path in (out_png.with_suffix(".pdf"),):
            fig.savefig(ext_path)
        plt.close(fig)


def save_count_colorbar(norm: mcolors.Normalize, out_png: Path) -> None:
    """Save a standalone colorbar PNG/PDF for the log10(bin-count) color scale.

    Saved separately (not on the per-scene plots) so it can be placed manually in
    Illustrator. Tick labels are shown as raw pixel counts (10**log10).
    """
    fig, cax = plt.subplots(figsize=(1.1, 4.0))
    cb = matplotlib.colorbar.ColorbarBase(
        cax, cmap=plt.get_cmap(COUNT_CMAP), norm=norm, orientation="vertical")
    # Ticks span the ACTUAL data range (norm.vmin..vmax), not rounded decades, so
    # the colored band fills the bar. Endpoints plus any integer decades inside.
    decades = [d for d in range(int(np.ceil(norm.vmin)), int(np.floor(norm.vmax)) + 1)]
    ticks = sorted({norm.vmin, *decades, norm.vmax})
    cb.set_ticks(ticks)
    cb.set_ticklabels([_fmt_count(round(10 ** t)) for t in ticks])
    cb.set_label("bin pixel count (log scale)", fontsize=8)
    cax.tick_params(labelsize=7)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    fig.savefig(out_png.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def per_scene_reliability_plot(q_tab: pd.DataFrame, fine_tab: pd.DataFrame,
                               metrics: dict, title: str, out_png: Path,
                               norm: mcolors.Normalize) -> None:
    """Per-scene reliability diagram with:
      - points colored by log10(bin count) on a SHARED scale (no in-plot colorbar);
      - fixed marker size; no count labels on the main panel;
      - a small axis margin so points at the (0,0)/(1,1) corners are not clipped;
      - an inset zoom near the origin, with count annotations only inside it;
      - a bottom panel: histogram of the predicted-probability distribution
        (fine 1000-bin pixel counts, log-y), sharing the x-axis.
    """
    t = q_tab[q_tab["count"] > 0]
    counts = t["count"].to_numpy()
    mean_pred = t["mean_pred"].to_numpy()
    obs_freq = t["obs_freq"].to_numpy()
    log_counts = np.log10(np.maximum(counts, 1.0))
    cmap = plt.get_cmap(COUNT_CMAP)

    # margin so markers at the corners (p~0, p~1) sit fully inside the axes
    MARGIN = 0.035

    # wider figure: main column on the left, inset sits OUTSIDE to the right
    fig = plt.figure(figsize=(6.6, 5.6))
    gs = fig.add_gridspec(2, 2, width_ratios=[3.0, 1.25], height_ratios=[3.2, 1.0],
                          hspace=0.10, wspace=0.30)
    ax = fig.add_subplot(gs[0, 0])
    axh = fig.add_subplot(gs[1, 0], sharex=ax)  # predicted-prob histogram
    axins = fig.add_subplot(gs[0, 1])           # inset zoom, outside main panel

    # --- main reliability panel ---
    ax.plot([0, 1], [0, 1], ls="--", lw=1.0, color="#888888", zorder=1)
    ax.plot(mean_pred, obs_freq, lw=1.0, color="#aaaaaa", zorder=2)
    ax.scatter(mean_pred, obs_freq, c=log_counts, cmap=cmap, norm=norm,
               s=42, edgecolor="#333333", linewidth=0.5, zorder=3)
    ax.set_xlim(-MARGIN, 1 + MARGIN); ax.set_ylim(-MARGIN, 1 + MARGIN)
    ax.set_aspect("equal")
    ax.set_ylabel("observed water frequency")
    ax.tick_params(labelbottom=False)  # x labels live on the histogram panel below
    ax.set_title(title, fontsize=9)
    ax.text(0.04, 0.96,
            f"ECE={metrics['ece_quantile']:.4f}\nprev={metrics['prevalence']:.4f}\n"
            f"n={_fmt_count(metrics['n_valid'])}",
            transform=ax.transAxes, ha="left", va="top", fontsize=8,
            bbox=dict(boxstyle="round", fc="white", ec="#cccccc", alpha=0.8))
    ax.grid(alpha=0.3)

    # --- inset zoom near origin (OUTSIDE main panel): same points, counts
    #     annotated only inside the inset, with collision-aware decluttering ---
    near = (mean_pred <= 0.25) & (obs_freq <= 0.25)
    zmax = 0.25
    if near.sum() >= 2:
        zmax = float(max(mean_pred[near].max(), obs_freq[near].max())) * 1.25
        zmax = min(max(zmax, 0.02), 0.5)
    zpad = 0.06 * zmax  # margin inside the inset too
    axins.plot([0, zmax], [0, zmax], ls="--", lw=0.8, color="#888888")
    axins.plot(mean_pred, obs_freq, lw=0.8, color="#aaaaaa")
    axins.scatter(mean_pred, obs_freq, c=log_counts, cmap=cmap, norm=norm,
                  s=34, edgecolor="#333333", linewidth=0.4, zorder=3)
    inside = (mean_pred <= zmax) & (obs_freq <= zmax)
    xs, ys, cs = mean_pred[inside], obs_freq[inside], counts[inside]
    min_sep = 0.08 * zmax
    placed = []
    for i in np.argsort(-cs):  # largest-count bins labelled first
        x, y, c = xs[i], ys[i], cs[i]
        if any(abs(x - px) < min_sep and abs(y - py) < min_sep for px, py in placed):
            continue
        axins.annotate(_fmt_count(c), (x, y), textcoords="offset points",
                       xytext=(3, 2), fontsize=6, color="#222222", zorder=4)
        placed.append((x, y))
    axins.set_xlim(-zpad, zmax + zpad); axins.set_ylim(-zpad, zmax + zpad)
    axins.set_aspect("equal")
    axins.set_title("zoom @ origin", fontsize=7, pad=3)
    axins.set_xlabel("mean pred", fontsize=6)
    axins.tick_params(labelsize=5, length=2)

    # --- bottom panel: predicted-probability distribution (fine 1000-bin) ---
    fb_left = fine_tab["bin_left"].to_numpy()
    fb_right = fine_tab["bin_right"].to_numpy()
    fb_count = fine_tab["count"].to_numpy()
    fb_width = fb_right - fb_left
    axh.bar(fb_left, np.maximum(fb_count, 0), width=fb_width, align="edge",
            color="#4c72b0", edgecolor="none")
    axh.set_yscale("log")
    axh.set_xlim(-MARGIN, 1 + MARGIN)
    axh.set_xlabel("predicted probability")
    axh.set_ylabel("pixel count", fontsize=8)
    axh.grid(alpha=0.3, which="both")

    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    fig.savefig(out_png.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #

def scene_date(s1_id: str) -> str:
    parts = str(s1_id).split("_")
    if len(parts) >= 2 and len(parts[1]) == 8:
        d = parts[1]
        return f"{d[:4]}-{d[4:6]}-{d[6:]}"
    return str(s1_id)


def git_commit(repo_root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def run_model(model_name: str, df_model: pd.DataFrame, cfg: dict, out_root: Path,
              log) -> dict:
    model_out = out_root / model_name
    (model_out / "per_scene").mkdir(parents=True, exist_ok=True)
    (model_out / "pooled").mkdir(parents=True, exist_ok=True)

    df_model = df_model.sort_values("s1_id")
    log(f"\n=== model: {model_name}  ({len(df_model)} scenes) ===")

    manifest = {"model_name": model_name, "scenes": {}}
    pooled = FineAccumulator(cfg["fine_bins"])
    grid_entries = []        # (date, q_tab, metrics) for the grid figure
    per_scene_entries = []   # (date, q_tab, metrics) rendered later with shared color norm
    summary_rows = []

    for _, row in df_model.iterrows():
        date = scene_date(row["s1_id"])
        log(f"  scene {date} ({row['s1_id']})")
        acc, diag = stream_scene(
            reference_path=row["reference_path"],
            prediction_path=row["prediction_path"],
            valid_mask_path=cfg["valid_mask_path"],
            valid_mask_value=cfg["valid_mask_value"],
            reference_water_values=cfg["reference_water_values"],
            reference_nodata_values=cfg["reference_nodata_values"],
            resolution_atol=cfg["resolution_atol"],
            fine_bins=cfg["fine_bins"],
            block_rows=cfg["block_rows"],
            log=log,
        )
        metrics, q_tab, f_tab = metrics_from_acc(acc, cfg["coarse_bins"])
        metrics["binning_params"] = {
            "fine_bins": cfg["fine_bins"], "coarse_bins": cfg["coarse_bins"],
            "quantile_equal_mass": True, "fixed_width": True,
        }
        metrics["masking_consistent"] = diag["masking_consistent"]

        bc = bins_csv(acc, q_tab, f_tab)
        bc.to_csv(model_out / "per_scene" / f"{date}_bins.csv", index=False)
        with open(model_out / "per_scene" / f"{date}_metrics.json", "w") as fh:
            json.dump(metrics, fh, indent=2)
        # per-scene reliability PNG is rendered later (main), once the global
        # log-count color scale is known (shared across all models/scenes).
        # Cache the fine (1000-bin) table too, for the predicted-probability
        # distribution histogram on the bottom panel.
        per_scene_entries.append((date, q_tab, _fine_table(acc), metrics))

        manifest["scenes"][date] = {
            "s1_id": row["s1_id"],
            "reference_path": row["reference_path"],
            "prediction_path": row["prediction_path"],
            "n_valid": metrics["n_valid"],
            "prevalence": metrics["prevalence"],
        }
        summary_rows.append({
            "scene": date, "n_valid": metrics["n_valid"],
            "prevalence": metrics["prevalence"],
            "ece_quantile": metrics["ece_quantile"],
            "ece_fixed": metrics["ece_fixed_width"],
            "brier": metrics["brier"],
        })
        grid_entries.append((date, q_tab, metrics))
        pooled.merge(acc)

    # pooled (repeated measures of a single AOI)
    p_metrics, p_q_tab, p_f_tab = metrics_from_acc(pooled, cfg["coarse_bins"])
    p_metrics["aggregation"] = "repeated_measures_single_AOI_semarang"
    p_metrics["note"] = ("Pooled over six dates of the SAME AOI; summary only, "
                         "NOT an independent-sample estimate.")
    p_metrics["n_scenes"] = len(df_model)
    bins_csv(pooled, p_q_tab, p_f_tab).to_csv(
        model_out / "pooled" / "pooled_bins.csv", index=False)
    with open(model_out / "pooled" / "pooled_metrics.json", "w") as fh:
        json.dump(p_metrics, fh, indent=2)
    reliability_plot(p_q_tab, p_metrics, f"{model_name}  pooled (6 dates, single AOI)",
                     model_out / "pooled" / "pooled_reliability.png")

    # 2x3 grid A-F by ascending date
    fig, axes = plt.subplots(2, 3, figsize=(11.5, 8))
    panel = list("ABCDEF")
    for ax, lbl, (date, q_tab, metrics) in zip(axes.ravel(), panel, grid_entries):
        # marker size still encodes count; skip per-point text to avoid clutter at grid scale
        reliability_plot(q_tab, metrics, f"{lbl}  {date}", out_png=None, ax=ax,
                         annotate_counts=False)
    for ax in axes.ravel()[len(grid_entries):]:
        ax.set_axis_off()
    fig.suptitle(f"{model_name} — per-scene reliability (quantile bins)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(model_out / "reliability_grid.png", dpi=200)
    plt.close(fig)

    with open(model_out / "manifest.json", "w") as fh:
        json.dump(manifest, fh, indent=2)

    summary = pd.DataFrame(summary_rows)
    return {"model_name": model_name, "out_dir": str(model_out),
            "model_out": model_out, "per_scene_entries": per_scene_entries,
            "summary": summary, "pooled_metrics": p_metrics}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", type=Path, required=True,
                    help="Evaluation run dir containing evaluation_manifest.csv + run_metadata.json")
    ap.add_argument("--models", nargs="*", default=None,
                    help="Explicit model_name list; overrides --models-contains")
    ap.add_argument("--models-contains", default="native224",
                    help="Substring filter on model_name (default: native224)")
    ap.add_argument("--fine-bins", type=int, default=1000)
    ap.add_argument("--coarse-bins", type=int, default=15)
    ap.add_argument("--block-rows", type=int, default=512)
    args = ap.parse_args()

    run_dir = args.run_dir.resolve()
    manifest_csv = run_dir / "evaluation_manifest.csv"
    meta_json = run_dir / "run_metadata.json"
    for p in (manifest_csv, meta_json):
        if not p.exists():
            raise FileNotFoundError(p)

    meta = json.loads(meta_json.read_text())
    # config-driven reference water/nodata values (match the evaluation config)
    ref_water = [1]
    ref_nodata = [255]

    cfg = {
        "valid_mask_path": meta.get("valid_mask_path"),
        "valid_mask_value": int(meta.get("valid_mask_value", 1)),
        "reference_water_values": ref_water,
        "reference_nodata_values": ref_nodata,
        "resolution_atol": float(meta.get("resolution_atol", 1e-12)),
        "fine_bins": args.fine_bins,
        "coarse_bins": args.coarse_bins,
        "block_rows": args.block_rows,
    }

    out_root = run_dir / "calibration"
    out_root.mkdir(parents=True, exist_ok=True)
    log_path = out_root / "log.txt"
    log_fh = open(log_path, "w")

    def log(msg=""):
        print(msg)
        log_fh.write(str(msg) + "\n")
        log_fh.flush()

    log(f"calibration_check  {datetime.now(timezone.utc).isoformat()}")
    log(f"run_dir: {run_dir}")
    log(f"valid_mask_path: {cfg['valid_mask_path']}  value={cfg['valid_mask_value']}")
    log(f"reference_water_values={ref_water}  reference_nodata_values={ref_nodata}")
    log("Masking: per tile, valid = (~reference_nodata) & (external_mask == value), "
        "applied with the SAME boolean index to label AND prediction.")

    df = pd.read_csv(manifest_csv)
    df = df[df["status"] == "success"].copy()
    if args.models:
        df = df[df["model_name"].isin(args.models)]
    else:
        df = df[df["model_name"].str.contains(args.models_contains, na=False)]
    models = sorted(df["model_name"].unique())
    if not models:
        raise SystemExit("No matching models with success rows in manifest.")
    log(f"models ({len(models)}): {models}")

    # run_config.json
    run_config = {
        "script": str(Path(__file__).resolve().relative_to(_REPO_ROOT)),
        "git_commit": git_commit(_REPO_ROOT),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(run_dir),
        "input_manifest": str(manifest_csv),
        "output_root": str(out_root),
        "binning": {"fine_bins": args.fine_bins, "coarse_bins": args.coarse_bins,
                    "quantile_equal_mass": True, "fixed_width": True},
        "block_rows": args.block_rows,
        "valid_mask_path": cfg["valid_mask_path"],
        "valid_mask_value": cfg["valid_mask_value"],
        "reference_water_values": ref_water,
        "reference_nodata_values": ref_nodata,
        "models": models,
    }
    with open(out_root / "run_config.json", "w") as fh:
        json.dump(run_config, fh, indent=2)

    results = []
    for m in models:
        res = run_model(m, df[df["model_name"] == m], cfg, out_root, log)
        results.append(res)

    # --- shared (global) log10(bin-count) color scale across ALL models/scenes ---
    all_counts = []
    for res in results:
        for _date, q_tab, _fine_tab, _metrics in res["per_scene_entries"]:
            c = q_tab.loc[q_tab["count"] > 0, "count"].to_numpy()
            all_counts.append(c)
    all_counts = np.concatenate(all_counts) if all_counts else np.array([1])
    log_lo = float(np.log10(max(all_counts.min(), 1)))
    log_hi = float(np.log10(max(all_counts.max(), 1)))
    if log_hi <= log_lo:
        log_hi = log_lo + 1e-6
    count_norm = mcolors.Normalize(vmin=log_lo, vmax=log_hi)
    log(f"\nGlobal per-scene color scale: log10(count) in "
        f"[{log_lo:.3f}, {log_hi:.3f}] -> [{_fmt_count(10**log_lo)}, {_fmt_count(10**log_hi)}] px")

    # one standalone colorbar (for Illustrator); not embedded in the per-scene plots
    save_count_colorbar(count_norm, out_root / "colorbar.png")

    # render per-scene reliability diagrams with the shared scale
    for res in results:
        for date, q_tab, fine_tab, metrics in res["per_scene_entries"]:
            per_scene_reliability_plot(
                q_tab, fine_tab, metrics, f"{res['model_name']}  {date}",
                res["model_out"] / "per_scene" / f"{date}_reliability.png",
                count_norm)

    # final summary table
    log("\n================ SUMMARY ================")
    header = f"{'model':45s} {'scene':12s} {'n_valid':>12s} {'prev':>8s} {'ECE_q':>8s} {'ECE_fix':>8s} {'Brier':>9s}"
    log(header)
    for res in results:
        for _, r in res["summary"].iterrows():
            log(f"{res['model_name']:45s} {r['scene']:12s} {int(r['n_valid']):>12,} "
                f"{r['prevalence']:>8.4f} {r['ece_quantile']:>8.4f} "
                f"{r['ece_fixed']:>8.4f} {r['brier']:>9.5f}")
        pm = res["pooled_metrics"]
        log(f"{res['model_name']:45s} {'POOLED':12s} {int(pm['n_valid']):>12,} "
            f"{pm['prevalence']:>8.4f} {pm['ece_quantile']:>8.4f} "
            f"{pm['ece_fixed_width']:>8.4f} {pm['brier']:>9.5f}")
    log("=========================================")
    log(f"\nOutputs under: {out_root}")
    log_fh.close()


if __name__ == "__main__":
    main()
