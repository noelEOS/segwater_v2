#!/usr/bin/env python3
"""
Spatial block bootstrap for SAR water-segmentation metrics over a single AOI.

This is the engine for honest confidence intervals on a single-AOI benchmark
(Semarang-Demak), where the naive "independent pixels" or "independent dates"
assumptions both overstate precision. The unit of resampling is a SPATIAL BLOCK
larger than the residual autocorrelation range, so the effective sample size is
the number of blocks B (a few dozen), not the millions of correlated pixels.

Pipeline (run as subcommands, or `all`):

  Step 0  variogram   Estimate the residual ( |p - y| ) autocorrelation range
                      from a spatial subsample, choose a primary block size that
                      comfortably exceeds it, and record 2-3 sensitivity sizes.

  Step 1  histograms  ONE streaming pass over pixels. For each (model, date,
                      block) accumulate a FINE histogram: per fine bin the pixel
                      count and positive-count (and sum_p). This is a sufficient
                      statistic for AP, ECE, and F1/IoU at any threshold, so no
                      pixel is ever touched again.

  Step 2-3 bootstrap  Resample B blocks with replacement (one shared multiplicity
                      vector m per iteration, used for EVERY model so model
                      comparisons are paired), aggregate the selected blocks'
                      histograms, recompute the metrics, and take percentile
                      (and BCa) CIs. Paired metric(A) - metric(B) differences are
                      stored per iteration for the 21 model contrasts.

  Step 4  report      Per-(model, date) metric + block CI (PRIMARY), a pooled
                      single-AOI summary (dates collapsed within block before
                      resampling, to preserve temporal correlation), across-date
                      variation reported descriptively, and a block-size
                      sensitivity table. B is stated explicitly.

Masking and pixel access REUSE scripts/inference_overlap_utils.py exactly: the
geospatial overlap of reference / prediction / valid-mask and the valid mask
(reference-nodata exclusion AND external mask == value) are identical to
scripts/evaluate_indonesia_inference_run_aucroc.py. Block ids are assigned in
the REFERENCE raster pixel frame, so a block is the same patch of ground in
every scene and for every model.

Usage
-----
    python scripts/analysis/spatial_block_bootstrap.py variogram \\
        --run-dir <.../semarang_probability_aucroc__20260617T092846Z>

    python scripts/analysis/spatial_block_bootstrap.py histograms \\
        --run-dir <...> --block-px <P> [--block-px <P2> ...]

    python scripts/analysis/spatial_block_bootstrap.py bootstrap \\
        --run-dir <...> --block-px <P> [--n-boot 2000] [--seed 42]

    python scripts/analysis/spatial_block_bootstrap.py all \\
        --run-dir <...>

Outputs go to <run-dir>/spatial_block_bootstrap/.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window

# --- repo imports: scripts/ holds inference_overlap_utils.py
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

# Manuscript models (native224_weighted family). The manifest also carries the
# large_crop_only variants; the benchmark/manuscript uses these.
#
# NOTE: Upernet_Swin_Large was only ever run at Demak/Semarang at stride s32 (the
# full-series sweeps built for the water-extent trend work). Runs that lack it --
# every s8 run -- simply score the other seven: the model list is intersected with
# each run's manifest in cmd_bootstrap, so a missing model is dropped, not an error.
MANUSCRIPT_MODELS = [
    "DeepLabV3plus_Resnet50_native224_weighted",
    "Unet_Resnet50_native224_weighted",
    "UnetPlusPlus_Resnet50_native224_weighted",
    "DPT_ViT_B_16_native224_weighted",
    "Segformer_MiT_B4_native224_weighted",
    "Upernet_ConvnextV2base_native224_weighted",
    "Upernet_ConvnextV2large_native224_weighted",
    "Upernet_Swin_Base_224_native224_weighted",
    "Upernet_Swin_Large_224_native224_weighted",
]

# Mean ground sample distance at the AOI latitude (~-6.9 deg) for the
# 8.983e-05 deg pixel: lon 8.983e-05*111320*cos(6.9deg) ~ 9.93 m,
# lat 8.983e-05*110574 ~ 9.93 m. Pixels are effectively square 10 m.
METRES_PER_PIXEL = 9.93


def scene_date(s1_id: str) -> str:
    parts = str(s1_id).split("_")
    if len(parts) >= 2 and len(parts[1]) == 8:
        d = parts[1]
        return f"{d[:4]}-{d[4:6]}-{d[6:]}"
    return str(s1_id)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


# --------------------------------------------------------------------------- #
# Shared scene loader: returns flat label/score AND the global pixel (row, col)
# of each valid pixel, in the REFERENCE raster frame, so block ids are shared
# across models and dates.
# --------------------------------------------------------------------------- #

def load_scene_with_coords(
    reference_path: str,
    prediction_path: str,
    valid_mask_path: str | None,
    valid_mask_value: int,
    reference_water_values: list[int],
    reference_nodata_values: list[int],
    resolution_atol: float,
    block_rows: int = 1024,
):
    """Stream one scene; yield (y, p, gr, gc) per row-block.

    gr, gc are GLOBAL pixel row/col in the reference raster frame (so identical
    across scenes for the same ground location). Masking mirrors
    _load_overlap_reference_and_probability exactly.
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
        if (int(pred_window.height), int(pred_window.width)) != (n_rows, n_cols):
            raise ValueError("Overlap shape mismatch ref vs pred.")
        if src_mask and (int(mask_window.height), int(mask_window.width)) != (n_rows, n_cols):
            raise ValueError("Overlap shape mismatch for valid mask.")

        # global offset of the overlap window in the reference pixel frame
        row_off = int(ref_window.row_off)
        col_off = int(ref_window.col_off)
        cols = np.arange(n_cols, dtype=np.int64)

        for r0 in range(0, n_rows, block_rows):
            h = min(block_rows, n_rows - r0)
            rw = Window(col_off, row_off + r0, n_cols, h)
            pw = Window(pred_window.col_off, pred_window.row_off + r0, n_cols, h)
            reference = src_ref.read(1, window=rw)
            probability = src_pred.read(1, window=pw)
            external = None
            if src_mask:
                mw = Window(mask_window.col_off, mask_window.row_off + r0, n_cols, h)
                external = src_mask.read(1, window=mw)

            valid = np.ones(reference.shape, dtype=bool)
            if reference_nodata_values:
                valid &= ~np.isin(reference, reference_nodata_values)
            if external is not None:
                valid &= external == valid_mask_value
            if not valid.any():
                continue

            y = np.isin(reference[valid], reference_water_values).astype(np.uint8)
            p = np.clip(probability[valid].astype(np.float32), 0.0, 1.0)
            # global row/col of each valid pixel in the reference frame
            rr, cc = np.nonzero(valid)
            gr = (rr + row_off + r0).astype(np.int64)
            gc = (cols[cc] + col_off).astype(np.int64)
            yield y, p, gr, gc

        if src_mask:
            src_mask.close()


def load_run_config(run_dir: Path) -> dict:
    meta = json.loads((run_dir / "run_metadata.json").read_text())
    return {
        "valid_mask_path": meta.get("valid_mask_path"),
        "valid_mask_value": int(meta.get("valid_mask_value", 1)),
        "reference_water_values": [1],
        "reference_nodata_values": [255],
        "resolution_atol": float(meta.get("resolution_atol", 1e-12)),
    }


def manifest_for_models(run_dir: Path, models: list[str]) -> pd.DataFrame:
    df = pd.read_csv(run_dir / "evaluation_manifest.csv")
    df = df[df["status"] == "success"].copy()
    df = df[df["model_name"].isin(models)]
    df["date"] = df["s1_id"].map(scene_date)
    return df.sort_values(["model_name", "s1_id"]).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Step 0: empirical variogram on residuals
# --------------------------------------------------------------------------- #

def cmd_variogram(args):
    run_dir = args.run_dir.resolve()
    out_dir = run_dir / "spatial_block_bootstrap"
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = load_run_config(run_dir)
    rng = np.random.default_rng(args.seed)

    model = args.variogram_model
    df = manifest_for_models(run_dir, [model])
    if df.empty:
        raise SystemExit(f"No success rows for variogram model {model}")

    # Spatially subsample valid pixels per date; pool residuals + coords.
    samples = []  # (gr, gc, resid, date_idx)
    per_date = []
    for di, (_, row) in enumerate(df.iterrows()):
        grs, gcs, res = [], [], []
        for y, p, gr, gc in load_scene_with_coords(
            row["reference_path"], row["prediction_path"],
            cfg["valid_mask_path"], cfg["valid_mask_value"],
            cfg["reference_water_values"], cfg["reference_nodata_values"],
            cfg["resolution_atol"],
        ):
            resid = np.abs(p - y.astype(np.float32))
            grs.append(gr); gcs.append(gc); res.append(resid)
        gr = np.concatenate(grs); gc = np.concatenate(gcs); resid = np.concatenate(res)
        n_take = min(args.subsample, gr.size)
        sel = rng.choice(gr.size, size=n_take, replace=False)
        samples.append((gr[sel], gc[sel], resid[sel], np.full(n_take, di)))
        per_date.append({"date": row["date"], "n_valid": int(gr.size), "n_sampled": int(n_take)})
        print(f"  {row['date']}: {gr.size:,} valid -> sampled {n_take:,}")

    gr = np.concatenate([s[0] for s in samples])
    gc = np.concatenate([s[1] for s in samples])
    resid = np.concatenate([s[2] for s in samples])
    didx = np.concatenate([s[3] for s in samples])

    # Empirical variogram by random pairs WITHIN the same date (so lag is purely
    # spatial). gamma(h) = 0.5 * mean( (z_i - z_j)^2 ) over pairs at lag ~ h.
    max_lag_px = args.max_lag_px
    n_pairs = args.n_pairs
    lag_edges_px = np.linspace(0, max_lag_px, args.n_lag_bins + 1)
    sq_sum = np.zeros(args.n_lag_bins)
    cnt = np.zeros(args.n_lag_bins, dtype=np.int64)

    for di in range(len(df)):
        m = didx == di
        idx = np.flatnonzero(m)
        if idx.size < 2:
            continue
        a = rng.choice(idx, size=n_pairs)
        b = rng.choice(idx, size=n_pairs)
        d_px = np.sqrt((gr[a] - gr[b]) ** 2 + (gc[a] - gc[b]) ** 2)
        within = d_px < max_lag_px
        a, b, d_px = a[within], b[within], d_px[within]
        bin_idx = np.minimum((d_px / max_lag_px * args.n_lag_bins).astype(int), args.n_lag_bins - 1)
        sq = (resid[a] - resid[b]) ** 2
        np.add.at(sq_sum, bin_idx, sq)
        np.add.at(cnt, bin_idx, 1)

    lag_mid_px = 0.5 * (lag_edges_px[:-1] + lag_edges_px[1:])
    gamma = np.where(cnt > 0, 0.5 * sq_sum / np.maximum(cnt, 1), np.nan)
    lag_mid_m = lag_mid_px * METRES_PER_PIXEL

    vg = pd.DataFrame({
        "lag_px": lag_mid_px,
        "lag_m": lag_mid_m,
        "n_pairs": cnt,
        "gamma": gamma,
    })
    vg.to_csv(out_dir / "variogram.csv", index=False)

    # Practical range estimate: lag at which gamma first reaches 95% of the sill
    # (sill = mean gamma over the far half of the lags).
    finite = vg.dropna()
    sill = float(finite[finite["lag_px"] > max_lag_px / 2]["gamma"].mean())
    thresh = 0.95 * sill
    reached = finite[finite["gamma"] >= thresh]
    range_px = float(reached["lag_px"].iloc[0]) if not reached.empty else float(max_lag_px)
    range_m = range_px * METRES_PER_PIXEL

    # Block-size recommendation: comfortably exceed the range (>= 3x), and a
    # sensitivity sweep around it.
    primary_px = int(np.ceil(3 * range_px / 100.0) * 100)
    sens_px = sorted({max(primary_px // 2, 100), primary_px, primary_px * 2})

    rec = {
        "timestamp_utc": utc_now(),
        "variogram_model": model,
        "metres_per_pixel": METRES_PER_PIXEL,
        "sill": sill,
        "range_px": range_px,
        "range_m": range_m,
        "range_threshold_frac_of_sill": 0.95,
        "primary_block_px": primary_px,
        "primary_block_m": primary_px * METRES_PER_PIXEL,
        "sensitivity_block_px": sens_px,
        "sensitivity_block_m": [s * METRES_PER_PIXEL for s in sens_px],
        "n_pairs_per_date": n_pairs,
        "subsample_per_date": args.subsample,
        "per_date": per_date,
        "note": (
            "Practical range = first lag at which the residual semivariance "
            "reaches 95% of the far-lag sill. Block side chosen >= 3x the range. "
            "EPSG:4326 grid; lags converted px->m at the AOI latitude."
        ),
    }
    (out_dir / "block_grid_recommendation.json").write_text(json.dumps(rec, indent=2))

    _plot_variogram(vg, sill, range_px, range_m, primary_px, model, out_dir / "variogram.png")

    print(f"\nSill ~ {sill:.4f}")
    print(f"Practical range ~ {range_px:.1f} px ({range_m:.0f} m)")
    print(f"Primary block: {primary_px} px ({primary_px * METRES_PER_PIXEL:.0f} m)")
    print(f"Sensitivity blocks (px): {sens_px}")
    print(f"\nWrote {out_dir / 'variogram.csv'}")
    print(f"Wrote {out_dir / 'block_grid_recommendation.json'}")
    print(f"Wrote {out_dir / 'variogram.png'}")


def _plot_variogram(vg, sill, range_px, range_m, primary_px, model, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    f = vg.dropna()
    ax.plot(f["lag_m"], f["gamma"], "o-", ms=3, lw=1.0, color="#1f4e79")
    ax.axhline(sill, ls=":", color="#888888", lw=1.0, label=f"sill ~ {sill:.3f}")
    ax.axvline(range_m, ls="--", color="#B05A43", lw=1.0,
               label=f"range ~ {range_m:.0f} m")
    ax.axvline(primary_px * METRES_PER_PIXEL, ls="-", color="#2a7", lw=1.0, alpha=0.6,
               label=f"block ~ {primary_px * METRES_PER_PIXEL:.0f} m")
    ax.set_xlabel("lag distance (m)")
    ax.set_ylabel(r"semivariance $\gamma(h)$  of  $|p-y|$")
    ax.set_title(f"Residual variogram — {model.replace('_', ' ')[:40]}", fontsize=8)
    ax.legend(fontsize=7, loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    fig.savefig(out_png.with_suffix(".pdf"))
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Step 1: per-(model, date, block) fine histograms (the only pixel pass)
# --------------------------------------------------------------------------- #

def _block_id(gr: np.ndarray, gc: np.ndarray, block_px: int) -> np.ndarray:
    """Global block id from reference-frame pixel coords. Shared across scenes."""
    br = gr // block_px
    bc = gc // block_px
    # encode (br, bc) -> single int; 100000 comfortably exceeds max block index
    return br.astype(np.int64) * 100_000 + bc.astype(np.int64)


def cmd_histograms(args):
    run_dir = args.run_dir.resolve()
    out_dir = run_dir / "spatial_block_bootstrap"
    cfg = load_run_config(run_dir)
    fine = args.fine_bins

    for block_px in args.block_px:
        hist_dir = out_dir / f"histograms_block{block_px}px"
        hist_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n=== histogram pass, block {block_px} px ({block_px*METRES_PER_PIXEL:.0f} m) ===")

        df = manifest_for_models(run_dir, args.models)
        dates = sorted(df["date"].unique())

        # Accumulate per (model,date) a dict block_id -> dense [fine] arrays,
        # using a single composite (block, fine_bin) bincount per row-block (no
        # Python loop over blocks). Then collect the global block set so every
        # (model,date) array shares the same block axis.
        per_md = {}   # (model, date) -> (block_ids_local, count[Bl,fine], pos, sum_p)
        global_blocks = set()

        for _, row in df.iterrows():
            # running per-scene composite accumulators keyed by local block id
            sc_count = {}  # block_id -> count[fine]
            sc_pos = {}
            sc_sump = {}
            for y, p, gr, gc in load_scene_with_coords(
                row["reference_path"], row["prediction_path"],
                cfg["valid_mask_path"], cfg["valid_mask_value"],
                cfg["reference_water_values"], cfg["reference_nodata_values"],
                cfg["resolution_atol"],
            ):
                bid = _block_id(gr, gc, block_px)
                fidx = np.minimum((p * fine).astype(np.int64), fine - 1)
                # map block ids to compact local indices for this row-block
                ub, inv = np.unique(bid, return_inverse=True)
                comp = inv.astype(np.int64) * fine + fidx     # (local_block, fine)
                size = len(ub) * fine
                c = np.bincount(comp, minlength=size).reshape(len(ub), fine)
                ps = np.bincount(comp, weights=y, minlength=size).reshape(len(ub), fine)
                sp = np.bincount(comp, weights=p, minlength=size).reshape(len(ub), fine)
                for j, b in enumerate(ub):
                    if b in sc_count:
                        sc_count[b] += c[j]; sc_pos[b] += ps[j].astype(np.int64); sc_sump[b] += sp[j]
                    else:
                        sc_count[b] = c[j].copy(); sc_pos[b] = ps[j].astype(np.int64); sc_sump[b] = sp[j].copy()

            bids_local = np.array(sorted(sc_count.keys()), dtype=np.int64)
            per_md[(row["model_name"], row["date"])] = (
                bids_local,
                np.stack([sc_count[b] for b in bids_local]) if bids_local.size else np.zeros((0, fine), np.int64),
                np.stack([sc_pos[b] for b in bids_local]) if bids_local.size else np.zeros((0, fine), np.int64),
                np.stack([sc_sump[b] for b in bids_local]) if bids_local.size else np.zeros((0, fine), np.float64),
            )
            global_blocks.update(bids_local.tolist())
            print(f"  {row['model_name'][:40]:40s} {row['date']}  blocks={len(bids_local)}")

        block_ids = np.array(sorted(global_blocks), dtype=np.int64)
        B = len(block_ids)
        bidx = {int(b): i for i, b in enumerate(block_ids)}
        np.save(hist_dir / "block_ids.npy", block_ids)

        # densify each (model,date) into [B x fine] count/pos/sum_p arrays
        for (model, date), (bids_local, c_arr, p_arr, s_arr) in per_md.items():
            count = np.zeros((B, fine), dtype=np.int64)
            pos = np.zeros((B, fine), dtype=np.int64)
            sum_p = np.zeros((B, fine), dtype=np.float64)
            rows_i = np.array([bidx[int(b)] for b in bids_local], dtype=np.int64)
            if rows_i.size:
                count[rows_i] = c_arr
                pos[rows_i] = p_arr
                sum_p[rows_i] = s_arr
            safe_model = model.replace("/", "_")
            np.savez_compressed(
                hist_dir / f"{safe_model}__{date}.npz",
                count=count, pos=pos, sum_p=sum_p,
            )

        meta = {
            "block_px": block_px,
            "block_m": block_px * METRES_PER_PIXEL,
            "fine_bins": fine,
            "B": B,
            "models": args.models,
            "dates": dates,
            "block_ids_file": "block_ids.npy",
            "timestamp_utc": utc_now(),
            "note": (
                "Per (model,date): [B x fine_bins] arrays of pixel count, "
                "positive-count, and sum of predicted probability. Block ids are "
                "shared across all (model,date) via the reference pixel frame. "
                "Sufficient statistic for AP, ECE, and F1/IoU at any threshold."
            ),
        }
        (hist_dir / "histograms_meta.json").write_text(json.dumps(meta, indent=2))
        print(f"  B = {B} blocks; wrote {hist_dir}")


# --------------------------------------------------------------------------- #
# Metrics computed from an aggregated fine histogram (count, pos, sum_p)
# --------------------------------------------------------------------------- #

def metrics_from_hist(count, pos, sum_p, fine, coarse_bins, thresholds):
    """All metrics from one fine histogram (1-D arrays of length fine_bins).

    Returns dict with AP, ECE (fixed-width), and F1/IoU at each fixed threshold.
    """
    N = count.sum()
    P = pos.sum()                       # total positives (water)
    out = {"n": int(N), "n_pos": int(P)}
    if N == 0 or P == 0:
        out["ap"] = np.nan
        out["ece"] = np.nan
        for t in thresholds:
            out[f"f1@{t:.2f}"] = np.nan
            out[f"iou@{t:.2f}"] = np.nan
        return out

    neg = count - pos                   # negatives per bin

    # --- AP (sklearn step rule): sweep thresholds high->low across bin boundaries
    # "predict positive" = bins at index >= k (probability in the upper bins).
    # cumulative from the top:
    cum_tp = np.cumsum(pos[::-1])[::-1]          # tp if threshold at left edge of bin k
    cum_fp = np.cumsum(neg[::-1])[::-1]
    tp_above = cum_tp                            # length fine
    fp_above = cum_fp
    with np.errstate(invalid="ignore", divide="ignore"):
        precision = np.where(tp_above + fp_above > 0, tp_above / (tp_above + fp_above), 1.0)
        recall = tp_above / P
    # AP = sum over thresholds of (R_k - R_{k+1}) * P_k, with recall increasing as
    # threshold drops (index decreases). Build monotone recall/precision arrays.
    # Order from highest threshold (smallest recall) to lowest:
    rec = recall          # recall[k] uses bins >= k; k=0 -> all bins -> recall=1
    prec = precision
    # AP step sum: sum_k (rec[k] - rec[k+1]) * prec[k] over k=0..fine-1 (k+1 -> 0 at end)
    rec_next = np.concatenate([rec[1:], [0.0]])
    ap = float(np.sum((rec - rec_next) * prec))
    out["ap"] = ap

    # --- F1 / IoU at each fixed threshold t (greater_than): bins with left edge >= t
    edges = np.linspace(0.0, 1.0, fine + 1)
    left = edges[:-1]
    for t in thresholds:
        above = left >= t            # greater_than semantics on bin left edge
        tp = pos[above].sum()
        fp = neg[above].sum()
        fn = P - tp
        denom_p = tp + fp
        denom_r = tp + fn
        prec_t = tp / denom_p if denom_p else 0.0
        rec_t = tp / denom_r if denom_r else 0.0
        f1 = (2 * prec_t * rec_t / (prec_t + rec_t)) if (prec_t + rec_t) else 0.0
        iou = tp / (tp + fp + fn) if (tp + fp + fn) else 0.0
        out[f"f1@{t:.2f}"] = float(f1)
        out[f"iou@{t:.2f}"] = float(iou)

    # --- ECE (fixed-width, coarse_bins equal-width bins) on the fine histogram
    cidx = np.minimum((left * coarse_bins).astype(np.int64), coarse_bins - 1)
    c_count = np.bincount(cidx, weights=count, minlength=coarse_bins)
    c_pos = np.bincount(cidx, weights=pos, minlength=coarse_bins)
    c_sump = np.bincount(cidx, weights=sum_p, minlength=coarse_bins)
    nz = c_count > 0
    mean_pred = np.where(nz, c_sump / np.maximum(c_count, 1), np.nan)
    obs_freq = np.where(nz, c_pos / np.maximum(c_count, 1), np.nan)
    gap = np.abs(obs_freq[nz] - mean_pred[nz])
    w = c_count[nz] / N
    out["ece"] = float(np.sum(w * gap))
    return out


# --------------------------------------------------------------------------- #
# Steps 2-3: spatial block bootstrap with shared resampling + paired contrasts
# --------------------------------------------------------------------------- #

def _percentile_ci(samples: np.ndarray) -> tuple[float, float]:
    s = samples[np.isfinite(samples)]
    if s.size == 0:
        return np.nan, np.nan
    return float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))


def _bca_ci(samples: np.ndarray, theta_hat: float, jackknife: np.ndarray) -> tuple[float, float]:
    """Bias-corrected and accelerated (BCa) 95% interval.

    samples: bootstrap replicates; theta_hat: full-data point estimate;
    jackknife: leave-one-block-out point estimates (for acceleration).
    Falls back to percentile if degenerate.
    """
    from scipy.stats import norm
    s = samples[np.isfinite(samples)]
    if s.size < 20:
        return _percentile_ci(samples)
    # bias correction z0
    prop = np.mean(s < theta_hat)
    if prop <= 0 or prop >= 1:
        return _percentile_ci(samples)
    z0 = norm.ppf(prop)
    # acceleration from jackknife
    jk = jackknife[np.isfinite(jackknife)]
    jbar = jk.mean()
    num = np.sum((jbar - jk) ** 3)
    den = 6.0 * (np.sum((jbar - jk) ** 2) ** 1.5)
    a = num / den if den != 0 else 0.0
    z_lo, z_hi = norm.ppf(0.025), norm.ppf(0.975)

    def adj(z):
        return norm.cdf(z0 + (z0 + z) / (1 - a * (z0 + z)))
    p_lo = adj(z_lo) * 100
    p_hi = adj(z_hi) * 100
    p_lo = min(max(p_lo, 0.1), 49.9)
    p_hi = min(max(p_hi, 50.1), 99.9)
    return float(np.percentile(s, p_lo)), float(np.percentile(s, p_hi))


def _load_hists(hist_dir: Path, models: list[str], dates: list[str], fine: int):
    """Return H[(model,date)] = (count[B,fine], pos, sum_p) and B."""
    H = {}
    B = None
    for model in models:
        safe = model.replace("/", "_")
        for date in dates:
            z = np.load(hist_dir / f"{safe}__{date}.npz")
            c, p, s = z["count"], z["pos"], z["sum_p"]
            if B is None:
                B = c.shape[0]
            H[(model, date)] = (c, p, s)
    return H, B


def _loo_thresholds(run_dir: Path, models: list[str], optimize_metric="iou"):
    """Per (model, date) LOO threshold and per-model mean LOO threshold."""
    d = pd.read_csv(run_dir / "loo_threshold_evaluation.csv")
    d = d[d["optimize_metric"] == optimize_metric]
    d["date"] = d["s1_id"].map(scene_date)
    per = {}
    mean_thr = {}
    for model in models:
        g = d[d["model_name"] == model]
        for _, r in g.iterrows():
            per[(model, r["date"])] = float(r["loo_selected_threshold"])
        if len(g):
            mean_thr[model] = float(g["loo_selected_threshold"].mean())
    return per, mean_thr


def cmd_bootstrap(args):
    run_dir = args.run_dir.resolve()
    out_dir = run_dir / "spatial_block_bootstrap"
    block_px = args.block_px
    hist_dir = out_dir / f"histograms_block{block_px}px"
    meta = json.loads((hist_dir / "histograms_meta.json").read_text())
    models = [m for m in args.models if (m, meta["dates"][0]) or True]
    models = [m for m in args.models if m in meta["models"]]
    dates = meta["dates"]
    fine = meta["fine_bins"]
    coarse = args.coarse_bins
    R = args.n_boot
    rng = np.random.default_rng(args.seed)

    H, B = _load_hists(hist_dir, models, dates, fine)
    loo_per, loo_mean = _loo_thresholds(run_dir, models, args.optimize_metric)

    # Per-date metric set: F1/IoU at 0.50 AND at the scene LOO threshold.
    # We round LOO thresholds to the fine-bin grid (2 dp) for the f1@/iou@ keys.
    def thr_keys_for(model, date):
        t_loo = round(loo_per.get((model, date), 0.5), 2)
        return [0.50, t_loo], t_loo

    # ---- point estimates (full data, all blocks) ----
    def agg_full(model, date):
        c, p, s = H[(model, date)]
        return c.sum(0), p.sum(0), s.sum(0)

    # ===================== PER-DATE bootstrap =====================
    # Store replicate matrices per (model, date) for AP, ECE, f1@0.5, iou@0.5,
    # f1@loo, iou@loo; plus paired diffs for AP/ECE between all model pairs.
    metric_names_perdate = ["ap", "ece", "f1@0.50", "iou@0.50", "f1@loo", "iou@loo"]
    # replicate storage: dict[(model,date)][metric] -> array(R)
    rep = {(m, d): {k: np.full(R, np.nan) for k in metric_names_perdate} for m in models for d in dates}
    # shared multiplicities per (date, r): same m used for every model at that date+iter
    print(f"Per-date bootstrap: R={R}, B={B}, models={len(models)}, dates={len(dates)}")
    for d in dates:
        for r in range(R):
            draw = rng.integers(0, B, size=B)
            m_vec = np.bincount(draw, minlength=B).astype(np.float64)  # multiplicities
            for model in models:
                c, p, s = H[(model, d)]
                cc = m_vec @ c
                pp = m_vec @ p
                ss = m_vec @ s
                thr_list, t_loo = thr_keys_for(model, d)
                res = metrics_from_hist(cc, pp, ss, fine, coarse, thr_list)
                rep[(model, d)]["ap"][r] = res["ap"]
                rep[(model, d)]["ece"][r] = res["ece"]
                rep[(model, d)]["f1@0.50"][r] = res["f1@0.50"]
                rep[(model, d)]["iou@0.50"][r] = res["iou@0.50"]
                rep[(model, d)]["f1@loo"][r] = res[f"f1@{t_loo:.2f}"]
                rep[(model, d)]["iou@loo"][r] = res[f"iou@{t_loo:.2f}"]
        print(f"  date {d} done")

    # point estimates + jackknife (leave-one-block-out) for BCa, per (model,date)
    perdate_rows = []
    for model in models:
        for d in dates:
            c, p, s = H[(model, d)]
            thr_list, t_loo = thr_keys_for(model, d)
            full = metrics_from_hist(c.sum(0), p.sum(0), s.sum(0), fine, coarse, thr_list)
            point = {
                "ap": full["ap"], "ece": full["ece"],
                "f1@0.50": full["f1@0.50"], "iou@0.50": full["iou@0.50"],
                "f1@loo": full[f"f1@{t_loo:.2f}"], "iou@loo": full[f"iou@{t_loo:.2f}"],
            }
            # jackknife
            jk = {k: np.full(B, np.nan) for k in metric_names_perdate}
            tot_c, tot_p, tot_s = c.sum(0), p.sum(0), s.sum(0)
            for b in range(B):
                jc, jp, js = tot_c - c[b], tot_p - p[b], tot_s - s[b]
                jm = metrics_from_hist(jc, jp, js, fine, coarse, thr_list)
                jk["ap"][b] = jm["ap"]; jk["ece"][b] = jm["ece"]
                jk["f1@0.50"][b] = jm["f1@0.50"]; jk["iou@0.50"][b] = jm["iou@0.50"]
                jk["f1@loo"][b] = jm[f"f1@{t_loo:.2f}"]; jk["iou@loo"][b] = jm[f"iou@{t_loo:.2f}"]
            for k in metric_names_perdate:
                lo_p, hi_p = _percentile_ci(rep[(model, d)][k])
                lo_b, hi_b = _bca_ci(rep[(model, d)][k], point[k], jk[k])
                perdate_rows.append({
                    "model_name": model, "date": d, "metric": k,
                    "point": point[k],
                    "boot_mean": float(np.nanmean(rep[(model, d)][k])),
                    "boot_std": float(np.nanstd(rep[(model, d)][k], ddof=1)),
                    "ci_lo_pct": lo_p, "ci_hi_pct": hi_p,
                    "ci_lo_bca": lo_b, "ci_hi_bca": hi_b,
                    "loo_threshold": round(t_loo, 2) if k.endswith("loo") else "",
                    "B": B, "R": R,
                })
    pd.DataFrame(perdate_rows).to_csv(out_dir / f"perdate_ci_block{block_px}px.csv", index=False)

    # Paired contrasts. iou@loo is the tuned operating point; iou@0.50 / f1@0.50 are
    # the nominal (untuned) operating point a deployment gets by default. Both are
    # carried through the SAME resamples, so the two contrasts are directly comparable.
    contrast_metrics = ["ap", "ece", "iou@loo", "iou@0.50", "f1@0.50"]
    pair_rows = []
    for d in dates:
        for i in range(len(models)):
            for j in range(i + 1, len(models)):
                A, Bm = models[i], models[j]
                for k in contrast_metrics:
                    diff = rep[(A, d)][k] - rep[(Bm, d)][k]
                    lo, hi = _percentile_ci(diff)
                    pair_rows.append({
                        "date": d, "model_A": A, "model_B": Bm, "metric": k,
                        "diff_mean": float(np.nanmean(diff)),
                        "diff_ci_lo": lo, "diff_ci_hi": hi,
                        "excludes_zero": bool((lo > 0) or (hi < 0)),
                        "B": B, "R": R,
                    })
    pd.DataFrame(pair_rows).to_csv(out_dir / f"perdate_contrasts_block{block_px}px.csv", index=False)

    # ===================== POOLED (single-AOI) bootstrap =====================
    # Collapse dates within block, then resample blocks (a resampled block
    # carries all six dates). Pooled threshold = per-model MEAN LOO threshold.
    Hpool = {}
    for model in models:
        c = sum(H[(model, d)][0] for d in dates)
        p = sum(H[(model, d)][1] for d in dates)
        s = sum(H[(model, d)][2] for d in dates)
        Hpool[model] = (c, p, s)

    pooled_metrics = ["ap", "ece", "f1@0.50", "iou@0.50", "f1@loo", "iou@loo"]
    rep_pool = {m: {k: np.full(R, np.nan) for k in pooled_metrics} for m in models}
    for r in range(R):
        draw = rng.integers(0, B, size=B)
        m_vec = np.bincount(draw, minlength=B).astype(np.float64)
        for model in models:
            c, p, s = Hpool[model]
            cc = m_vec @ c; pp = m_vec @ p; ss = m_vec @ s
            t_loo = round(loo_mean.get(model, 0.5), 2)
            res = metrics_from_hist(cc, pp, ss, fine, coarse, [0.50, t_loo])
            rep_pool[model]["ap"][r] = res["ap"]
            rep_pool[model]["ece"][r] = res["ece"]
            rep_pool[model]["f1@0.50"][r] = res["f1@0.50"]
            rep_pool[model]["iou@0.50"][r] = res["iou@0.50"]
            rep_pool[model]["f1@loo"][r] = res[f"f1@{t_loo:.2f}"]
            rep_pool[model]["iou@loo"][r] = res[f"iou@{t_loo:.2f}"]
    print("Pooled bootstrap done")

    pooled_rows = []
    for model in models:
        c, p, s = Hpool[model]
        t_loo = round(loo_mean.get(model, 0.5), 2)
        full = metrics_from_hist(c.sum(0), p.sum(0), s.sum(0), fine, coarse, [0.50, t_loo])
        point = {
            "ap": full["ap"], "ece": full["ece"],
            "f1@0.50": full["f1@0.50"], "iou@0.50": full["iou@0.50"],
            "f1@loo": full[f"f1@{t_loo:.2f}"], "iou@loo": full[f"iou@{t_loo:.2f}"],
        }
        jk = {k: np.full(B, np.nan) for k in pooled_metrics}
        tc, tp, ts = c.sum(0), p.sum(0), s.sum(0)
        for b in range(B):
            jm = metrics_from_hist(tc - c[b], tp - p[b], ts - s[b], fine, coarse, [0.50, t_loo])
            jk["ap"][b] = jm["ap"]; jk["ece"][b] = jm["ece"]
            jk["f1@0.50"][b] = jm["f1@0.50"]; jk["iou@0.50"][b] = jm["iou@0.50"]
            jk["f1@loo"][b] = jm[f"f1@{t_loo:.2f}"]; jk["iou@loo"][b] = jm[f"iou@{t_loo:.2f}"]
        for k in pooled_metrics:
            lo_p, hi_p = _percentile_ci(rep_pool[model][k])
            lo_b, hi_b = _bca_ci(rep_pool[model][k], point[k], jk[k])
            # across-date DESCRIPTIVE range (not a CI): point per date
            per_d_vals = []
            for d in dates:
                cd, pd_, sd = H[(model, d)]
                td = round(loo_per.get((model, d), 0.5), 2)
                fd = metrics_from_hist(cd.sum(0), pd_.sum(0), sd.sum(0), fine, coarse, [0.50, td])
                key = k if not k.endswith("loo") else (f"f1@{td:.2f}" if k.startswith("f1") else f"iou@{td:.2f}")
                per_d_vals.append(fd[key] if key in fd else np.nan)
            per_d_vals = np.array(per_d_vals, dtype=float)
            pooled_rows.append({
                "model_name": model, "metric": k, "scope": "pooled_single_AOI",
                "point": point[k],
                "boot_mean": float(np.nanmean(rep_pool[model][k])),
                "boot_std": float(np.nanstd(rep_pool[model][k], ddof=1)),
                "ci_lo_pct": lo_p, "ci_hi_pct": hi_p,
                "ci_lo_bca": lo_b, "ci_hi_bca": hi_b,
                "across_date_min": float(np.nanmin(per_d_vals)),
                "across_date_max": float(np.nanmax(per_d_vals)),
                "across_date_iqr": float(np.nanpercentile(per_d_vals, 75) - np.nanpercentile(per_d_vals, 25)),
                "pooled_loo_threshold": t_loo if k.endswith("loo") else "",
                "B": B, "R": R,
            })
    pd.DataFrame(pooled_rows).to_csv(out_dir / f"pooled_ci_block{block_px}px.csv", index=False)

    # pooled paired contrasts (shared m)
    pool_pair_rows = []
    for i in range(len(models)):
        for j in range(i + 1, len(models)):
            A, Bm = models[i], models[j]
            for k in contrast_metrics:
                diff = rep_pool[A][k] - rep_pool[Bm][k]
                lo, hi = _percentile_ci(diff)
                pool_pair_rows.append({
                    "model_A": A, "model_B": Bm, "metric": k,
                    "diff_mean": float(np.nanmean(diff)),
                    "diff_ci_lo": lo, "diff_ci_hi": hi,
                    "excludes_zero": bool((lo > 0) or (hi < 0)),
                    "B": B, "R": R,
                })
    pd.DataFrame(pool_pair_rows).to_csv(out_dir / f"pooled_contrasts_block{block_px}px.csv", index=False)

    summary = {
        "timestamp_utc": utc_now(),
        "block_px": block_px, "block_m": block_px * METRES_PER_PIXEL,
        "B": B, "R": R, "seed": args.seed,
        "models": models, "dates": dates,
        "fine_bins": fine, "coarse_bins_for_ece": coarse,
        "optimize_metric_for_loo": args.optimize_metric,
        "n_model_contrasts": len(models) * (len(models) - 1) // 2,
        "ci_methods": ["percentile", "BCa"],
        "outputs": {
            "perdate_ci": f"perdate_ci_block{block_px}px.csv",
            "perdate_contrasts": f"perdate_contrasts_block{block_px}px.csv",
            "pooled_ci": f"pooled_ci_block{block_px}px.csv",
            "pooled_contrasts": f"pooled_contrasts_block{block_px}px.csv",
        },
        "notes": [
            "Per-date CI is PRIMARY. Pooled CI is a labeled single-AOI summary "
            "(dates collapsed within block before resampling, preserving temporal "
            "correlation). Across-date variation is reported descriptively "
            "(min/max/IQR), NOT as a CI.",
            "Shared block multiplicities m are reused across all models within each "
            "iteration so paired model contrasts cancel shared spatial variance.",
            "F1/IoU reported at 0.50 and at the LOO-IoU threshold (per-scene for "
            "per-date; per-model mean for pooled).",
        ],
    }
    (out_dir / f"bootstrap_summary_block{block_px}px.json").write_text(json.dumps(summary, indent=2))
    print(f"\nWrote CIs + contrasts under {out_dir} (block {block_px}px)")


# --------------------------------------------------------------------------- #
# Step 4: report figures + markdown
# --------------------------------------------------------------------------- #

SHORT = {
    "DeepLabV3plus_Resnet50_native224_weighted": "DeepLabV3+",
    "Unet_Resnet50_native224_weighted": "U-Net",
    "UnetPlusPlus_Resnet50_native224_weighted": "U-Net++",
    "DPT_ViT_B_16_native224_weighted": "DPT-ViT-B",
    "Segformer_MiT_B4_native224_weighted": "SegFormer-B4",
    "Upernet_ConvnextV2base_native224_weighted": "ConvNeXtV2-B",
    "Upernet_ConvnextV2large_native224_weighted": "ConvNeXtV2-L",
    "Upernet_Swin_Base_224_native224_weighted": "Swin-B",
    "Upernet_Swin_Large_224_native224_weighted": "Swin-L",
}
ORDER = list(SHORT.keys())


def _forest(ax, df, metric, ci="pct", title=""):
    """Caterpillar/forest plot of pooled point + CI per model for one metric."""
    d = df[df["metric"] == metric].set_index("model_name").reindex(ORDER).dropna(subset=["point"])
    y = np.arange(len(d))[::-1]
    lo = d[f"ci_lo_{ci}"].to_numpy()
    hi = d[f"ci_hi_{ci}"].to_numpy()
    pt = d["point"].to_numpy()
    ax.hlines(y, lo, hi, color="#1f4e79", lw=2.0)
    ax.plot(pt, y, "o", color="#B05A43", ms=6, mec="#404040", mew=0.8, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels([SHORT[m] for m in d.index])
    ax.set_title(title, fontsize=9)
    ax.grid(axis="x", alpha=0.3)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def cmd_report(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.size": 8.5})

    run_dir = args.run_dir.resolve()
    out_dir = run_dir / "spatial_block_bootstrap"
    block_px = args.block_px
    pooled = pd.read_csv(out_dir / f"pooled_ci_block{block_px}px.csv")
    perdate = pd.read_csv(out_dir / f"perdate_ci_block{block_px}px.csv")
    pool_contr = pd.read_csv(out_dir / f"pooled_contrasts_block{block_px}px.csv")
    summary = json.loads((out_dir / f"bootstrap_summary_block{block_px}px.json").read_text())
    B, R = summary["B"], summary["R"]

    # ---- Figure 1: pooled forest plots, AP / ECE / IoU@LOO ----
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.4))
    _forest(axes[0], pooled, "ap", title="Average precision")
    axes[0].set_xlabel("AP")
    _forest(axes[1], pooled, "ece", title="Expected calibration error")
    axes[1].set_xlabel("ECE")
    _forest(axes[2], pooled, "iou@loo", title=r"IoU at LOO threshold")
    axes[2].set_xlabel("IoU")
    fig.suptitle(
        f"Single-AOI spatial-block bootstrap (Semarang-Demak): pooled point estimate "
        f"with 95% block CI  (B={B} blocks of {block_px*METRES_PER_PIXEL/1000:.1f} km, R={R})",
        fontsize=9.5)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out_dir / f"fig_pooled_forest_block{block_px}px.png", dpi=300)
    fig.savefig(out_dir / f"fig_pooled_forest_block{block_px}px.pdf")
    plt.close(fig)

    # ---- Figure 2: per-date AP with CI (small multiples by model) ----
    dates = summary["dates"]
    fig, ax = plt.subplots(figsize=(8.0, 4.2))
    x = np.arange(len(dates))
    offsets = np.linspace(-0.32, 0.32, len(ORDER))
    cmap = plt.get_cmap("tab10")
    for mi, model in enumerate(ORDER):
        d = perdate[(perdate["model_name"] == model) & (perdate["metric"] == "ap")]
        d = d.set_index("date").reindex(dates)
        pt = d["point"].to_numpy()
        lo = d["ci_lo_pct"].to_numpy(); hi = d["ci_hi_pct"].to_numpy()
        ax.errorbar(x + offsets[mi], pt, yerr=[pt - lo, hi - pt], fmt="o", ms=3.5,
                    lw=0.9, capsize=1.5, color=cmap(mi), label=SHORT[model])
    ax.set_xticks(x); ax.set_xticklabels(dates, rotation=30, ha="right")
    ax.set_ylabel("Average precision")
    ax.set_title(f"Per-date AP with 95% spatial-block CI  (B={B}, R={R})", fontsize=9)
    ax.legend(fontsize=7, ncol=4, loc="lower center", bbox_to_anchor=(0.5, -0.42))
    ax.grid(axis="y", alpha=0.3)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_dir / f"fig_perdate_ap_block{block_px}px.png", dpi=300, bbox_inches="tight")
    fig.savefig(out_dir / f"fig_perdate_ap_block{block_px}px.pdf", bbox_inches="tight")
    plt.close(fig)

    # ---- Figure 3: paired contrasts vs Swin (shared-m differences) ----
    # Generalized over metric. Expressed so a POSITIVE value always means
    # "Swin-B is better": Swin - other for AP/IoU (higher is better), and
    # other - Swin for ECE (lower is better, so Swin's deficit is positive).
    swin = "Upernet_Swin_Base_224_native224_weighted"

    def swin_contrast_fig(metric, label, stem, swin_better_is_higher=True):
        rows = []
        for _, r in pool_contr[pool_contr["metric"] == metric].iterrows():
            if r["model_A"] == swin or r["model_B"] == swin:
                other = r["model_B"] if r["model_A"] == swin else r["model_A"]
                # base sign expresses (Swin - other)
                base = 1.0 if r["model_A"] == swin else -1.0
                sign = base if swin_better_is_higher else -base
                rows.append({"other": other, "diff": sign * r["diff_mean"],
                             "lo": sign * r["diff_ci_lo"], "hi": sign * r["diff_ci_hi"]})
        dc = pd.DataFrame(rows).set_index("other").reindex([m for m in ORDER if m != swin]).dropna()
        lo = np.minimum(dc["lo"], dc["hi"]).to_numpy()
        hi = np.maximum(dc["lo"], dc["hi"]).to_numpy()
        pt = dc["diff"].to_numpy()
        y = np.arange(len(dc))[::-1]
        fig, ax = plt.subplots(figsize=(5.6, 3.4))
        ax.axvline(0, color="#888888", lw=1.0, ls="--")
        ax.hlines(y, lo, hi, color="#1f4e79", lw=2.0)
        ax.plot(pt, y, "o", color="#B05A43", ms=6, mec="#404040", mew=0.8, zorder=3)
        ax.set_yticks(y); ax.set_yticklabels([SHORT[m] for m in dc.index])
        ax.set_xlabel(label)
        ax.set_title(f"Swin-B advantage, 95% paired CI  (B={B}, R={R})", fontsize=9)
        ax.grid(axis="x", alpha=0.3)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        fig.tight_layout()
        fig.savefig(out_dir / f"{stem}_block{block_px}px.png", dpi=300)
        fig.savefig(out_dir / f"{stem}_block{block_px}px.pdf")
        plt.close(fig)

    swin_contrast_fig(
        "ap", r"$\Delta$AP  (Swin-B $-$ model),  paired block bootstrap",
        "fig_swin_ap_contrasts", swin_better_is_higher=True)
    swin_contrast_fig(
        "iou@loo", r"$\Delta$IoU@LOO  (Swin-B $-$ model),  paired block bootstrap",
        "fig_swin_iou_contrasts", swin_better_is_higher=True)

    print(f"Wrote 4 figures (PNG+PDF) under {out_dir}")


def build_arg_parser():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def add_common(p):
        p.add_argument("--run-dir", type=Path, required=True)
        p.add_argument("--models", nargs="*", default=MANUSCRIPT_MODELS)
        p.add_argument("--seed", type=int, default=42)

    pv = sub.add_parser("variogram", help="Step 0: residual variogram + block-grid recommendation")
    add_common(pv)
    pv.add_argument("--variogram-model", default="Upernet_Swin_Base_224_native224_weighted")
    pv.add_argument("--subsample", type=int, default=200_000,
                    help="valid pixels sampled per date for the variogram")
    pv.add_argument("--n-pairs", type=int, default=4_000_000,
                    help="random within-date pairs per date")
    pv.add_argument("--max-lag-px", type=int, default=300,
                    help="maximum lag (pixels) for the variogram (~3 km)")
    pv.add_argument("--n-lag-bins", type=int, default=40)
    pv.set_defaults(func=cmd_variogram)

    ph = sub.add_parser("histograms", help="Step 1: per-(model,date,block) fine histograms")
    add_common(ph)
    ph.add_argument("--block-px", type=int, nargs="+", default=[200],
                    help="block side(s) in pixels (primary 200 px ~ 2 km)")
    ph.add_argument("--fine-bins", type=int, default=1000)
    ph.set_defaults(func=cmd_histograms)

    pb = sub.add_parser("bootstrap", help="Steps 2-3: block bootstrap CIs + paired contrasts")
    add_common(pb)
    pb.add_argument("--block-px", type=int, default=200, help="block side used in Step 1")
    pb.add_argument("--n-boot", type=int, default=2000)
    pb.add_argument("--coarse-bins", type=int, default=15, help="equal-width bins for ECE")
    pb.add_argument("--optimize-metric", default="iou", help="LOO threshold metric")
    pb.set_defaults(func=cmd_bootstrap)

    pr = sub.add_parser("report", help="Step 4: manuscript figures from the CI/contrast CSVs")
    add_common(pr)
    pr.add_argument("--block-px", type=int, default=200)
    pr.set_defaults(func=cmd_report)

    return ap


def main():
    ap = build_arg_parser()
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
