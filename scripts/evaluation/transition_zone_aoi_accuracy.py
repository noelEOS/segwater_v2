#!/usr/bin/env python3
"""
S1-S2 concurrent accuracy for Swin-B / ConvNeXtV2-B / Swin-L / ConvNeXtV2-L,
restricted to the Demak transition-zone AOI polygon.

Same estimand and same masking as the multi-seed Semarang benchmark
(experiments/demak_semarang/s1_s2_concurrent_accuracy_multiseed), with ONE
addition: pixels are further restricted to a user-supplied AOI polygon.

Masking is layered, and the order matters:
  1. geospatial overlap of (reference, prediction, GSHHG valid mask)  -- via
     scripts/inference_overlap_utils.py, byte-for-byte the same code path the
     benchmark uses, so nothing about the reference decoding is re-implemented;
  2. reference nodata (255) excluded;
  3. external GSHHG/GSW combined valid mask == 1;
  4. NEW: AOI polygon rasterised onto the overlap window.

Reported at the NOMINAL 0.5 threshold (the out-of-the-box estimand, matching
artifacts_threshold050/), plus threshold-free ROC-AUC and average precision.

Uncertainty is reported on two axes that are NEVER pooled, per
artifacts_threshold050/NOTES.md:
  * across-seed SD (n=3 seeds, ddof=1) -- training-init variability;
  * paired spatial block bootstrap for model-vs-model contrasts, sharing block
    multiplicities across models so the (dominant, shared) spatial-sampling
    variance cancels. Marginal CIs are NOT used to compare models.

Usage:
    python scripts/evaluation/transition_zone_aoi_accuracy.py \
        --aoi experiments/demak_semarang/transition_zone_aoi/demak_transition_zone_aoi.gpkg \
        --out experiments/demak_semarang/transition_zone_aoi/accuracy
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
from rasterio.features import rasterize
from rasterio.windows import transform as window_transform
from sklearn.metrics import average_precision_score, roc_auc_score

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))

import geopandas as gpd  # noqa: E402

from evaluation.metrics import compute_binary_metrics  # noqa: E402
from evaluation.runsel import RunDirError  # noqa: E402
from evaluation.runsel import resolve_run_dir as _resolve_run_dir  # noqa: E402
from inference_overlap_utils import (  # noqa: E402
    intersection_bounds,
    read_profile,
    rounded_window_from_bounds,
    threshold_probability,
)

BENCH = REPO / "experiments/demak_semarang/s1_s2_concurrent_accuracy_multiseed"
RUNS = BENCH / "runs"

REFERENCE_DIR = Path(
    "/Volumes/noel_wd_black_sn850x/MACBOOK_AIR_M2_Backup/Downloads/"
    "Concurrent_S2_ndwi_mndwi_awei_ndvi"
)
VALID_MASK_PATH = Path(
    "/Users/noel/Documents/VisualStudio/projects/sen12coast_global_dl/INDONESIA/"
    "results/S1_S2_Accuracy_Assessment/GSHHG_GlobalSurfaceWater_combined_mask.tif"
)
VALID_MASK_VALUE = 1
REFERENCE_WATER_VALUES = [1]
REFERENCE_NODATA_VALUES = [255]
RESOLUTION_ATOL = 1.0e-12
THRESHOLD = 0.5
COMPARISON = "greater_than"

SEEDS = ["s19", "s42", "s58"]
STRIDE = "s32"

# display name -> run-dir model token
# Large variants (Swin-L, ConvNeXtV2-L) dropped from this eval 2026-07-17 (user
# decision) — reduced to the Swin-B vs ConvNeXtV2-B architecture match-up.
MODELS = {
    "Swin-B": "upernet_swin_base_224",
    "ConvNeXtV2-B": "upernet_tu-convnextv2_base",
}
MODEL_ORDER = list(MODELS)

# Paired contrast: the architecture match-up at equal (base) capacity.
CONTRASTS = [
    ("Swin-B", "ConvNeXtV2-B"),
]

BLOCK_PX = 200  # 2 km at 10 m -- matches the benchmark's >=2 km block choice
N_BOOT = 2000
BOOT_SEED = 42


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def resolve_run_dir(seed: str, model_token: str) -> Path:
    """Exactly one run dir for (seed, model token) at the module stride.

    The sweep name is fully known (`dev_sweep_all_demak_<seed>`), so anchor on
    it rather than a leading `*`: the anchored form additionally requires the
    UTC stamp, which stops a longer sweep name from being absorbed. `suffix`
    pins the arch token so `unet_resnet50` cannot match `unetplusplus_resnet50`.
    """
    suffix = f"_{model_token}_native224_weighted_224_b0_{STRIDE}"
    try:
        return _resolve_run_dir(RUNS, f"dev_sweep_all_demak_{seed}", suffix=suffix)
    except RunDirError as exc:
        raise SystemExit(str(exc)) from exc


def load_aoi_geoms(aoi_path: Path, target_crs: str):
    gdf = gpd.read_file(aoi_path)
    if gdf.crs is None:
        raise SystemExit("AOI has no CRS; refusing to guess.")
    gdf = gdf.to_crs(target_crs)
    return list(gdf.geometry)


def load_pair(
    reference_path: Path,
    prediction_path: Path,
    aoi_geoms,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Return (y_true, y_score, block_id, diagnostics) inside overlap AND AOI.

    Mirrors inference_overlap_utils._load_overlap_reference_and_probability and
    then intersects with the rasterised AOI. Block ids are assigned in the
    REFERENCE pixel frame, so a block is the same patch of ground in every scene
    and for every model -- the precondition for a paired bootstrap.
    """
    ref_profile = read_profile(str(reference_path))
    pred_profile = read_profile(str(prediction_path))
    mask_profile = read_profile(str(VALID_MASK_PATH))

    for other, label in ((pred_profile, "prediction"), (mask_profile, "valid_mask")):
        if ref_profile["crs"] != other["crs"]:
            raise SystemExit(f"CRS mismatch reference vs {label}: {ref_profile['crs']} vs {other['crs']}")
        if (
            abs(ref_profile["res_x"] - other["res_x"]) > RESOLUTION_ATOL
            or abs(ref_profile["res_y"] - other["res_y"]) > RESOLUTION_ATOL
        ):
            raise SystemExit(f"Resolution mismatch reference vs {label}")

    overlap = intersection_bounds(ref_profile["bounds"], pred_profile["bounds"], mask_profile["bounds"])

    with rasterio.open(reference_path) as src_ref, rasterio.open(prediction_path) as src_pred:
        ref_window = rounded_window_from_bounds(overlap, src_ref.transform)
        pred_window = rounded_window_from_bounds(overlap, src_pred.transform)
        reference = src_ref.read(1, window=ref_window)
        probability = src_pred.read(1, window=pred_window)
        win_transform = window_transform(ref_window, src_ref.transform)
        row_off, col_off = int(ref_window.row_off), int(ref_window.col_off)

    with rasterio.open(VALID_MASK_PATH) as src_mask:
        mask_window = rounded_window_from_bounds(overlap, src_mask.transform)
        valid_external = src_mask.read(1, window=mask_window)

    if not (reference.shape == probability.shape == valid_external.shape):
        raise SystemExit(
            f"Overlap window shape mismatch: ref={reference.shape} pred={probability.shape} mask={valid_external.shape}"
        )

    aoi_mask = rasterize(
        [(geom, 1) for geom in aoi_geoms],
        out_shape=reference.shape,
        transform=win_transform,
        fill=0,
        all_touched=False,
        dtype="uint8",
    ).astype(bool)

    valid = np.ones(reference.shape, dtype=bool)
    valid &= ~np.isin(reference, REFERENCE_NODATA_VALUES)
    valid &= valid_external == VALID_MASK_VALUE
    n_before_aoi = int(valid.sum())
    valid &= aoi_mask

    if valid.sum() == 0:
        raise SystemExit(f"No valid pixels inside AOI for {reference_path.name}")

    # Block ids in the ABSOLUTE reference pixel frame (window offset added back),
    # so ids are stable across scenes/models even if overlap windows differ.
    rows, cols = np.nonzero(valid)
    block_row = (rows + row_off) // BLOCK_PX
    block_col = (cols + col_off) // BLOCK_PX
    block_id = block_row.astype(np.int64) * 100_000 + block_col.astype(np.int64)

    y_true = np.isin(reference[valid], REFERENCE_WATER_VALUES).astype(np.uint8)
    y_score = probability[valid].astype(np.float32)

    diagnostics = {
        "overlap_shape": str(reference.shape),
        "valid_pixels_before_aoi": n_before_aoi,
        "valid_pixels_in_aoi": int(valid.sum()),
        "aoi_retained_fraction": float(valid.sum() / n_before_aoi) if n_before_aoi else 0.0,
        "reference_water_pixels": int(y_true.sum()),
        "reference_water_fraction": float(y_true.mean()),
        "n_blocks": int(np.unique(block_id).size),
    }
    return y_true, y_score, block_id, diagnostics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--aoi", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--n-boot", type=int, default=N_BOOT)
    args = parser.parse_args()

    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    reference_files = sorted(REFERENCE_DIR.glob("*veto_lt-02.tif"))
    if not reference_files:
        raise SystemExit(f"No reference rasters under {REFERENCE_DIR}")

    import re

    pairs = []
    for ref in reference_files:
        m = re.search(r"(S1_\d{8}_\d{6}_\d+_\d+_\d+)", ref.name)
        if m:
            pairs.append((m.group(1), ref))
    print(f"Found {len(pairs)} concurrent S1-S2 pairs")

    with rasterio.open(pairs[0][1]) as src:
        target_crs = src.crs.to_string()
    aoi_geoms = load_aoi_geoms(args.aoi, target_crs)
    print(f"AOI reprojected to {target_crs}: {len(aoi_geoms)} polygon(s)")

    per_scene_rows = []
    # scores[(model, seed)][s1_id] = (y_true, y_score, block_id)
    cache: dict[tuple[str, str], dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]] = {}

    for model_name, token in MODELS.items():
        for seed in SEEDS:
            run_dir = resolve_run_dir(seed, token)
            cache[(model_name, seed)] = {}
            for s1_id, ref_path in pairs:
                pred_path = run_dir / s1_id / f"{s1_id}_probability_water.tif"
                if not pred_path.exists():
                    raise SystemExit(f"Missing prediction: {pred_path}")
                y_true, y_score, block_id, diag = load_pair(ref_path, pred_path, aoi_geoms)
                cache[(model_name, seed)][s1_id] = (y_true, y_score, block_id)

                y_pred = threshold_probability(y_score, THRESHOLD, COMPARISON)
                metrics = compute_binary_metrics(y_true=y_true, y_pred=y_pred, include_counts=True)
                row = {
                    "model": model_name,
                    "seed": seed,
                    "stride": STRIDE,
                    "s1_id": s1_id,
                    "threshold": THRESHOLD,
                    **metrics,
                    "roc_auc": float(roc_auc_score(y_true, y_score)) if y_true.min() != y_true.max() else np.nan,
                    "average_precision": float(average_precision_score(y_true, y_score))
                    if y_true.min() != y_true.max()
                    else np.nan,
                    **diag,
                    "run_dir": str(run_dir),
                }
                per_scene_rows.append(row)
            print(f"  scored {model_name:14s} {seed}  ({len(pairs)} scenes)")

    df_scene = pd.DataFrame(per_scene_rows)
    df_scene.to_csv(out_dir / "per_scene_metrics.csv", index=False)

    # ---- per-seed model summary: macro-average across scenes (benchmark convention)
    metric_cols = ["iou", "f1", "precision", "recall", "oa", "mcc", "roc_auc", "average_precision"]
    df_seed = (
        df_scene.groupby(["model", "seed"], sort=False)[metric_cols]
        .mean()
        .reset_index()
        .assign(model=lambda d: pd.Categorical(d["model"], MODEL_ORDER, ordered=True))
        .sort_values(["model", "seed"])
    )
    df_seed.to_csv(out_dir / "per_seed_model_metrics.csv", index=False)

    # ---- across-seed summary: mean +/- seed SD (ddof=1, n=3)
    rows = []
    for model in MODEL_ORDER:
        sub = df_seed[df_seed["model"] == model]
        row = {"model": model, "n_seeds": int(len(sub))}
        for metric in metric_cols:
            row[f"{metric}_mean"] = float(sub[metric].mean())
            row[f"{metric}_seed_sd"] = float(sub[metric].std(ddof=1))
        rows.append(row)
    df_across = pd.DataFrame(rows)
    df_across.to_csv(out_dir / "across_seed_model_summary.csv", index=False)

    # ---- paired spatial block bootstrap on POOLED (scene-stacked) pixels.
    # One shared multiplicity vector per iteration, reused for every model, so
    # the shared spatial-sampling variance cancels in the A-B differences.
    all_blocks = sorted(
        set().union(
            *[
                set(np.unique(cache[(MODEL_ORDER[0], SEEDS[0])][s1_id][2]).tolist())
                for s1_id, _ in pairs
            ]
        )
    )
    block_index = {b: i for i, b in enumerate(all_blocks)}
    n_blocks = len(all_blocks)
    print(f"Paired block bootstrap: {n_blocks} blocks of {BLOCK_PX}px (~{BLOCK_PX * 10 / 1000:.0f} km), R={args.n_boot}")

    # Per (model, seed): block-level confusion counts at 0.5, pooled over scenes.
    # Shape (n_blocks, 4) columns tp, fp, fn, tn -- everything IoU needs.
    def block_counts(model: str, seed: str) -> np.ndarray:
        acc = np.zeros((n_blocks, 4), dtype=np.int64)
        for s1_id, _ in pairs:
            y_true, y_score, block_id = cache[(model, seed)][s1_id]
            y_pred = threshold_probability(y_score, THRESHOLD, COMPARISON)
            idx = np.array([block_index[b] for b in block_id], dtype=np.int64)
            t = y_true.astype(bool)
            p = y_pred.astype(bool)
            np.add.at(acc[:, 0], idx[t & p], 1)
            np.add.at(acc[:, 1], idx[~t & p], 1)
            np.add.at(acc[:, 2], idx[t & ~p], 1)
            np.add.at(acc[:, 3], idx[~t & ~p], 1)
        return acc

    counts = {(m, s): block_counts(m, s) for m in MODEL_ORDER for s in SEEDS}

    def iou_from(agg: np.ndarray) -> float:
        tp, fp, fn = agg[0], agg[1], agg[2]
        den = tp + fp + fn
        return float(tp / den) if den else np.nan

    def f1_from(agg: np.ndarray) -> float:
        tp, fp, fn = agg[0], agg[1], agg[2]
        den = 2 * tp + fp + fn
        return float(2 * tp / den) if den else np.nan

    rng = np.random.default_rng(BOOT_SEED)
    boot_iou = {key: np.empty(args.n_boot) for key in counts}
    boot_f1 = {key: np.empty(args.n_boot) for key in counts}
    for it in range(args.n_boot):
        mult = rng.multinomial(n_blocks, np.full(n_blocks, 1.0 / n_blocks))  # shared across ALL models
        for key, arr in counts.items():
            agg = mult @ arr
            boot_iou[key][it] = iou_from(agg)
            boot_f1[key][it] = f1_from(agg)

    # Pooled point estimates (all blocks, multiplicity 1)
    pooled_rows = []
    for model in MODEL_ORDER:
        for seed in SEEDS:
            agg = counts[(model, seed)].sum(axis=0)
            pooled_rows.append(
                {
                    "model": model,
                    "seed": seed,
                    "iou_pooled": iou_from(agg),
                    "f1_pooled": f1_from(agg),
                    "iou_ci_lower": float(np.percentile(boot_iou[(model, seed)], 2.5)),
                    "iou_ci_upper": float(np.percentile(boot_iou[(model, seed)], 97.5)),
                }
            )
    pd.DataFrame(pooled_rows).to_csv(out_dir / "pooled_metrics_with_marginal_ci.csv", index=False)

    # ---- contrasts: paired differences, per seed
    contrast_rows = []
    for a, b in CONTRASTS:
        for seed in SEEDS:
            d_iou = boot_iou[(a, seed)] - boot_iou[(b, seed)]
            d_f1 = boot_f1[(a, seed)] - boot_f1[(b, seed)]
            agg_a = counts[(a, seed)].sum(axis=0)
            agg_b = counts[(b, seed)].sum(axis=0)
            lo, hi = float(np.percentile(d_iou, 2.5)), float(np.percentile(d_iou, 97.5))
            contrast_rows.append(
                {
                    "model_a": a,
                    "model_b": b,
                    "seed": seed,
                    "d_iou_point": iou_from(agg_a) - iou_from(agg_b),
                    "d_iou_ci_lower": lo,
                    "d_iou_ci_upper": hi,
                    "d_iou_excludes_zero": bool(lo > 0 or hi < 0),
                    "d_f1_point": f1_from(agg_a) - f1_from(agg_b),
                    "d_f1_ci_lower": float(np.percentile(d_f1, 2.5)),
                    "d_f1_ci_upper": float(np.percentile(d_f1, 97.5)),
                    "prob_a_better": float((d_iou > 0).mean()),
                }
            )
    df_contrast = pd.DataFrame(contrast_rows)
    df_contrast.to_csv(out_dir / "contrasts_paired_bootstrap.csv", index=False)

    summary = (
        df_contrast.groupby(["model_a", "model_b"], sort=False)
        .agg(
            mean_d_iou=("d_iou_point", "mean"),
            min_d_iou=("d_iou_point", "min"),
            max_d_iou=("d_iou_point", "max"),
            n_seeds=("seed", "count"),
            n_seeds_ci_excludes_zero=("d_iou_excludes_zero", "sum"),
        )
        .reset_index()
    )
    summary.to_csv(out_dir / "contrasts_summary.csv", index=False)

    write_meta = {
        "generated_utc": utc_now(),
        "aoi_path": str(args.aoi),
        "aoi_crs_used": target_crs,
        "reference_dir": str(REFERENCE_DIR),
        "valid_mask_path": str(VALID_MASK_PATH),
        "threshold": THRESHOLD,
        "comparison": COMPARISON,
        "seeds": SEEDS,
        "stride": STRIDE,
        "models": {k: str(resolve_run_dir(SEEDS[0], v).name) for k, v in MODELS.items()},
        "n_pairs": len(pairs),
        "block_px": BLOCK_PX,
        "n_blocks": n_blocks,
        "n_boot": args.n_boot,
        "bootstrap_seed": BOOT_SEED,
        "valid_pixels_in_aoi_per_scene": int(df_scene["valid_pixels_in_aoi"].iloc[0]),
    }
    (out_dir / "run_metadata.json").write_text(json.dumps(write_meta, indent=2), encoding="utf-8")

    print("\n=== Across-seed summary (AOI, threshold 0.5, macro-avg over 6 scenes) ===")
    show = df_across[["model", "iou_mean", "iou_seed_sd", "f1_mean", "f1_seed_sd", "roc_auc_mean", "average_precision_mean"]]
    print(show.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("\n=== Paired contrasts (pooled IoU, shared block resamples) ===")
    print(summary.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\nWrote artifacts to {out_dir}")


if __name__ == "__main__":
    main()
