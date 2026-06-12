"""
Backscatter analysis of shared CNN false negatives on SAR water mapping.

Why do the CNN-based segmentation models (DeepLabV3+, U-Net, U-Net++; all
ResNet-50) share systematic false negatives? For each tiling strategy
(native224, large_crop_1024) this script compares the Sentinel-1 VV/VH
backscatter (dB) distributions of three reference-WATER pixel groups:

  1. cnn_consensus_fn   - missed by ALL three CNN models simultaneously
  2. cnn_correct        - correctly classified by ALL three CNN models
  3. swin_fn_cnn_correct- missed by Swin-B but correct in all three CNNs
                          (Swin-B exists only as native224; it is used as the
                          transformer reference for both tiling groups)

Pixel groups are derived from the confusion rasters (TP=1 FP=2 FN=3 TN=4,
0=nodata, threshold 0.5) and restricted to the external valid mask, applied
exactly as in evaluate_indonesia_inference_run_benchmark.py /
inference_overlap_utils.py: the mask is windowed onto the confusion-raster
grid (same CRS/resolution, rounded window) and pixels != valid value are
excluded. S1 VV/VH are windowed onto the same grid.

Outputs (journal-ready) under --output-dir:
  backscatter_group_stats.csv        pooled stats per tiling/group/band
  backscatter_group_counts_per_scene.csv
  data_behind_figures.csv            stratified random subsample (VV,VH,group)
  <tiling>__kde2d_vv_vh.(png|pdf)    2D KDE density panels + overlay contours
  <tiling>__kde1d_marginals.(png|pdf) 1D KDE marginals for VV and VH
  README.md                          methods, group definitions, captions

Usage
-----
    python scripts/evaluation/backscatter_fn_analysis.py \
        --confusion-rasters-dir <dir> --s1-dir <dir> \
        --valid-mask-path mask.tif --output-dir <dir>
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import from_bounds as window_from_bounds
from scipy.stats import gaussian_kde

CM_TP, CM_FP, CM_FN, CM_TN, CM_NODATA = 1, 2, 3, 4, 0

CNN_MODELS = {
    "native224": [
        "DeepLabV3plus_Resnet50_native224_weighted",
        "Unet_Resnet50_native224_weighted",
        "UnetPlusPlus_Resnet50_native224_weighted",
    ],
    "large_crop_1024": [
        "DeepLabV3plus_Resnet50_large_crop_only_1024_b128_s1024",
        "Unet_Resnet50_large_crop_only_1024_b128_s1024",
        "UnetPlusPlus_Resnet50_large_crop_only_1024_b128_s1024",
    ],
}
SWIN_MODEL = "Upernet_Swin_Base_224_native224_weighted"

GROUPS = ["cnn_consensus_fn", "cnn_correct", "swin_fn_cnn_correct"]
GROUP_LABELS = {
    "cnn_consensus_fn": "FN in all CNNs",
    "cnn_correct": "correct in all CNNs",
    "swin_fn_cnn_correct": "FN in Swin-B, correct in CNNs",
}
GROUP_COLORS = {
    "cnn_consensus_fn": "#ca0020",
    "cnn_correct": "#0571b0",
    "swin_fn_cnn_correct": "#f4a582",
}

RNG = np.random.default_rng(42)
KDE_2D_MAX = 60_000
KDE_1D_MAX = 150_000
SUBSAMPLE_CSV_PER_GROUP = 20_000


def windowed_read(path: str, bounds, expected_shape, band: int = 1,
                  boundless: bool = False):
    with rasterio.open(path) as src:
        win = window_from_bounds(*bounds, transform=src.transform)
        win = win.round_offsets().round_lengths()
        arr = src.read(band, window=win, boundless=boundless,
                       fill_value=0 if boundless else None)
    if arr.shape != expected_shape:
        raise ValueError(f"{path} band {band}: window {arr.shape} != {expected_shape}")
    return arr


def read_cm(root: Path, model: str, scene: str):
    with rasterio.open(root / model / scene / "confusion_matrix.tif") as src:
        return src.read(1), src.bounds


def collect_pixels(root: Path, s1_dir: Path, mask_path: str, mask_value: int,
                   tiling: str) -> tuple[dict, pd.DataFrame]:
    """Returns ({group: {"vv": 1d, "vh": 1d}}, per-scene counts dataframe)."""
    scenes = sorted(p.name for p in (root / SWIN_MODEL).iterdir() if p.is_dir())
    pools = {g: {"vv": [], "vh": []} for g in GROUPS}
    count_rows = []

    for scene in scenes:
        cms = {}
        bounds = None
        for model in CNN_MODELS[tiling] + [SWIN_MODEL]:
            cms[model], bounds = read_cm(root, model, scene)
        shape = cms[SWIN_MODEL].shape

        # external valid mask, applied as in the benchmark evaluator
        mask = windowed_read(mask_path, bounds, shape, boundless=True)
        valid = mask == mask_value
        for model in cms:
            valid &= cms[model] != CM_NODATA

        ref_water = np.isin(cms[SWIN_MODEL], (CM_TP, CM_FN))  # reference water
        cnn_fn_all = np.ones(shape, dtype=bool)
        cnn_tp_all = np.ones(shape, dtype=bool)
        for model in CNN_MODELS[tiling]:
            cnn_fn_all &= cms[model] == CM_FN
            cnn_tp_all &= cms[model] == CM_TP
        swin_fn = cms[SWIN_MODEL] == CM_FN

        groups = {
            "cnn_consensus_fn": valid & ref_water & cnn_fn_all,
            "cnn_correct": valid & ref_water & cnn_tp_all,
            "swin_fn_cnn_correct": valid & ref_water & swin_fn & cnn_tp_all,
        }

        s1_path = s1_dir / f"{scene}.tif"
        vv = windowed_read(str(s1_path), bounds, shape, band=1)
        vh = windowed_read(str(s1_path), bounds, shape, band=2)
        finite = np.isfinite(vv) & np.isfinite(vh)

        row = {"tiling": tiling, "scene": scene}
        for g, sel in groups.items():
            sel = sel & finite
            pools[g]["vv"].append(vv[sel])
            pools[g]["vh"].append(vh[sel])
            row[g] = int(sel.sum())
        count_rows.append(row)
        print(f"  {scene}: " + ", ".join(f"{g}={row[g]}" for g in GROUPS))

    pooled = {g: {b: np.concatenate(pools[g][b]) for b in ("vv", "vh")}
              for g in GROUPS}
    return pooled, pd.DataFrame(count_rows)


def subsample(arr: np.ndarray, n: int) -> np.ndarray:
    if arr.size <= n:
        return arr
    return arr[RNG.choice(arr.size, size=n, replace=False)]


def stats_rows(pooled: dict, tiling: str) -> list[dict]:
    rows = []
    for g in GROUPS:
        for band in ("vv", "vh"):
            v = pooled[g][band]
            if v.size == 0:
                continue
            rows.append({
                "tiling": tiling, "group": g, "band": band.upper(),
                "n_pixels": int(v.size),
                "mean_db": float(v.mean()), "std_db": float(v.std()),
                "median_db": float(np.median(v)),
                "p5_db": float(np.percentile(v, 5)),
                "p25_db": float(np.percentile(v, 25)),
                "p75_db": float(np.percentile(v, 75)),
                "p95_db": float(np.percentile(v, 95)),
            })
    return rows


def axis_limits(pooled_all: list[dict]) -> tuple[tuple, tuple]:
    vv = np.concatenate([subsample(p[g]["vv"], 50_000)
                         for p in pooled_all for g in GROUPS if p[g]["vv"].size])
    vh = np.concatenate([subsample(p[g]["vh"], 50_000)
                         for p in pooled_all for g in GROUPS if p[g]["vh"].size])
    pad = 1.0
    return ((np.percentile(vv, 0.5) - pad, np.percentile(vv, 99.5) + pad),
            (np.percentile(vh, 0.5) - pad, np.percentile(vh, 99.5) + pad))


def plot_kde2d(pooled: dict, tiling: str, vv_lim, vh_lim, out_base: Path):
    gx, gy = np.mgrid[vv_lim[0]:vv_lim[1]:200j, vh_lim[0]:vh_lim[1]:200j]
    grid = np.vstack([gx.ravel(), gy.ravel()])
    densities = {}
    for g in GROUPS:
        vv = subsample(pooled[g]["vv"], KDE_2D_MAX)
        vh = subsample(pooled[g]["vh"], KDE_2D_MAX)
        if vv.size < 100:
            densities[g] = None
            continue
        kde = gaussian_kde(np.vstack([vv, vh]))
        densities[g] = kde(grid).reshape(gx.shape)

    fig, axes = plt.subplots(1, 4, figsize=(22, 5.2), sharex=True, sharey=True)
    for ax, g in zip(axes[:3], GROUPS):
        if densities[g] is None:
            ax.set_axis_off()
            continue
        ax.contourf(gx, gy, densities[g], levels=12, cmap="viridis")
        ax.set_title(f"{GROUP_LABELS[g]}\n(n={pooled[g]['vv'].size:,})", fontsize=11)
        ax.set_xlabel("VV [dB]")
    axes[0].set_ylabel("VH [dB]")
    for g in GROUPS:
        if densities[g] is None:
            continue
        d = densities[g]
        levels = np.percentile(d[d > 0], [60, 90])  # ~outer/inner mass contours
        axes[3].contour(gx, gy, d, levels=levels,
                        colors=GROUP_COLORS[g], linewidths=1.6)
        axes[3].plot([], [], color=GROUP_COLORS[g], label=GROUP_LABELS[g])
    axes[3].set_title("overlay (60th / 90th density percentiles)", fontsize=11)
    axes[3].set_xlabel("VV [dB]")
    axes[3].legend(fontsize=9, loc="upper left")
    fig.suptitle(f"S1 backscatter of reference-water pixels by CNN outcome - {tiling}",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    for ext in ("png", "pdf"):
        fig.savefig(f"{out_base}.{ext}", dpi=300)
    plt.close(fig)


def plot_kde1d(pooled: dict, tiling: str, vv_lim, vh_lim, out_base: Path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    for ax, band, lim in [(axes[0], "vv", vv_lim), (axes[1], "vh", vh_lim)]:
        xs = np.linspace(lim[0], lim[1], 512)
        for g in GROUPS:
            v = subsample(pooled[g][band], KDE_1D_MAX)
            if v.size < 100:
                continue
            ax.plot(xs, gaussian_kde(v)(xs), color=GROUP_COLORS[g],
                    lw=2, label=f"{GROUP_LABELS[g]} (n={pooled[g][band].size:,})")
            ax.axvline(np.median(v), color=GROUP_COLORS[g], lw=0.8, ls="--")
        ax.set_xlabel(f"{band.upper()} [dB]")
        ax.set_ylabel("density")
        ax.legend(fontsize=8)
    fig.suptitle(f"Marginal backscatter distributions - {tiling} "
                 "(dashed: medians)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    for ext in ("png", "pdf"):
        fig.savefig(f"{out_base}.{ext}", dpi=300)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--confusion-rasters-dir", required=True)
    p.add_argument("--s1-dir", required=True)
    p.add_argument("--valid-mask-path", required=True)
    p.add_argument("--valid-mask-value", type=int, default=1)
    p.add_argument("--output-dir", required=True)
    args = p.parse_args()

    root = Path(args.confusion_rasters_dir)
    s1_dir = Path(args.s1_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_stats, all_counts, pooled_by_tiling = [], [], {}
    for tiling in CNN_MODELS:
        print(f"=== {tiling} ===")
        pooled, counts = collect_pixels(
            root, s1_dir, args.valid_mask_path, args.valid_mask_value, tiling
        )
        pooled_by_tiling[tiling] = pooled
        all_counts.append(counts)
        all_stats.extend(stats_rows(pooled, tiling))

    pd.DataFrame(all_stats).to_csv(out_dir / "backscatter_group_stats.csv", index=False)
    pd.concat(all_counts, ignore_index=True).to_csv(
        out_dir / "backscatter_group_counts_per_scene.csv", index=False
    )

    vv_lim, vh_lim = axis_limits(list(pooled_by_tiling.values()))
    sample_rows = []
    for tiling, pooled in pooled_by_tiling.items():
        plot_kde2d(pooled, tiling, vv_lim, vh_lim, out_dir / f"{tiling}__kde2d_vv_vh")
        plot_kde1d(pooled, tiling, vv_lim, vh_lim, out_dir / f"{tiling}__kde1d_marginals")
        for g in GROUPS:
            n = min(SUBSAMPLE_CSV_PER_GROUP, pooled[g]["vv"].size)
            if n == 0:
                continue
            idx = RNG.choice(pooled[g]["vv"].size, size=n, replace=False)
            sample_rows.append(pd.DataFrame({
                "tiling": tiling, "group": g,
                "vv_db": pooled[g]["vv"][idx], "vh_db": pooled[g]["vh"][idx],
            }))
    pd.concat(sample_rows, ignore_index=True).to_csv(
        out_dir / "data_behind_figures.csv", index=False
    )

    print(f"\nOutputs in {out_dir}")
    print(pd.DataFrame(all_stats)[["tiling", "group", "band", "n_pixels",
                                   "median_db", "p25_db", "p75_db"]]
          .to_string(index=False))


if __name__ == "__main__":
    main()
