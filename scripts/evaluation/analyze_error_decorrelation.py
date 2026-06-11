"""
Analyze whether segmentation models make errors in the same places, using the
confusion rasters produced by generate_confusion_rasters.py.

Motivation: an ensemble can only fix a pixel if at least one member gets it
right. If all models fail on the same pixels, no combination rule helps.

For each tiling-strategy group (native224, large_crop_1024) the script computes:

1. Pairwise error overlap between models, pooled over scenes:
     - Jaccard of error masks (FP|FN), and separately for FP-only / FN-only
     - conditional overlap P(model B errs | model A errs)
2. Error multiplicity: for every pixel wrong in >=1 model, how many models
   are wrong there (split by FP / FN).
3. Ensemble potential:
     - majority-vote ensemble (strict majority, tie -> water) IoU/P/R
     - oracle upper bound (pixel correct if >=1 member correct)
     - greedy forward selection of ensemble members by pooled majority-vote IoU
4. Mechanism grouping (CNN / transformer / hybrid): mean pairwise error
   Jaccard within vs across mechanism groups.

Outputs CSVs, heatmap/bar figures and a markdown report under
<confusion_rasters_dir>/../error_decorrelation_analysis/.

Usage
-----
    python scripts/evaluation/analyze_error_decorrelation.py \
        path/to/confusion_rasters [--output-dir path]
"""

import argparse
import itertools
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import from_bounds as window_from_bounds

CM_TP, CM_FP, CM_FN, CM_TN, CM_NODATA = 1, 2, 3, 4, 0

# Directory name on disk -> canonical model name (fixes the mislabeled UperNet run)
DIR_RENAMES = {
    "Upernet_ConvnextV2base_native224_weighted_large_crop_only_1024_b128_s1024":
        "Upernet_ConvnextV2base_large_crop_only_1024_b128_s1024",
}

TILING_SUFFIXES = {
    "native224": "_native224_weighted",
    "large_crop_1024": "_large_crop_only_1024_b128_s1024",
}

MECHANISM = {
    "DeepLabV3plus_Resnet50": "CNN",
    "Unet_Resnet50": "CNN",
    "UnetPlusPlus_Resnet50": "CNN",
    "DPT_ViT_B_16": "Transformer",
    "Segformer_MiT_B4": "Transformer",
    "Upernet_Swin_Base_224": "Transformer",
    "Upernet_ConvnextV2base": "Hybrid",
}


def short_name(model: str) -> str:
    for suffix in TILING_SUFFIXES.values():
        if model.endswith(suffix):
            return model[: -len(suffix)]
    return model


def discover(root: Path) -> pd.DataFrame:
    rows = []
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        model = DIR_RENAMES.get(d.name, d.name)
        tiling = next(
            (t for t, suf in TILING_SUFFIXES.items() if model.endswith(suf)), None
        )
        if tiling is None:
            raise ValueError(f"Cannot infer tiling strategy for {model}")
        base = short_name(model)
        for scene_dir in sorted(p for p in d.iterdir() if p.is_dir()):
            rows.append(
                {
                    "model": model,
                    "base": base,
                    "mechanism": MECHANISM[base],
                    "tiling": tiling,
                    "scene": scene_dir.name,
                    "cm_path": scene_dir / "confusion_matrix.tif",
                }
            )
    return pd.DataFrame(rows)


def load_scene_stack(
    df_scene: pd.DataFrame,
    valid_mask_path: str | None = None,
    valid_mask_value: int = 1,
) -> tuple[list[str], np.ndarray]:
    """Returns (models, cm_stack[n_models, H, W]).

    If valid_mask_path is given, the external mask is windowed onto the
    confusion-raster grid (same CRS/resolution, as in
    inference_overlap_utils._load_overlap_reference_and_probability) and
    pixels where mask != valid_mask_value are set to CM_NODATA.
    """
    models, arrays = [], []
    bounds = transform = None
    for _, row in df_scene.iterrows():
        with rasterio.open(row["cm_path"]) as src:
            arrays.append(src.read(1))
            bounds, transform = src.bounds, src.transform
        models.append(row["base"])
    cm = np.stack(arrays)

    if valid_mask_path:
        with rasterio.open(valid_mask_path) as src:
            win = window_from_bounds(*bounds, transform=src.transform)
            win = win.round_offsets().round_lengths()
            mask = src.read(1, window=win, boundless=True, fill_value=0)
        if mask.shape != cm.shape[1:]:
            raise ValueError(
                f"Valid mask window shape {mask.shape} != raster shape {cm.shape[1:]}"
            )
        cm[:, mask != valid_mask_value] = CM_NODATA
    return models, cm


def metrics_from_counts(tp: int, fp: int, fn: int) -> dict:
    union = tp + fp + fn
    return {
        "iou": tp / union if union else float("nan"),
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def vote_metrics(pred_stack: np.ndarray, ref: np.ndarray, valid: np.ndarray,
                 members: list[int]) -> dict:
    """Majority vote over selected members (tie -> water)."""
    votes = pred_stack[members].sum(axis=0)
    ens = (2 * votes >= len(members)).astype(np.uint8) if len(members) % 2 == 0 \
        else (2 * votes > len(members)).astype(np.uint8)
    tp = int(np.sum(valid & (ens == 1) & (ref == 1)))
    fp = int(np.sum(valid & (ens == 1) & (ref == 0)))
    fn = int(np.sum(valid & (ens == 0) & (ref == 1)))
    return metrics_from_counts(tp, fp, fn)


def analyze_group(
    df: pd.DataFrame,
    tiling: str,
    out_dir: Path,
    valid_mask_path: str | None = None,
    valid_mask_value: int = 1,
) -> dict:
    sub = df[df["tiling"] == tiling]
    scenes = sorted(sub["scene"].unique())
    models = sorted(sub["base"].unique())
    n = len(models)
    idx = {m: i for i, m in enumerate(models)}

    # Pooled accumulators
    err_count = np.zeros(n, dtype=np.int64)            # |E_i|
    fp_count = np.zeros(n, dtype=np.int64)
    fn_count = np.zeros(n, dtype=np.int64)
    inter = np.zeros((n, n), dtype=np.int64)           # |E_i & E_j|
    inter_fp = np.zeros((n, n), dtype=np.int64)
    inter_fn = np.zeros((n, n), dtype=np.int64)
    mult_hist = np.zeros(n + 1, dtype=np.int64)        # error multiplicity
    mult_hist_fp = np.zeros(n + 1, dtype=np.int64)
    mult_hist_fn = np.zeros(n + 1, dtype=np.int64)
    valid_total = 0

    # Pooled confusion counts for singles / ensembles
    single_counts = {m: np.zeros(3, dtype=np.int64) for m in models}  # tp,fp,fn

    # Per-scene cached stacks for ensemble evaluation
    scene_data = []  # (pred_stack, ref, valid)

    for scene in scenes:
        df_scene = sub[sub["scene"] == scene].sort_values("base")
        scene_models, cm = load_scene_stack(df_scene, valid_mask_path, valid_mask_value)
        assert scene_models == models
        valid = cm[0] != CM_NODATA
        # sanity: valid masks identical across models (same reference)
        assert all(np.array_equal(valid, cm[k] != CM_NODATA) for k in range(1, n))
        valid_total += int(valid.sum())

        pred = ((cm == CM_TP) | (cm == CM_FP)).astype(np.uint8)
        ref = ((cm[0] == CM_TP) | (cm[0] == CM_FN)).astype(np.uint8)
        err = (cm == CM_FP) | (cm == CM_FN)
        fp_m = cm == CM_FP
        fn_m = cm == CM_FN

        err_count += err.reshape(n, -1).sum(axis=1)
        fp_count += fp_m.reshape(n, -1).sum(axis=1)
        fn_count += fn_m.reshape(n, -1).sum(axis=1)

        ef = err.reshape(n, -1).astype(np.float32)
        inter += (ef @ ef.T).astype(np.int64)
        ff = fp_m.reshape(n, -1).astype(np.float32)
        inter_fp += (ff @ ff.T).astype(np.int64)
        nf = fn_m.reshape(n, -1).astype(np.float32)
        inter_fn += (nf @ nf.T).astype(np.int64)

        mult = err.sum(axis=0)
        mult_hist += np.bincount(mult.ravel(), minlength=n + 1)
        mult_hist_fp += np.bincount(fp_m.sum(axis=0).ravel(), minlength=n + 1)
        mult_hist_fn += np.bincount(fn_m.sum(axis=0).ravel(), minlength=n + 1)

        for k, m in enumerate(models):
            single_counts[m] += np.array(
                [int((cm[k] == CM_TP).sum()), int(fp_m[k].sum()), int(fn_m[k].sum())]
            )

        scene_data.append((pred, ref.astype(np.uint8), valid))

    # ---- pairwise matrices
    union = err_count[:, None] + err_count[None, :] - inter
    jaccard = np.where(union > 0, inter / np.maximum(union, 1), np.nan)
    cond = inter / np.maximum(err_count[None, :], 1)  # P(row errs | col errs)
    union_fp = fp_count[:, None] + fp_count[None, :] - inter_fp
    jaccard_fp = np.where(union_fp > 0, inter_fp / np.maximum(union_fp, 1), np.nan)
    union_fn = fn_count[:, None] + fn_count[None, :] - inter_fn
    jaccard_fn = np.where(union_fn > 0, inter_fn / np.maximum(union_fn, 1), np.nan)

    for name, mat in [
        ("error_jaccard", jaccard),
        ("error_jaccard_fp", jaccard_fp),
        ("error_jaccard_fn", jaccard_fn),
        ("error_conditional_row_given_col", cond),
    ]:
        pd.DataFrame(mat, index=models, columns=models).to_csv(
            out_dir / f"{tiling}__{name}.csv"
        )

    # ---- multiplicity
    mult_df = pd.DataFrame(
        {
            "n_models_wrong": np.arange(n + 1),
            "pixels_any_error": mult_hist,
            "pixels_fp": mult_hist_fp,
            "pixels_fn": mult_hist_fn,
        }
    )
    mult_df.to_csv(out_dir / f"{tiling}__error_multiplicity.csv", index=False)

    # ---- single-model pooled metrics
    singles = pd.DataFrame(
        [
            {"model": m, "mechanism": MECHANISM[m],
             **metrics_from_counts(*single_counts[m])}
            for m in models
        ]
    ).sort_values("iou", ascending=False)
    singles.to_csv(out_dir / f"{tiling}__single_model_pooled_metrics.csv", index=False)
    best_single = singles.iloc[0]

    # ---- ensemble evaluation helpers (pooled over scenes)
    def pooled_vote(members_idx: list[int]) -> dict:
        tp = fp = fn = 0
        for pred, ref, valid in scene_data:
            m = vote_metrics(pred, ref, valid, members_idx)
            tp, fp, fn = tp + m["tp"], fp + m["fp"], fn + m["fn"]
        return metrics_from_counts(tp, fp, fn)

    def pooled_oracle(members_idx: list[int]) -> dict:
        """Pixel correct if >=1 member correct (upper bound)."""
        tp = fp = fn = 0
        for pred, ref, valid in scene_data:
            correct = (pred[members_idx] == ref[None]).any(axis=0)
            tp += int(np.sum(valid & correct & (ref == 1)))
            fp += int(np.sum(valid & ~correct & (ref == 0)))
            fn += int(np.sum(valid & ~correct & (ref == 1)))
        return metrics_from_counts(tp, fp, fn)

    ens_rows = []
    all_idx = list(range(n))
    ens_rows.append({"ensemble": "ALL_majority_vote", "k": n, **pooled_vote(all_idx)})
    ens_rows.append({"ensemble": "ALL_oracle_upper_bound", "k": n,
                     **pooled_oracle(all_idx)})

    # mechanism sub-ensembles
    for mech in sorted({MECHANISM[m] for m in models}):
        mem = [idx[m] for m in models if MECHANISM[m] == mech]
        if len(mem) >= 2:
            ens_rows.append(
                {"ensemble": f"{mech}_majority_vote", "k": len(mem),
                 **pooled_vote(mem)}
            )

    # greedy forward selection
    selected = [idx[best_single["model"]]]
    greedy_log = [{"step": 1, "added": best_single["model"],
                   **pooled_vote(selected)}]
    remaining = [i for i in all_idx if i not in selected]
    while remaining:
        scored = [(pooled_vote(selected + [c])["iou"], c) for c in remaining]
        best_iou, best_c = max(scored)
        selected.append(best_c)
        remaining.remove(best_c)
        greedy_log.append(
            {"step": len(selected), "added": models[best_c],
             **pooled_vote(selected)}
        )
    greedy_df = pd.DataFrame(greedy_log)
    greedy_df.to_csv(out_dir / f"{tiling}__greedy_ensemble_selection.csv", index=False)

    best_step = greedy_df.loc[greedy_df["iou"].idxmax()]
    best_members = [r["added"] for r in greedy_log[: int(best_step["step"])]]
    ens_rows.append(
        {"ensemble": f"GREEDY_BEST[{','.join(best_members)}]",
         "k": int(best_step["step"]),
         **{k: best_step[k] for k in ["iou", "precision", "recall", "tp", "fp", "fn"]}}
    )
    ens_df = pd.DataFrame(ens_rows)
    ens_df.to_csv(out_dir / f"{tiling}__ensemble_metrics.csv", index=False)

    # ---- mechanism-level within/cross Jaccard
    mech_rows = []
    for mi, mj in itertools.combinations_with_replacement(
        sorted({MECHANISM[m] for m in models}), 2
    ):
        vals = [
            jaccard[idx[a], idx[b]]
            for a in models for b in models
            if a < b and {MECHANISM[a], MECHANISM[b]} == ({mi, mj} if mi != mj else {mi})
        ]
        if vals:
            mech_rows.append(
                {"pair": f"{mi}-{mj}", "mean_error_jaccard": float(np.mean(vals)),
                 "n_pairs": len(vals)}
            )
    mech_df = pd.DataFrame(mech_rows)
    mech_df.to_csv(out_dir / f"{tiling}__mechanism_error_jaccard.csv", index=False)

    # ---- figures
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    for ax, (mat, title) in zip(
        axes,
        [(jaccard, "Error Jaccard (FP|FN)"), (jaccard_fp, "FP Jaccard"),
         (jaccard_fn, "FN Jaccard")],
    ):
        im = ax.imshow(mat, vmin=0, vmax=1, cmap="viridis")
        ax.set_xticks(range(n), models, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(n), models, fontsize=8)
        ax.set_title(f"{tiling}: {title}")
        for i in range(n):
            for j in range(n):
                if not np.isnan(mat[i, j]):
                    ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center",
                            color="w" if mat[i, j] < 0.6 else "k", fontsize=7)
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(out_dir / f"{tiling}__pairwise_error_jaccard.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(1, n + 1)
    tot = mult_hist[1:].sum()
    ax.bar(x - 0.25, mult_hist[1:] / tot, 0.25, label="any error")
    ax.bar(x, mult_hist_fp[1:] / max(mult_hist_fp[1:].sum(), 1), 0.25, label="FP")
    ax.bar(x + 0.25, mult_hist_fn[1:] / max(mult_hist_fn[1:].sum(), 1), 0.25,
           label="FN")
    ax.set_xlabel("number of models wrong at the pixel")
    ax.set_ylabel("fraction of error pixels")
    ax.set_title(f"{tiling}: error multiplicity ({n} models)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / f"{tiling}__error_multiplicity.png", dpi=150)
    plt.close(fig)

    frac_all_wrong = mult_hist[n] / max(mult_hist[1:].sum(), 1)
    frac_recoverable = 1.0 - frac_all_wrong
    return {
        "tiling": tiling,
        "models": models,
        "valid_pixels_pooled": valid_total,
        "best_single": best_single.to_dict(),
        "ensembles": ens_df.to_dict("records"),
        "greedy": greedy_df.to_dict("records"),
        "mechanism_jaccard": mech_df.to_dict("records"),
        "multiplicity": mult_df.to_dict("records"),
        "frac_errors_shared_by_all": float(frac_all_wrong),
        "frac_errors_recoverable": float(frac_recoverable),
        "mean_offdiag_jaccard": float(
            np.nanmean(jaccard[~np.eye(n, dtype=bool)])
        ),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("confusion_rasters_dir")
    p.add_argument("--output-dir", default=None)
    p.add_argument(
        "--valid-mask-path",
        default=None,
        help="External valid-mask GeoTIFF (same CRS/grid); pixels != "
             "--valid-mask-value are excluded, as in the benchmark evaluator",
    )
    p.add_argument("--valid-mask-value", type=int, default=1)
    args = p.parse_args()

    root = Path(args.confusion_rasters_dir)
    default_name = "error_decorrelation_analysis_masked" if args.valid_mask_path \
        else "error_decorrelation_analysis"
    out_dir = Path(args.output_dir) if args.output_dir else root.parent / default_name
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.valid_mask_path:
        print(f"External valid mask: {args.valid_mask_path} "
              f"(valid value: {args.valid_mask_value})")

    df = discover(root)
    print(f"Found {df['model'].nunique()} models, {df['scene'].nunique()} scenes")

    results = {}
    for tiling in TILING_SUFFIXES:
        if (df["tiling"] == tiling).any():
            print(f"\n=== {tiling} ===")
            results[tiling] = analyze_group(
                df, tiling, out_dir, args.valid_mask_path, args.valid_mask_value
            )
            r = results[tiling]
            print(f"  models: {r['models']}")
            print(f"  mean off-diagonal error Jaccard: "
                  f"{r['mean_offdiag_jaccard']:.3f}")
            print(f"  errors shared by ALL models: "
                  f"{r['frac_errors_shared_by_all']:.1%} "
                  f"(recoverable: {r['frac_errors_recoverable']:.1%})")
            print(f"  best single IoU: {r['best_single']['iou']:.4f} "
                  f"({r['best_single']['model']})")
            for e in r["ensembles"]:
                print(f"  {e['ensemble']}: IoU={e['iou']:.4f} "
                      f"P={e['precision']:.4f} R={e['recall']:.4f}")

    with open(out_dir / "summary.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nOutputs written to: {out_dir}")


if __name__ == "__main__":
    main()
