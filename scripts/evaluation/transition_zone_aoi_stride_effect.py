#!/usr/bin/env python3
"""
Does the inference stride matter inside the Demak transition-zone AOI?

Strides 8, 32 and 112, contrasted against the operational stride 32.

Scores Swin-B and ConvNeXtV2-B at BOTH strides, on the same AOI, same 6
concurrent S1-S2 pairs, same 3 seeds, same masking as
transition_zone_aoi_accuracy.py (which this imports, so the mask logic is
defined exactly once).

The contrast that answers the question is the WITHIN-SEED, WITHIN-MODEL paired
difference s8 - s32, evaluated on shared block resamples. Holding seed and model
fixed cancels BOTH the spatial-sampling variance (shared blocks) AND the
training-init variance (same checkpoint) -- the stride is then the only thing
that differs, so a CI on that difference is a clean test of the stride effect.

Comparing marginal s8 and s32 numbers instead would drown a ~0.005 stride effect
in the ~0.07-wide marginal CIs and in seed noise, and would answer nothing.

Usage:
    python scripts/evaluation/transition_zone_aoi_stride_effect.py \
        --aoi experiments/demak_semarang/transition_zone_aoi/demak_transition_zone_aoi.gpkg \
        --out experiments/demak_semarang/transition_zone_aoi/stride_effect
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from sklearn.metrics import average_precision_score, roc_auc_score

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "scripts" / "evaluation"))

from evaluation.metrics import compute_binary_metrics  # noqa: E402
from evaluation.runsel import RunDirError  # noqa: E402
from evaluation.runsel import resolve_run_dir as _resolve_run_dir  # noqa: E402
from inference_overlap_utils import threshold_probability  # noqa: E402
from transition_zone_aoi_accuracy import (  # noqa: E402
    BLOCK_PX,
    BOOT_SEED,
    COMPARISON,
    N_BOOT,
    REFERENCE_DIR,
    RUNS,
    SEEDS,
    THRESHOLD,
    load_aoi_geoms,
    load_pair,
    utc_now,
)

# FULL ROSTER, deliberately wider than transition_zone_aoi_accuracy.py.
#
# That script was reduced to the Swin-B vs ConvNeXtV2-B match-up on 2026-07-17
# because its question is an ARCHITECTURE comparison at equal capacity. This
# script's question is different -- how the stride moves the metric -- and it is
# answered per architecture (each row is a within-model, within-seed paired
# delta), so nothing is gained by restricting the roster and the extra rows show
# whether the stride response is architecture-dependent. All 7 roster models have
# 3 seeds at BOTH strides in the concurrent tree; the large variants (Swin-L,
# ConvNeXtV2-L) have s32 only and are therefore excluded -- they cannot be paired.
#
# Order matches Table 5 of the manuscript draft.
MODELS = {
    "Swin-B": "upernet_swin_base_224",
    "ConvNeXtV2-B": "upernet_tu-convnextv2_base",
    "DPT-ViT-B/16": "dpt_vit_b_16",
    "SegFormer-B4": "segformer_mit_b4",
    "DeepLabV3+": "deeplabv3plus_resnet50",
    "U-Net": "unet_resnet50",
    "U-Net++": "unetplusplus_resnet50",
}
MODEL_ORDER = list(MODELS)
# s112 added 2026-08-14 (inference run on gpu-rtx-hpo-west, project peak-tide-504014;
# keep-set egressed as demak_concurrent_s112_20260814.tar.zst). The operational
# stride is s32, so it is the REFERENCE of every paired contrast below and the
# comparisons are s8-s32 and s112-s32.
#
# NOTE the bootstrap below is unaffected by adding a stride: `mult` is drawn ONCE
# per iteration from an RNG seeded only by BOOT_SEED and applied to every key, so
# the draw sequence depends on (nb, n_boot) alone. Adding s112 keeps the existing
# s8-s32 numbers bit-identical (verified on re-run).
STRIDES = ["s8", "s32", "s112"]

# Every paired contrast is (stride - REFERENCE_STRIDE), within model and seed.
REFERENCE_STRIDE = "s32"


def resolve_run_dir(seed: str, token: str, stride: str) -> Path:
    """Exactly one run dir for (seed, token, stride).

    Anchored on the known sweep name + UTC stamp rather than a leading `*`, so
    a longer sweep name cannot be absorbed; `suffix` pins the arch token and
    stride (`unet_resnet50` vs `unetplusplus_resnet50`).
    """
    suffix = f"_{token}_native224_weighted_224_b0_{stride}"
    try:
        return _resolve_run_dir(RUNS, f"dev_sweep_all_demak_{seed}", suffix=suffix)
    except RunDirError as exc:
        raise SystemExit(str(exc)) from exc


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--aoi", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    pairs = []
    for ref in sorted(REFERENCE_DIR.glob("*veto_lt-02.tif")):
        m = re.search(r"(S1_\d{8}_\d{6}_\d+_\d+_\d+)", ref.name)
        if m:
            pairs.append((m.group(1), ref))
    with rasterio.open(pairs[0][1]) as src:
        target_crs = src.crs.to_string()
    aoi_geoms = load_aoi_geoms(args.aoi, target_crs)
    print(f"{len(pairs)} pairs; AOI in {target_crs}")

    rows = []
    cache: dict[tuple[str, str, str], dict[str, tuple]] = {}
    for model, token in MODELS.items():
        for stride in STRIDES:
            for seed in SEEDS:
                run_dir = resolve_run_dir(seed, token, stride)
                cache[(model, stride, seed)] = {}
                for s1_id, ref_path in pairs:
                    pred = run_dir / s1_id / f"{s1_id}_probability_water.tif"
                    if not pred.exists():
                        raise SystemExit(f"Missing {pred}")
                    y_true, y_score, block_id, diag = load_pair(ref_path, pred, aoi_geoms)
                    cache[(model, stride, seed)][s1_id] = (y_true, y_score, block_id)
                    y_pred = threshold_probability(y_score, THRESHOLD, COMPARISON)
                    rows.append(
                        {
                            "model": model,
                            "stride": stride,
                            "seed": seed,
                            "s1_id": s1_id,
                            **compute_binary_metrics(y_true=y_true, y_pred=y_pred, include_counts=True),
                            "roc_auc": float(roc_auc_score(y_true, y_score)),
                            "average_precision": float(average_precision_score(y_true, y_score)),
                            "valid_pixels_in_aoi": diag["valid_pixels_in_aoi"],
                        }
                    )
                print(f"  scored {model:13s} {stride:4s} {seed}")

    df = pd.DataFrame(rows)
    df.to_csv(args.out / "per_scene_metrics.csv", index=False)

    metric_cols = ["iou", "f1", "precision", "recall", "roc_auc", "average_precision"]
    per_seed = df.groupby(["model", "stride", "seed"], sort=False)[metric_cols].mean().reset_index()
    per_seed.to_csv(args.out / "per_seed_metrics.csv", index=False)

    across = (
        per_seed.groupby(["model", "stride"], sort=False)[metric_cols]
        .agg(["mean", "std"])
        .reset_index()
    )
    across.columns = ["_".join(c).rstrip("_") for c in across.columns]
    across.to_csv(args.out / "across_seed_summary.csv", index=False)

    # ---- paired block bootstrap: shared multiplicities across ALL (model,stride,seed)
    all_blocks = sorted(set(np.unique(cache[(MODEL_ORDER[0], "s32", SEEDS[0])][pairs[0][0]][2]).tolist()))
    for key in cache:
        for s1_id, _ in pairs:
            all_blocks = sorted(set(all_blocks) | set(np.unique(cache[key][s1_id][2]).tolist()))
    bidx = {b: i for i, b in enumerate(all_blocks)}
    nb = len(all_blocks)
    print(f"Paired bootstrap: {nb} blocks of {BLOCK_PX}px, R={args.n_boot}")

    def block_counts(key) -> np.ndarray:
        acc = np.zeros((nb, 4), dtype=np.int64)
        for s1_id, _ in pairs:
            y_true, y_score, block_id = cache[key][s1_id]
            y_pred = threshold_probability(y_score, THRESHOLD, COMPARISON)
            idx = np.array([bidx[b] for b in block_id], dtype=np.int64)
            t, p = y_true.astype(bool), y_pred.astype(bool)
            np.add.at(acc[:, 0], idx[t & p], 1)
            np.add.at(acc[:, 1], idx[~t & p], 1)
            np.add.at(acc[:, 2], idx[t & ~p], 1)
            np.add.at(acc[:, 3], idx[~t & ~p], 1)
        return acc

    counts = {k: block_counts(k) for k in cache}

    def iou_of(a):
        den = a[0] + a[1] + a[2]
        return float(a[0] / den) if den else np.nan

    def f1_of(a):
        den = 2 * a[0] + a[1] + a[2]
        return float(2 * a[0] / den) if den else np.nan

    rng = np.random.default_rng(BOOT_SEED)
    biou = {k: np.empty(args.n_boot) for k in counts}
    bf1 = {k: np.empty(args.n_boot) for k in counts}
    for it in range(args.n_boot):
        mult = rng.multinomial(nb, np.full(nb, 1.0 / nb))  # shared by every key
        for k, arr in counts.items():
            agg = mult @ arr
            biou[k][it] = iou_of(agg)
            bf1[k][it] = f1_of(agg)

    # within-(model, seed): stride - REFERENCE_STRIDE. Seed and checkpoint held
    # fixed => stride is the only difference. This is THE test.
    #
    # Long form: one row per (model, seed, stride) for every non-reference stride,
    # so adding a stride adds rows rather than columns. `d_iou` is signed so that
    # POSITIVE means the compared stride scores higher than s32.
    out = []
    others = [s for s in STRIDES if s != REFERENCE_STRIDE]
    for model in MODEL_ORDER:
        for seed in SEEDS:
            kref = (model, REFERENCE_STRIDE, seed)
            aref = counts[kref].sum(axis=0)
            for stride in others:
                k = (model, stride, seed)
                d = biou[k] - biou[kref]
                df1 = bf1[k] - bf1[kref]
                a = counts[k].sum(axis=0)
                lo, hi = float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))
                out.append(
                    {
                        "model": model,
                        "seed": seed,
                        "stride": stride,
                        "reference_stride": REFERENCE_STRIDE,
                        "iou": iou_of(a),
                        "iou_reference": iou_of(aref),
                        "d_iou": iou_of(a) - iou_of(aref),
                        "d_iou_ci_lower": lo,
                        "d_iou_ci_upper": hi,
                        "d_iou_excludes_zero": bool(lo > 0 or hi < 0),
                        "prob_stride_better": float((d > 0).mean()),
                        "d_f1": f1_of(a) - f1_of(aref),
                        "d_f1_ci_lower": float(np.percentile(df1, 2.5)),
                        "d_f1_ci_upper": float(np.percentile(df1, 97.5)),
                    }
                )
    ds = pd.DataFrame(out)
    ds.to_csv(args.out / "stride_contrast_paired.csv", index=False)

    # Back-compatible wide view of the s8-vs-s32 contrast, with the original
    # column names, so anything written against the 2-stride CSV keeps working.
    legacy = (
        ds[ds["stride"] == "s8"]
        .rename(
            columns={
                "iou": "iou_s8",
                "iou_reference": "iou_s32",
                "d_iou": "d_iou_s8_minus_s32",
                "prob_stride_better": "prob_s8_better",
                "d_f1": "d_f1_s8_minus_s32",
            }
        )
        .drop(columns=["stride", "reference_stride"])
    )
    legacy.to_csv(args.out / "stride_contrast_paired_s8_vs_s32.csv", index=False)

    # Does stride change the Swin-B vs ConvNeXtV2-B verdict?
    verdict = []
    for stride in STRIDES:
        for seed in SEEDS:
            d = biou[("Swin-B", stride, seed)] - biou[("ConvNeXtV2-B", stride, seed)]
            aS = counts[("Swin-B", stride, seed)].sum(axis=0)
            aC = counts[("ConvNeXtV2-B", stride, seed)].sum(axis=0)
            lo, hi = float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))
            verdict.append(
                {
                    "stride": stride,
                    "seed": seed,
                    "d_iou_swinb_minus_cnxb": iou_of(aS) - iou_of(aC),
                    "ci_lower": lo,
                    "ci_upper": hi,
                    "excludes_zero": bool(lo > 0 or hi < 0),
                }
            )
    pd.DataFrame(verdict).to_csv(args.out / "arch_contrast_by_stride.csv", index=False)

    (args.out / "run_metadata.json").write_text(
        json.dumps(
            {
                "generated_utc": utc_now(),
                "aoi_path": str(args.aoi),
                "threshold": THRESHOLD,
                "models": MODEL_ORDER,
                "strides": STRIDES,
                "seeds": SEEDS,
                "n_pairs": len(pairs),
                "block_px": BLOCK_PX,
                "n_blocks": nb,
                "n_boot": args.n_boot,
                "bootstrap_seed": BOOT_SEED,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    pd.set_option("display.width", 220)
    print("\n=== Across-seed (AOI, thr 0.5, macro-avg over 6 scenes) ===")
    print(
        across[["model", "stride", "iou_mean", "iou_std", "f1_mean", "roc_auc_mean", "average_precision_mean"]]
        .to_string(index=False, float_format=lambda v: f"{v:.4f}")
    )
    print(
        f"\n=== STRIDE EFFECT: paired (stride - {REFERENCE_STRIDE}), "
        "seed & checkpoint held fixed ==="
    )
    print(
        ds[["model", "seed", "stride", "iou_reference", "iou", "d_iou",
            "d_iou_ci_lower", "d_iou_ci_upper", "d_iou_excludes_zero",
            "prob_stride_better"]]
        .to_string(index=False, float_format=lambda v: f"{v:.4f}")
    )
    print("\n=== Swin-B - ConvNeXtV2-B, at each stride ===")
    print(pd.DataFrame(verdict).to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
