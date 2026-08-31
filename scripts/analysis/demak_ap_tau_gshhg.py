#!/usr/bin/env python3
"""Demak AP + leave-one-scene-out tau*, on the ADOPTED GSHHG-only mask.

Supersedes scripts/analysis/demak_ap_tau_per_architecture.py, which read the
101-point threshold sweeps under
`s1_s2_concurrent_accuracy_multiseed/evaluation/` -- that tree is scored on
`GSHHG_GlobalSurfaceWater_combined_mask.tif`, the RETIRED domain. The published
Demak estimand is `data/GSHHG_mask.tif` (GSHHG-only, ~3.24 M px, stride 32); see
the demak-accuracy-mask-decision note. The combined mask carves out 284,412 px
of permanent open water, which changes both the accuracy values and the
water prevalence the no-skill line sits at.

No rasters are re-scored. The GSHHG shadow tree already holds 1000-bin
probability histograms per (model, date), keyed by spatial block:

    $TMPDIR/gshhg_eval/semarang_probability_aucroc_<seed>_s32/
        spatial_block_bootstrap/histograms_block200px/<model>__<date>.npz
            count (B, 1000)  total pixels per block per bin
            pos   (B, 1000)  water pixels per block per bin

Summing over blocks gives the scene's full probability histogram, which is a
sufficient statistic for AP, precision/recall at any threshold, and IoU. The
1000-bin grid is FINER than the 101-point sweep the superseded artifacts used,
so nothing is lost by working from it.

Estimand (unchanged from the superseded script, so the two are comparable
except for the mask):
  - AP per (seed, arch, scene), then macro-averaged over scenes, then reported
    as mean +/- SD across the 3 seeds.
  - AP rule: step-interpolated (sklearn `average_precision_score`), computed
    here by the same cumulative-from-the-top construction as
    `spatial_block_bootstrap.metrics_from_hist`.
  - tau*: for each held-out scene, the IoU-maximising threshold selected on the
    OTHER five scenes, evaluated per (seed, arch).
  - no-skill / water prevalence: pos.sum() / count.sum() per scene -- the
    precision a classifier gets by calling every pixel water.

Outputs (into --outdir):
  demak_ap_tau_gshhg_per_scene.csv    one row per (seed, arch, scene)
  demak_ap_tau_gshhg_per_seed.csv     one row per (seed, arch)
  demak_ap_tau_gshhg_summary.csv      one row per arch, mean +/- seed SD
  PROVENANCE.json

Run: /opt/homebrew/Caskroom/miniforge/base/envs/eda/bin/python demak_ap_tau_gshhg.py
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

SEEDS = ("s19", "s42", "s58")
STRIDE = "s32"
FINE = 1000

# Histogram-file model stem -> manuscript notation.
ARCH = {
    "Upernet_Swin_Base_224_native224_weighted": "UPerNet/Swin-B",
    "Upernet_ConvnextV2base_native224_weighted": "UPerNet/CNXt-B",
    "DPT_ViT_B_16_native224_weighted": "DPT/ViT",
    "DeepLabV3plus_Resnet50_native224_weighted": "DLv3+/R50",
    "Segformer_MiT_B4_native224_weighted": "SegFormer/MiT-B4",
    "Unet_Resnet50_native224_weighted": "UNet/R50",
    "UnetPlusPlus_Resnet50_native224_weighted": "UNet++/R50",
}

# Bin left edges; a bin is "predicted water" when its left edge >= threshold,
# matching metrics_from_hist's greater-than convention.
EDGES = np.linspace(0.0, 1.0, FINE + 1)
LEFT = EDGES[:-1]


def curve_from_hist(pos: np.ndarray, count: np.ndarray):
    """Precision/recall/IoU at every bin boundary, from one scene's histogram.

    Cumulative-from-the-top: index k means "predict water for bins >= k".
    """
    neg = count - pos
    tp = np.cumsum(pos[::-1])[::-1]
    fp = np.cumsum(neg[::-1])[::-1]
    P = pos.sum()
    fn = P - tp
    with np.errstate(invalid="ignore", divide="ignore"):
        precision = np.where(tp + fp > 0, tp / (tp + fp), 1.0)
        recall = tp / P if P else np.zeros_like(tp, dtype=float)
        iou = np.where(tp + fp + fn > 0, tp / (tp + fp + fn), 0.0)
    return precision, recall, iou


def average_precision(precision: np.ndarray, recall: np.ndarray) -> float:
    """Step-interpolated AP -- identical rule to metrics_from_hist."""
    rec_next = np.concatenate([recall[1:], [0.0]])
    return float(np.sum((recall - rec_next) * precision))


def main() -> None:
    ap_ = argparse.ArgumentParser()
    ap_.add_argument(
        "--tree",
        type=Path,
        default=Path(os.environ.get("TMPDIR", "/tmp")) / "gshhg_eval",
        help="GSHHG shadow tree (rebuild with gshhg_variant/scripts/build_gshhg_tree.sh)",
    )
    ap_.add_argument(
        "--outdir",
        type=Path,
        default=Path(
            "/Users/noel/Documents/Research_Projects/segwater_v2/experiments/"
            "demak_semarang/s1_s2_concurrent_accuracy_multiseed/artifacts_ap_tau_gshhg"
        ),
    )
    args = ap_.parse_args()

    if not args.tree.exists():
        raise SystemExit(
            f"GSHHG tree not found at {args.tree} — rebuild it with "
            "mask_sensitivity/gshhg_variant/scripts/build_gshhg_tree.sh"
        )
    args.outdir.mkdir(parents=True, exist_ok=True)

    rows = []
    curve_cache: dict[tuple[str, str, str], tuple] = {}
    masks_seen: set[str] = set()
    for seed in SEEDS:
        run = args.tree / f"semarang_probability_aucroc_{seed}_{STRIDE}"
        hdir = run / "spatial_block_bootstrap" / "histograms_block200px"
        if not hdir.is_dir():
            raise SystemExit(f"missing histograms: {hdir}")

        # Guard: this script is only valid on the GSHHG-only domain.
        meta_txt = (run / "run_metadata.json").read_text()
        found = re.findall(r"[^\"]*mask[^\"]*\.tif", meta_txt)
        masks_seen.update(found)
        if not any(m.endswith("GSHHG_mask.tif") for m in found):
            raise SystemExit(
                f"{run.name}: expected GSHHG_mask.tif, run_metadata names {found}"
            )

        fine = json.loads((hdir / "histograms_meta.json").read_text())["fine_bins"]
        if fine != FINE:
            raise SystemExit(f"{run.name}: fine_bins={fine}, expected {FINE}")

        for f in sorted(hdir.glob("*__*.npz")):
            stem, date = f.stem.split("__")
            if stem not in ARCH:
                continue
            z = np.load(f)
            pos = z["pos"].sum(axis=0)      # sum over blocks -> whole scene
            count = z["count"].sum(axis=0)
            n, n_pos = int(count.sum()), int(pos.sum())

            precision, recall, iou = curve_from_hist(pos, count)
            curve_cache[(seed, ARCH[stem], date)] = (pos, count)
            rows.append(
                {
                    "seed": seed,
                    "arch": ARCH[stem],
                    "date": date,
                    "ap": average_precision(precision, recall),
                    "water_fraction": n_pos / n if n else np.nan,
                    "n_valid": n,
                    "n_water": n_pos,
                    # IoU profile is kept out of the CSV; tau* is derived below.
                    # No leading underscore: itertuples() renames such columns.
                    "iou_profile": iou,
                }
            )

    per_scene = pd.DataFrame(rows)
    if per_scene.empty:
        raise SystemExit("no histograms matched the expected model stems")

    # ---- leave-one-scene-out tau*, per (seed, arch) ----
    iou_lookup = {
        (r.seed, r.arch, r.date): r.iou_profile for r in per_scene.itertuples()
    }
    dates = sorted(per_scene["date"].unique())
    tau_rows = []
    for (seed, arch), _ in per_scene.groupby(["seed", "arch"]):
        for held in dates:
            others = [d for d in dates if d != held]
            total = np.sum([iou_lookup[(seed, arch, d)] for d in others], axis=0)
            k = int(np.argmax(total))
            tau_rows.append(
                {
                    "seed": seed,
                    "arch": arch,
                    "date": held,
                    "tau_star": float(LEFT[k]),
                    "iou_at_tau": float(iou_lookup[(seed, arch, held)][k]),
                    "n_selection_scenes": len(others),
                }
            )

    # ---- PR curves, on a 101-point grid for the figures ----
    # The histograms are 1000-bin; the figure scripts consume a per-(seed, arch,
    # scene) precision/recall table. Emit one at 0.01 spacing (every 10th bin
    # boundary) so the curve CSV stays the same size as the superseded sweep it
    # replaces. AP itself is computed above at the FULL 1000-bin resolution.
    curve_rows = []
    for r in per_scene.itertuples():
        pos, count = curve_cache[(r.seed, r.arch, r.date)]
        precision, recall, _ = curve_from_hist(pos, count)
        idx = np.arange(0, FINE, 10)
        curve_rows.append(
            pd.DataFrame({
                "seed": r.seed, "arch": r.arch, "date": r.date,
                "threshold": LEFT[idx],
                "precision": precision[idx],
                "recall": recall[idx],
            })
        )
    curves = pd.concat(curve_rows, ignore_index=True)
    curves.to_csv(args.outdir / "demak_pr_curves_gshhg.csv", index=False)

    per_scene = per_scene.drop(columns=["iou_profile"]).merge(
        pd.DataFrame(tau_rows), on=["seed", "arch", "date"], how="left"
    )
    per_scene.to_csv(args.outdir / "demak_ap_tau_gshhg_per_scene.csv", index=False)

    per_seed = (
        per_scene.groupby(["seed", "arch"], as_index=False)
        .agg(ap=("ap", "mean"), tau_star=("tau_star", "mean"),
             iou_at_tau=("iou_at_tau", "mean"), n_scenes=("date", "nunique"))
    )
    per_seed.to_csv(args.outdir / "demak_ap_tau_gshhg_per_seed.csv", index=False)

    summary = (
        per_seed.groupby("arch", as_index=False)
        .agg(ap_mean=("ap", "mean"), ap_sd=("ap", "std"),
             tau_mean=("tau_star", "mean"), tau_sd=("tau_star", "std"),
             iou_tau_mean=("iou_at_tau", "mean"), n_seeds=("seed", "nunique"))
        .sort_values("ap_mean", ascending=False)
    )
    summary.to_csv(args.outdir / "demak_ap_tau_gshhg_summary.csv", index=False)

    wf = per_scene.groupby("date")["water_fraction"].first()
    prov = {
        "mask": "GSHHG_mask.tif (GSHHG-only) — the ADOPTED Demak domain",
        "supersedes": "artifacts_ap_tau/ (combined GSHHG+GSW mask, retired)",
        "source_tree": str(args.tree),
        "masks_named_in_run_metadata": sorted(masks_seen),
        "stride": STRIDE,
        "seeds": list(SEEDS),
        "fine_bins": FINE,
        "n_scenes": int(per_scene["date"].nunique()),
        "ap_rule": "step-interpolated on the 1000-bin histogram grid "
                   "(same construction as spatial_block_bootstrap.metrics_from_hist)",
        "tau_rule": "LOO IoU-optimal, selected on the other 5 scenes",
        "water_fraction_mean": float(wf.mean()),
        "water_fraction_min": float(wf.min()),
        "water_fraction_max": float(wf.max()),
    }
    (args.outdir / "PROVENANCE.json").write_text(json.dumps(prov, indent=2))

    with pd.option_context("display.width", 200):
        print(summary.round(4).to_string(index=False))
    print(f"\nAP spread across architectures: {summary.ap_mean.min():.3f}–"
          f"{summary.ap_mean.max():.3f} "
          f"(spread {summary.ap_mean.max() - summary.ap_mean.min():.3f}, "
          f"{(summary.ap_mean.max() - summary.ap_mean.min()) / summary.ap_sd.max():.1f}x "
          f"max seed SD)")
    print(f"tau* range: {summary.tau_mean.min():.3f}–{summary.tau_mean.max():.3f}")
    print(f"water fraction (no-skill): mean {wf.mean():.4f}, "
          f"per-scene {wf.min():.3f}–{wf.max():.3f}")
    print(f"\nwrote 3 CSVs + PROVENANCE.json to {args.outdir}")


if __name__ == "__main__":
    main()
