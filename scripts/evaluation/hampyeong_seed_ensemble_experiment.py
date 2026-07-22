"""EXPERIMENTAL — soft ensembling of Hampyeong probability rasters.

Questions:
  1. Does aggregating the three seed probability rasters within an architecture beat the
     individual seeds, and does it lift Swin-B past the site-finetuned ResNet50?
  2. Does pooling all six runs (2 architectures x 3 seeds) do better still?
  3. How much of any gain comes from ARCHITECTURE diversity rather than seed diversity?

Motivation: Swin-B's per-seed IoU spread at 20210422 is 0.0806 (SD 0.0403), 6.4x
ConvNeXtV2's, yet within-architecture seed ensembling recovers little of it -- evidence
the seeds err in a correlated direction. Cross-architecture ensembling supplies the one
kind of diversity seed-ensembling cannot.

Ensemble families, all soft (aggregate probabilities, THEN threshold):

    within_arch    3 seeds of one architecture              (2 sets)
    cross_arch     2 architectures at one matched seed      (3 sets)
    all_six        all 2 architectures x 3 seeds            (1 set)

Aggregators: mean and median, per-pixel.

    NOTE on median at even n: with 6 members np.median averages the two middle values,
    so the 6-member "median" is not a true order statistic (no member's value need be
    selected). It is reported for completeness; the mean is the principled aggregator here.

Threshold 0.5 with `>=`, matching run_manifest.csv. Ground truth, valid mask and grid are
identical to the main comparison, so every number is directly comparable to
`experiments/hampyeong/evaluation/per_date_metrics.csv`.

This is an EXPERIMENT. Outputs land in a separate directory and are not part of the
headline comparison. Chief caveat: ensembles are selected and evaluated on the same three
scenes, so their advantage is optimistic. See the generated markdown.

Run with an interpreter that has rasterio, e.g.
    /opt/homebrew/Caskroom/miniforge/base/envs/eda/bin/python
"""

from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

_SCRIPTS = Path(__file__).resolve().parent.parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from evaluation.hampyeong_model_comparison import (  # noqa: E402
    DATES,
    EXPECTED_VALID_PIXELS,
    NEW_RUNS,
    OLD_RUNS,
    PROBABILITY_COMPARISON,
    PROBABILITY_THRESHOLD,
    audit_checkpoint_provenance,
    gt_path,
    load_pair,
    new_pred_path,
    old_pred_path,
    valid_mask_path,
)
from evaluation.metrics import METRIC_NAMES, compute_binary_metrics  # noqa: E402
from inference_overlap_utils import rounded_window_from_bounds, threshold_probability  # noqa: E402

DEFAULT_NAS_ROOT = "/Volumes/WD_8tb_RedPlus_NAS_A/MACKBOOK_AIR_M2_BACKUP/Documents/EOS/ACDC"
DEFAULT_RUNS_ROOT = "experiments/hampyeong/runs"
DEFAULT_OUT_DIR = "experiments/hampyeong/experimental_seed_ensemble"

ARCHS = ("Swin-B", "ConvNeXtV2")
SEEDS = (19, 42, 58)

AGGREGATORS = {
    "mean": lambda stack: stack.mean(axis=0),
    "median": lambda stack: np.median(stack, axis=0),
}


def ensemble_sets() -> list[tuple[str, str, list[tuple[str, int]]]]:
    """(family, label, [(arch, seed), ...]) for every ensemble we evaluate."""
    sets: list[tuple[str, str, list[tuple[str, int]]]] = []
    for arch in ARCHS:
        sets.append(("within_arch", arch, [(arch, s) for s in SEEDS]))
    for seed in SEEDS:
        sets.append(("cross_arch", f"Swin-B+ConvNeXtV2 s{seed}", [(a, seed) for a in ARCHS]))
    sets.append(("all_six", "all 6 runs", [(a, s) for a in ARCHS for s in SEEDS]))
    return sets


def _aoi_window_read(raster_path: Path, aoi_bounds) -> np.ndarray:
    """Read a raster over the AOI bounds. Predictions may sit on the larger full clip."""
    with rasterio.open(raster_path) as src:
        return src.read(1, window=rounded_window_from_bounds(aoi_bounds, src.transform))


def load_member_probabilities(runs_root: Path, date: str, aoi_bounds, valid: np.ndarray) -> dict[tuple[str, int], np.ndarray]:
    """Probability at valid pixels for every (arch, seed) run on this date."""
    run_dir_by_member = {(arch, seed): run_dir for _, arch, seed, run_dir in NEW_RUNS}
    probabilities: dict[tuple[str, int], np.ndarray] = {}
    for member, run_dir in run_dir_by_member.items():
        raster = _aoi_window_read(new_pred_path(runs_root, run_dir, date), aoi_bounds)
        if raster.shape != valid.shape:
            raise AssertionError(f"{member} {date}: raster {raster.shape} != mask {valid.shape}")
        probabilities[member] = raster.astype(np.float32)[valid]

    # Two runs with identical probabilities would mean duplicate weights (miswiring signature).
    for a, b in itertools.combinations(sorted(probabilities), 2):
        if np.array_equal(probabilities[a], probabilities[b]):
            raise AssertionError(f"{date}: {a} and {b} have identical probability rasters")
    return probabilities


def evaluate(nas_root: Path, runs_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    mask_path = valid_mask_path(nas_root)
    with rasterio.open(mask_path) as src_mask:
        aoi_bounds = src_mask.bounds
        valid = src_mask.read(1) == 1
    if int(valid.sum()) != EXPECTED_VALID_PIXELS:
        raise AssertionError(f"valid mask has {valid.sum()} pixels, expected {EXPECTED_VALID_PIXELS}")

    ensemble_rows: list[dict] = []
    member_rows: list[dict] = []

    for date in DATES:
        reference_path = gt_path(nas_root, date)
        with rasterio.open(reference_path) as src_ref:
            reference = src_ref.read(1)
        y_true = (reference[valid] > 0).astype(np.uint8)

        # Baselines via the SAME loader the main comparison uses: these rows must
        # reproduce per_date_metrics.csv exactly.
        for model, arch, subdir in OLD_RUNS:
            _, y_pred = load_pair(reference_path, old_pred_path(nas_root, subdir, date), mask_path)
            member_rows.append({"date": date, "arch": arch, "member": model, "seed": None,
                                **compute_binary_metrics(y_true, y_pred, include_counts=True)})
        for model, arch, seed, run_dir in NEW_RUNS:
            _, y_pred = load_pair(reference_path, new_pred_path(runs_root, run_dir, date), mask_path)
            member_rows.append({"date": date, "arch": arch, "member": model, "seed": seed,
                                **compute_binary_metrics(y_true, y_pred, include_counts=True)})

        probabilities = load_member_probabilities(runs_root, date, aoi_bounds, valid)

        for family, label, members in ensemble_sets():
            stack = np.stack([probabilities[m] for m in members], axis=0)
            for name, fn in AGGREGATORS.items():
                p_ensemble = fn(stack)
                y_pred = threshold_probability(p_ensemble, PROBABILITY_THRESHOLD, PROBABILITY_COMPARISON)
                metrics = compute_binary_metrics(y_true, y_pred, include_counts=True)
                if metrics["tn"] + metrics["fp"] + metrics["fn"] + metrics["tp"] != EXPECTED_VALID_PIXELS:
                    raise AssertionError("confusion counts do not sum to the valid pixel count")

                member_ious = [
                    compute_binary_metrics(y_true, threshold_probability(probabilities[m], PROBABILITY_THRESHOLD, PROBABILITY_COMPARISON))["iou"]
                    for m in members
                ]
                ensemble_rows.append({
                    "date": date, "family": family, "ensemble": label, "aggregator": name,
                    "n_members": len(members),
                    "members": "+".join(f"{a}:s{s}" for a, s in members),
                    "even_n_median_is_interpolated": bool(name == "median" and len(members) % 2 == 0),
                    **metrics, "n_valid": EXPECTED_VALID_PIXELS,
                    "water_fraction_gt": float(y_true.mean()),
                    "member_iou_mean": float(np.mean(member_ious)),
                    "member_iou_best": float(np.max(member_ious)),
                    "member_iou_worst": float(np.min(member_ious)),
                })
                print(f"  {date}  {family:11s} {label:26s} {name:6s}  IoU={metrics['iou']:.4f}  F1={metrics['f1']:.4f}  MCC={metrics['mcc']:.4f}")

    return pd.DataFrame(ensemble_rows), pd.DataFrame(member_rows)


def build_comparison(ensembles: pd.DataFrame, members: pd.DataFrame) -> pd.DataFrame:
    finetuned = members[members.member == "resnet50_finetuned"].set_index("date")
    rows: list[dict] = []
    for _, row in ensembles.iterrows():
        baseline_iou = float(finetuned.loc[row.date, "iou"])
        rows.append({
            "date": row.date, "family": row.family, "ensemble": row.ensemble,
            "aggregator": row.aggregator, "n_members": row.n_members,
            "iou": float(row.iou), "f1": float(row.f1), "mcc": float(row.mcc),
            "member_iou_mean": float(row.member_iou_mean),
            "member_iou_best": float(row.member_iou_best),
            "iou_minus_member_mean": float(row.iou - row.member_iou_mean),
            "iou_minus_member_best": float(row.iou - row.member_iou_best),
            "beats_all_members": bool(row.iou > row.member_iou_best),
            "finetuned_resnet50_iou": baseline_iou,
            "beats_finetuned_resnet50": bool(row.iou > baseline_iou),
        })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--nas-root", type=Path, default=Path(DEFAULT_NAS_ROOT))
    parser.add_argument("--runs-root", type=Path, default=Path(DEFAULT_RUNS_ROOT))
    parser.add_argument("--out-dir", type=Path, default=Path(DEFAULT_OUT_DIR))
    args = parser.parse_args()

    if not args.nas_root.exists():
        raise SystemExit(f"NAS root not found: {args.nas_root}\nIs the external volume mounted?")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    audit_checkpoint_provenance(args.runs_root)

    n_sets = len(ensemble_sets())
    print(f"EXPERIMENTAL: {n_sets} ensemble sets x {len(AGGREGATORS)} aggregators x {len(DATES)} dates\n")
    ensembles, members = evaluate(args.nas_root, args.runs_root)
    comparison = build_comparison(ensembles, members)

    ensembles.to_csv(args.out_dir / "experimental_ensemble_metrics.csv", index=False)
    members.to_csv(args.out_dir / "experimental_single_member_metrics.csv", index=False)
    comparison.to_csv(args.out_dir / "experimental_ensemble_vs_members.csv", index=False)
    print(f"\nWrote 3 CSVs to {args.out_dir}")

    print("\nIoU by family (mean aggregator), vs finetuned ResNet50:")
    view = comparison[comparison.aggregator == "mean"]
    for _, r in view.sort_values(["date", "family", "ensemble"]).iterrows():
        print(f"  {r.date} {r.family:11s} {r.ensemble:26s} n={r.n_members} "
              f"IoU={r.iou:.4f} ({r.iou_minus_member_mean:+.4f} vs member mean, "
              f"{r.iou_minus_member_best:+.4f} vs best) "
              f"{'beats' if r.beats_finetuned_resnet50 else 'loses to'} ft")


if __name__ == "__main__":
    main()
