"""Accuracy assessment of the segwater architectures against the deprecated ResNet50 study.

Evaluates 23 model-runs (2 old ResNet50-UNet + 7 new architectures x 3 seeds) against the
DEM flood ground-truth masks at Hampyeong Bay, on the three SAR dates used by the
original study, through a single code path on identical pixels.

The original analysis lived in four scripts under sen12coast_global_dl/. Their
point-estimate metrics were correct but every inferential statistic in them was
broken (see OLD_SCRIPTS_AUDIT.md). This script therefore reports point estimates
and a seed-level spread only; it deliberately reports no p-values. Metrics and
raster alignment are delegated to the repo-canonical modules rather than
reimplemented.

Ground truth, valid mask, and the old ResNet50 predictions live on an external
volume; pass --nas-root if it is mounted elsewhere.

Run with an interpreter that has rasterio, e.g.
    /opt/homebrew/Caskroom/miniforge/base/envs/eda/bin/python
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

_SCRIPTS = Path(__file__).resolve().parent.parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from evaluation.metrics import METRIC_NAMES, compute_binary_metrics  # noqa: E402
from inference_overlap_utils import load_overlap_reference_and_prediction  # noqa: E402

DEFAULT_NAS_ROOT = "/Volumes/WD_8tb_RedPlus_NAS_A/MACKBOOK_AIR_M2_BACKUP/Documents/EOS/ACDC"
DEFAULT_RUNS_ROOT = "experiments/hampyeong/runs"
DEFAULT_OUT_DIR = "experiments/hampyeong/evaluation"

# The three SAR dates the deprecated study evaluated. Ground truth exists for
# 20210504 / 20210913 / 20211019 too, but 20211019 was inferred on a pre-clipped
# input (different spatial context), so the set is not homogeneous beyond these.
DATES = ["20210305", "20210422", "20210621"]

# scene_id stems, keyed by date, shared by all four new runs.
SCENE_IDS = {
    "20210305": "S1B_IW_GRDH_1SDV_20210305T213224_20210305T213249_025885_031658_6933_Clipped",
    "20210422": "S1B_IW_GRDH_1SDV_20210422T213225_20210422T213250_026585_032CBA_CF3C_Clipped",
    "20210621": "S1B_IW_GRDH_1SDV_20210621T213228_20210621T213253_027460_034782_6C41_Clipped",
}

# The 21 new runs: (model label, arch, seed, run-dir name).
# Note the dir naming: the FIRST s<n> is the seed, the trailing s32 is the stride.
# The three resnet50-encoder architectures (deeplabv3+, unet, unet++) are distinct
# models despite the shared encoder -- verified to produce distinct predictions.
_NEW_RUN_SPECS = [
    # (arch label, checkpoint key stem, {seed: run-dir name})
    ("Swin-B", "upernet_tu-swin_base_patch4_window7_224", {
        19: "dev_sweep_all_hampyeong_s19_20260710T100323Z_upernet_swin_base_224_native224_weighted_224_b0_s32",
        42: "dev_sweep_all_hampyeong_s42_20260716T063710Z_upernet_swin_base_224_native224_weighted_224_b0_s32",
        58: "dev_sweep_all_hampyeong_s58_20260710T101618Z_upernet_swin_base_224_native224_weighted_224_b0_s32",
    }),
    ("ConvNeXtV2", "upernet_tu-convnextv2_base", {
        19: "dev_sweep_all_hampyeong_s19_20260710T100323Z_upernet_tu-convnextv2_base_native224_weighted_224_b0_s32",
        42: "dev_sweep_all_hampyeong_s42_20260710T100817Z_upernet_tu-convnextv2_base_native224_weighted_224_b0_s32",
        58: "dev_sweep_all_hampyeong_s58_20260710T101618Z_upernet_tu-convnextv2_base_native224_weighted_224_b0_s32",
    }),
    ("DeepLabV3+", "deeplabv3plus_resnet50", {
        19: "dev_sweep_all_hampyeong_s19_20260710T161002Z_deeplabv3plus_resnet50_native224_weighted_224_b0_s32",
        42: "dev_sweep_all_hampyeong_s42_20260710T161020Z_deeplabv3plus_resnet50_native224_weighted_224_b0_s32",
        58: "dev_sweep_all_hampyeong_s58_20260710T161048Z_deeplabv3plus_resnet50_native224_weighted_224_b0_s32",
    }),
    ("DPT-ViT-B", "dpt_tu-vit_base_patch16_224.mae", {
        19: "dev_sweep_all_hampyeong_s19_20260710T161002Z_dpt_vit_b_16_native224_weighted_224_b0_s32",
        42: "dev_sweep_all_hampyeong_s42_20260710T161020Z_dpt_vit_b_16_native224_weighted_224_b0_s32",
        58: "dev_sweep_all_hampyeong_s58_20260710T161048Z_dpt_vit_b_16_native224_weighted_224_b0_s32",
    }),
    ("SegFormer-B4", "segformer_mit_b4", {
        19: "dev_sweep_all_hampyeong_s19_20260710T161002Z_segformer_mit_b4_native224_weighted_224_b0_s32",
        42: "dev_sweep_all_hampyeong_s42_20260710T161020Z_segformer_mit_b4_native224_weighted_224_b0_s32",
        58: "dev_sweep_all_hampyeong_s58_20260710T161048Z_segformer_mit_b4_native224_weighted_224_b0_s32",
    }),
    ("UNet-R50", "unet_resnet50", {
        19: "dev_sweep_all_hampyeong_s19_20260710T161002Z_unet_resnet50_native224_weighted_224_b0_s32",
        42: "dev_sweep_all_hampyeong_s42_20260710T161020Z_unet_resnet50_native224_weighted_224_b0_s32",
        58: "dev_sweep_all_hampyeong_s58_20260710T161048Z_unet_resnet50_native224_weighted_224_b0_s32",
    }),
    ("UNet++-R50", "unetplusplus_resnet50", {
        19: "dev_sweep_all_hampyeong_s19_20260710T161002Z_unetplusplus_resnet50_native224_weighted_224_b0_s32",
        42: "dev_sweep_all_hampyeong_s42_20260710T161020Z_unetplusplus_resnet50_native224_weighted_224_b0_s32",
        58: "dev_sweep_all_hampyeong_s58_20260710T161048Z_unetplusplus_resnet50_native224_weighted_224_b0_s32",
    }),
]

# Flattened (model label, arch, seed, run-dir) plus the checkpoint-key stem per arch.
NEW_RUNS = []
CKPT_KEY_BY_ARCH = {}
for _arch, _ckpt_key, _dirs in _NEW_RUN_SPECS:
    CKPT_KEY_BY_ARCH[_arch] = _ckpt_key
    _slug = _arch.lower().replace("+", "plus").replace("-", "_").replace(".", "")
    for _seed, _dir in _dirs.items():
        NEW_RUNS.append((f"{_slug}_s{_seed}", _arch, _seed, _dir))

# Architectures that carry the multi-seed treatment (all new ones do; 3 seeds each).
NEW_ARCHS = [spec[0] for spec in _NEW_RUN_SPECS]

# The two old ResNet50 runs, as subdirectories under the NAS inference tree.
OLD_RUNS = [
    ("resnet50_baseline", "ResNet50-baseline", "Pretrained_Baseline_Resnet50_for_reproj"),
    ("resnet50_finetuned", "ResNet50-finetuned", "Baseline_Resnet50_reproj_FINETUNED"),
]

# Manuscript Table 2, Hampyeong Bay (deprecated draft). Point estimates only;
# the published values are means of 10k bootstrap subsamples of 400 pixels, so
# they sit within ~1e-3 of the full-population values this script computes.
# MCC was never published; the manuscript reports no p-values.
PUBLISHED = {
    ("20210305", "resnet50_baseline"): dict(oa=0.947, recall=0.997, precision=0.949, f1=0.973, iou=0.947),
    ("20210305", "resnet50_finetuned"): dict(oa=0.941, recall=0.987, precision=0.952, f1=0.969, iou=0.941),
    ("20210422", "resnet50_baseline"): dict(oa=0.854, recall=0.943, precision=0.806, f1=0.869, iou=0.769),
    ("20210422", "resnet50_finetuned"): dict(oa=0.909, recall=0.925, precision=0.901, f1=0.912, iou=0.839),
    ("20210621", "resnet50_baseline"): dict(oa=0.881, recall=0.947, precision=0.833, f1=0.886, iou=0.796),
    ("20210621", "resnet50_finetuned"): dict(oa=0.931, recall=0.929, precision=0.929, f1=0.929, iou=0.867),
}

EXPECTED_VALID_PIXELS = 1_179_967
REPRO_TOLERANCE = 0.0015
PROBABILITY_THRESHOLD = 0.5
PROBABILITY_COMPARISON = "greater_equal"
RESOLUTION_ATOL = 1e-6


def gt_path(nas_root: Path, date: str) -> Path:
    return nas_root / "sen12coast/Validation/Hampyeong/DEM_FLOOD_MASKS/Descending_reproj" / f"DEM_FLOOD_S1_DESC_{date}_VAL_AOI.tif"


def valid_mask_path(nas_root: Path) -> Path:
    return nas_root / "Tide_Gauge/Korean_Peninsula/DEM_wrt_WGS84_TBM/DEM_VALID_MASK_aoi.tif"


def old_pred_path(nas_root: Path, subdir: str, date: str) -> Path:
    return nas_root / "sen12coast/Inference/Hampeyeong/Models" / subdir / "DESC/Geotiffs_clipped_Val_AOI" / f"Inference_DESC_{date}_Clipped_VAL_AOI.tif"


def new_pred_path(runs_root: Path, run_dir: str, date: str) -> Path:
    scene = SCENE_IDS[date]
    return runs_root / run_dir / scene / f"{scene}_probability_water.tif"


def _require(path: Path, what: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"{what} not found: {path}")
    return path


def audit_checkpoint_provenance(runs_root: Path) -> None:
    """Assert each new run loaded the checkpoint its directory name advertises.

    This project has twice hit a bug where a run directory labelled with one seed
    loaded another seed's checkpoint. Config, summary and manifest must agree with
    each other and with the directory name, and all checkpoints must be distinct.
    The checkpoint-key stem (e.g. `unet_resnet50`) disambiguates the three
    resnet50-encoder architectures, which share an encoder but are distinct models.
    Metadata agreement alone cannot catch a mislabelled checkpoint file, so
    evaluate() additionally relies on the predictions themselves differing.
    """
    seen: dict[str, str] = {}
    for model, arch, seed, run_dir in NEW_RUNS:
        base = _require(runs_root / run_dir, f"run dir for {model}")
        config = yaml.safe_load((base / "run_config.yaml").read_text())
        summary = json.loads((base / "run_summary.json").read_text())

        cfg_ckpt = config["inference"]["checkpoint_path"]
        expected_key = CKPT_KEY_BY_ARCH[arch]

        if cfg_ckpt != summary["checkpoint_path"]:
            raise AssertionError(f"{model}: run_config checkpoint {cfg_ckpt} != run_summary {summary['checkpoint_path']}")

        # Checkpoint path is outputs/stage2/{key}/s{seed}/best.pth
        expected_ckpt = f"outputs/stage2/{expected_key}/s{seed}/best.pth"
        if cfg_ckpt != expected_ckpt:
            raise AssertionError(f"{model}: expected checkpoint {expected_ckpt}, loaded {cfg_ckpt}")

        ckpt_seed = re.search(r"/s(\d+)/", cfg_ckpt)
        if not ckpt_seed or int(ckpt_seed.group(1)) != seed:
            raise AssertionError(f"{model}: directory advertises seed {seed} but loaded checkpoint {cfg_ckpt}")

        manifest = pd.read_csv(base / "run_manifest.csv")
        manifest_ckpts = set(manifest["checkpoint_path"].unique())
        if manifest_ckpts != {cfg_ckpt}:
            raise AssertionError(f"{model}: run_manifest checkpoints {manifest_ckpts} disagree with config {cfg_ckpt}")

        if cfg_ckpt in seen:
            raise AssertionError(f"checkpoint collision: {model} and {seen[cfg_ckpt]} both load {cfg_ckpt}")
        seen[cfg_ckpt] = model

    print(f"Checkpoint provenance audit passed: {len(seen)} distinct checkpoints across {len(NEW_RUNS)} runs\n")


def load_pair(reference: Path, prediction: Path, mask: Path,
              threshold: float = PROBABILITY_THRESHOLD) -> tuple[np.ndarray, np.ndarray]:
    """Return (y_true, y_pred) over the valid mask, on the shared intersection window.

    Both old (binary {0,1}) and new (float32 probability) predictions go through
    the same loader. Thresholding a binary raster at >=0.5 is the identity, so
    the single code path is exact for both.

    threshold defaults to the deployment operating point (0.5); the
    threshold-calibration analysis passes per-model values selected on the
    training validation split (docs/threshold_calibration/).

    reference_nodata_values is None: the DEM flood masks carry no nodata sentinel
    and are strictly {0,1}. All masking is done by the external valid mask.
    """
    y_true, y_pred, diagnostics = load_overlap_reference_and_prediction(
        reference_path=str(reference),
        prediction_path=str(prediction),
        reference_water_values=[1],
        reference_nodata_values=None,
        probability_threshold=threshold,
        probability_comparison=PROBABILITY_COMPARISON,
        resolution_atol=RESOLUTION_ATOL,
        valid_mask_path=str(mask),
        valid_mask_value=1,
    )
    n_valid = diagnostics["valid_pixels_after_all_masks"]
    if n_valid != EXPECTED_VALID_PIXELS:
        raise AssertionError(
            f"valid pixel count {n_valid} != expected {EXPECTED_VALID_PIXELS} for "
            f"prediction {prediction.name}. The valid mask or grid alignment changed."
        )
    return y_true, y_pred


def evaluate(nas_root: Path, runs_root: Path) -> pd.DataFrame:
    mask = _require(valid_mask_path(nas_root), "valid mask")
    rows: list[dict] = []

    for date in DATES:
        reference = _require(gt_path(nas_root, date), f"ground truth {date}")
        y_true_reference: np.ndarray | None = None
        prediction_digests: dict[str, str] = {}

        specs: list[tuple[str, str, int | None, Path]] = []
        for model, arch, subdir in OLD_RUNS:
            specs.append((model, arch, None, _require(old_pred_path(nas_root, subdir, date), f"{model} {date}")))
        for model, arch, seed, run_dir in NEW_RUNS:
            specs.append((model, arch, seed, _require(new_pred_path(runs_root, run_dir, date), f"{model} {date}")))

        for model, arch, seed, prediction in specs:
            y_true, y_pred = load_pair(reference, prediction, mask)

            # Two runs producing identical predictions would mean they loaded the same
            # weights regardless of what their configs claim. Metadata cannot catch this.
            if seed is not None:
                digest = hashlib.sha1(y_pred.tobytes()).hexdigest()
                if digest in prediction_digests:
                    raise AssertionError(
                        f"{model} and {prediction_digests[digest]} produced identical predictions on "
                        f"{date}. Two runs loaded the same weights — suspect seed miswiring."
                    )
                prediction_digests[digest] = model

            # The invariant bootstrap_permutation_accurcy_assessment.py violated:
            # every model on a given date must see the identical ground-truth vector.
            if y_true_reference is None:
                y_true_reference = y_true
            elif not np.array_equal(y_true, y_true_reference):
                raise AssertionError(f"y_true differs across models on {date} (model={model})")

            metrics = compute_binary_metrics(y_true, y_pred, include_counts=True)
            counts_total = metrics["tn"] + metrics["fp"] + metrics["fn"] + metrics["tp"]
            if counts_total != EXPECTED_VALID_PIXELS:
                raise AssertionError(f"confusion counts sum to {counts_total}, expected {EXPECTED_VALID_PIXELS}")

            rows.append({
                "date": date,
                "model": model,
                "arch": arch,
                "seed": seed,
                **{m: metrics[m] for m in METRIC_NAMES},
                **{c: metrics[c] for c in ("tn", "fp", "fn", "tp")},
                "n_valid": counts_total,
                "water_fraction_gt": float(y_true.mean()),
                "prediction_path": str(prediction),
            })
            print(f"  {date}  {model:20s} IoU={metrics['iou']:.4f}  F1={metrics['f1']:.4f}  MCC={metrics['mcc']:.4f}")

    return pd.DataFrame(rows)


def aggregate_by_architecture(df: pd.DataFrame) -> pd.DataFrame:
    """Seed as replicate: mean +/- SD (ddof=1) across seeds, per architecture x date.

    Three seeds (19/42/58). Per the project's multi-seed reporting convention the SD is
    a descriptive spread, not the basis of a significance test at this n.
    Old ResNet50 runs are single-seed and pass through with std = NaN.
    """
    rows: list[dict] = []
    for (date, arch), group in df.groupby(["date", "arch"], sort=False):
        row = {"date": date, "arch": arch, "n_seeds": int(group["seed"].notna().sum()) or 1}
        for metric in METRIC_NAMES:
            values = group[metric].to_numpy()
            row[f"{metric}_mean"] = float(np.mean(values))
            row[f"{metric}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else np.nan
            row[f"{metric}_min"] = float(np.min(values))
            row[f"{metric}_max"] = float(np.max(values))
        rows.append(row)
    return pd.DataFrame(rows)


def reproduction_check(df: pd.DataFrame) -> pd.DataFrame:
    """Compare recomputed ResNet50 metrics against the deprecated manuscript's Table 2."""
    rows: list[dict] = []
    indexed = df.set_index(["date", "model"])
    for (date, model), published in PUBLISHED.items():
        recomputed = indexed.loc[(date, model)]
        for metric, pub_value in published.items():
            got = float(recomputed[metric])
            rows.append({
                "date": date,
                "model": model,
                "metric": metric,
                "published": pub_value,
                "recomputed": got,
                "delta": got - pub_value,
                "abs_delta": abs(got - pub_value),
            })
    out = pd.DataFrame(rows)
    worst = out["abs_delta"].max()
    if worst >= REPRO_TOLERANCE:
        offenders = out.loc[out["abs_delta"] >= REPRO_TOLERANCE, ["date", "model", "metric", "published", "recomputed"]]
        raise AssertionError(
            f"Reproduction of the published Table 2 failed: max |delta| = {worst:.5f} "
            f">= {REPRO_TOLERANCE}.\n{offenders.to_string(index=False)}"
        )
    print(f"\nReproduction check passed: max |delta| = {worst:.5f} across {len(out)} published cells")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--nas-root", type=Path, default=Path(DEFAULT_NAS_ROOT), help="Root of the ACDC tree holding ground truth, valid mask, and old ResNet50 predictions.")
    parser.add_argument("--runs-root", type=Path, default=Path(DEFAULT_RUNS_ROOT), help="Directory holding the four new dev_sweep run dirs.")
    parser.add_argument("--out-dir", type=Path, default=Path(DEFAULT_OUT_DIR), help="Where the CSV artifacts are written.")
    args = parser.parse_args()

    if not args.nas_root.exists():
        raise SystemExit(f"NAS root not found: {args.nas_root}\nIs the external volume mounted?")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    audit_checkpoint_provenance(args.runs_root)

    print(f"Evaluating {len(OLD_RUNS) + len(NEW_RUNS)} model-runs x {len(DATES)} dates\n")
    per_date = evaluate(args.nas_root, args.runs_root)
    by_arch = aggregate_by_architecture(per_date)
    repro = reproduction_check(per_date)

    per_date.to_csv(args.out_dir / "per_date_metrics.csv", index=False)
    by_arch.to_csv(args.out_dir / "by_architecture.csv", index=False)
    repro.to_csv(args.out_dir / "published_vs_recomputed.csv", index=False)
    print(f"\nWrote 3 CSVs to {args.out_dir}")


if __name__ == "__main__":
    main()
