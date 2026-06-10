from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from pandas.errors import EmptyDataError


METRICS = ["oa", "f1", "precision", "recall", "iou", "mcc"]
DEFAULT_BASELINE_PRESETS = [
    "native224_weighted_224_b0_s112",
    "large_crop_only_1024_b128_s1024",
]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def load_yaml(path: Path) -> dict[str, Any]:
    return OmegaConf.to_container(OmegaConf.load(path), resolve=True)


def write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config=OmegaConf.create(payload), f=str(path))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=False), encoding="utf-8")


def read_csv_allow_empty(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except EmptyDataError:
        return pd.DataFrame()


def run_command(cmd: list[str], log_path: Path | None = None) -> None:
    started = time.time()
    print("$ " + " ".join(cmd), flush=True)

    if log_path is None:
        result = subprocess.run(cmd)
    else:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w", encoding="utf-8") as log_file:
            result = subprocess.run(cmd, stdout=log_file, stderr=subprocess.STDOUT, text=True)

    elapsed = (time.time() - started) / 60.0
    if result.returncode != 0:
        message = f"Command failed after {elapsed:.2f} min with exit code {result.returncode}: {' '.join(cmd)}"
        if log_path is not None:
            message += f"\nSee log: {log_path}"
        raise RuntimeError(message)


def list_subdirs(path: Path) -> set[Path]:
    if not path.exists():
        return set()
    return {p for p in path.iterdir() if p.is_dir()}


def find_newest_subdir(path: Path, before: set[Path] | None = None) -> Path:
    before = before or set()
    candidates = [p for p in list_subdirs(path) if p not in before]
    if not candidates:
        candidates = list(list_subdirs(path))
    if not candidates:
        raise FileNotFoundError(f"No subdirectories found under {path}")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def run_inference_sweep(sweep_config: Path, log_path: Path | None = None) -> Path:
    sweep_cfg = load_yaml(sweep_config)
    sweep = sweep_cfg["sweep"]
    sweep_output_root = Path(sweep.get("output_root", "outputs/inference/sweeps"))
    before = list_subdirs(sweep_output_root)

    run_command(["python", "scripts/run_inference_sweep.py", str(sweep_config)], log_path=log_path)

    return find_newest_subdir(sweep_output_root, before=before)


def run_evaluation(evaluation_config: Path, log_path: Path | None = None) -> Path:
    eval_cfg = load_yaml(evaluation_config)
    output_cfg = eval_cfg.get("output", {})
    output_root = Path(output_cfg.get("root", "outputs/evaluation/indonesia_inference_run_benchmark"))
    before = list_subdirs(output_root)

    run_command(
        ["python", "scripts/evaluate_indonesia_inference_run_benchmark.py", str(evaluation_config)],
        log_path=log_path,
    )

    return find_newest_subdir(output_root, before=before)


def normalize_model_name(value: str) -> str:
    return str(value).replace(" ", "_")


def build_model_entries_from_manifest(manifest: pd.DataFrame) -> tuple[dict[str, dict[str, str]], pd.DataFrame]:
    required = {"checkpoint_name", "preset_name", "run_name", "scene_output_dir"}
    missing = sorted(required - set(manifest.columns))
    if missing:
        raise ValueError(f"Sweep manifest is missing required columns: {missing}")

    rows: list[dict[str, Any]] = []
    models: dict[str, dict[str, str]] = {}

    group_cols = ["checkpoint_name", "preset_name", "run_name"]
    for (checkpoint_name, preset_name, run_name), group in manifest.groupby(group_cols, sort=False, dropna=False):
        first = group.iloc[0]
        scene_output_dir = Path(str(first["scene_output_dir"]))
        run_dir = scene_output_dir.parent
        model_name = normalize_model_name(f"{checkpoint_name}__{preset_name}")

        models[model_name] = {
            "run_dir": str(run_dir),
            "inference_mode": str(preset_name),
        }

        metadata = {
            "model_name": model_name,
            "checkpoint_name": checkpoint_name,
            "preset_name": preset_name,
            "run_name": run_name,
            "run_dir": str(run_dir),
            "num_manifest_rows": int(len(group)),
        }

        for col in [
            "checkpoint_path",
            "model_arch",
            "model_encoder",
            "tile_size",
            "buffer_size",
            "stride",
            "edge_policy",
            "stitching_mode",
            "blend_window",
            "tta_enabled",
            "tta_transforms",
            "elapsed_minutes",
            "status",
            "return_code",
            "error_message",
        ]:
            if col in first.index:
                metadata[col] = first[col]

        rows.append(metadata)

    return models, pd.DataFrame(rows)


def generate_evaluation_config(
    template_path: Path,
    manifest_path: Path,
    output_path: Path,
) -> tuple[Path, Path]:
    template = load_yaml(template_path)
    manifest = read_csv_allow_empty(manifest_path)
    models, model_metadata = build_model_entries_from_manifest(manifest)

    if not models:
        raise ValueError(f"No model entries could be generated from {manifest_path}")

    generated = dict(template)
    generated.setdefault("evaluation", {})
    generated["evaluation"]["models"] = models
    generated["evaluation"].pop("model_runs", None)

    output_cfg = generated.setdefault("output", {})
    output_cfg["run_name"] = f"{manifest_path.parent.name}__evaluation"

    write_yaml(output_path, generated)

    metadata_path = output_path.with_name("generated_model_metadata.csv")
    model_metadata.to_csv(metadata_path, index=False)
    return output_path, metadata_path


def summarize_by_model(metrics: pd.DataFrame, model_metadata: pd.DataFrame) -> pd.DataFrame:
    if metrics.empty or "model_name" not in metrics.columns:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for model_name, group in metrics.groupby("model_name", sort=False, dropna=False):
        row: dict[str, Any] = {
            "model_name": model_name,
            "n_scenes": int(len(group)),
            "n_evaluation_ids": int(group["evaluation_id"].nunique()) if "evaluation_id" in group else int(len(group)),
        }

        if "valid_pixels" in group:
            row["valid_pixels_total"] = int(group["valid_pixels"].sum())
        if "reference_water_pixels" in group:
            row["reference_water_pixels_total"] = int(group["reference_water_pixels"].sum())
        if "prediction_water_pixels" in group:
            row["prediction_water_pixels_total"] = int(group["prediction_water_pixels"].sum())

        for metric in METRICS:
            if metric not in group:
                continue
            values = group[metric].dropna().to_numpy(dtype=float)
            row[f"{metric}_mean"] = float(np.mean(values)) if len(values) else np.nan
            row[f"{metric}_median"] = float(np.median(values)) if len(values) else np.nan
            row[f"{metric}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else np.nan
            row[f"{metric}_min"] = float(np.min(values)) if len(values) else np.nan
            row[f"{metric}_max"] = float(np.max(values)) if len(values) else np.nan
            row[f"{metric}_ci_lower"] = float(np.percentile(values, 2.5)) if len(values) else np.nan
            row[f"{metric}_ci_upper"] = float(np.percentile(values, 97.5)) if len(values) else np.nan

        rows.append(row)

    summary = pd.DataFrame(rows)
    if not model_metadata.empty:
        summary = summary.merge(model_metadata, on="model_name", how="left")

    if "iou_mean" in summary:
        summary = summary.sort_values("iou_mean", ascending=False).reset_index(drop=True)
        summary.insert(0, "rank_by_iou", np.arange(1, len(summary) + 1))

    return summary


def paired_deltas_against_baseline(
    metrics: pd.DataFrame,
    model_metadata: pd.DataFrame,
    baseline_preset: str,
    metric: str = "iou",
) -> pd.DataFrame:
    if metrics.empty or metric not in metrics.columns:
        return pd.DataFrame()

    metadata_cols = ["model_name", "checkpoint_name", "preset_name"]
    missing = [col for col in metadata_cols if col not in model_metadata.columns]
    if missing:
        raise ValueError(f"Model metadata missing required columns for paired deltas: {missing}")

    annotated = metrics.merge(model_metadata[metadata_cols], on="model_name", how="left")
    id_cols = ["checkpoint_name", "evaluation_id"]
    if "evaluation_id" not in annotated.columns:
        raise ValueError("metrics_per_scene.csv must contain evaluation_id for paired deltas")

    baseline = annotated[annotated["preset_name"] == baseline_preset]
    if baseline.empty:
        return pd.DataFrame()

    baseline_values = baseline[id_cols + [metric]].rename(columns={metric: f"baseline_{metric}"})
    paired = annotated.merge(baseline_values, on=id_cols, how="inner")
    paired[f"delta_{metric}"] = paired[metric] - paired[f"baseline_{metric}"]
    paired = paired[paired["preset_name"] != baseline_preset]

    rows: list[dict[str, Any]] = []
    group_cols = ["checkpoint_name", "model_name", "preset_name"]
    for group_key, group in paired.groupby(group_cols, sort=False, dropna=False):
        checkpoint_name, model_name, preset_name = group_key
        values = group[f"delta_{metric}"].dropna().to_numpy(dtype=float)
        rows.append(
            {
                "baseline_preset": baseline_preset,
                "checkpoint_name": checkpoint_name,
                "model_name": model_name,
                "preset_name": preset_name,
                "n_paired_scenes": int(len(values)),
                f"delta_{metric}_mean": float(np.mean(values)) if len(values) else np.nan,
                f"delta_{metric}_median": float(np.median(values)) if len(values) else np.nan,
                f"delta_{metric}_std": float(np.std(values, ddof=1)) if len(values) > 1 else np.nan,
                f"delta_{metric}_min": float(np.min(values)) if len(values) else np.nan,
                f"delta_{metric}_max": float(np.max(values)) if len(values) else np.nan,
                f"delta_{metric}_ci_lower": float(np.percentile(values, 2.5)) if len(values) else np.nan,
                f"delta_{metric}_ci_upper": float(np.percentile(values, 97.5)) if len(values) else np.nan,
            }
        )

    return pd.DataFrame(rows)


def summarize_tta_gains(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty or "preset_name" not in summary or "checkpoint_name" not in summary:
        return pd.DataFrame()

    records: list[dict[str, Any]] = []
    lookup = summary.set_index(["checkpoint_name", "preset_name"])

    for _, row in summary.iterrows():
        preset = str(row.get("preset_name", ""))
        if not preset.endswith("_tta_flip4"):
            continue
        base_preset = preset.removesuffix("_tta_flip4")
        key = (row["checkpoint_name"], base_preset)
        if key not in lookup.index:
            continue
        base = lookup.loc[key]
        if isinstance(base, pd.DataFrame):
            base = base.iloc[0]
        record = {
            "checkpoint_name": row["checkpoint_name"],
            "base_preset": base_preset,
            "tta_preset": preset,
            "base_model_name": base["model_name"],
            "tta_model_name": row["model_name"],
        }
        for metric in METRICS:
            col = f"{metric}_mean"
            if col in summary.columns:
                record[f"{metric}_gain_mean"] = row.get(col, np.nan) - base.get(col, np.nan)
        records.append(record)

    return pd.DataFrame(records)


def build_rankings(
    evaluation_dir: Path,
    model_metadata_path: Path,
    output_dir: Path,
    baseline_presets: list[str],
) -> None:
    metrics_path = evaluation_dir / "metrics_per_scene.csv"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Expected evaluator output not found: {metrics_path}")

    metrics = read_csv_allow_empty(metrics_path)
    model_metadata = read_csv_allow_empty(model_metadata_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = summarize_by_model(metrics, model_metadata)
    summary.to_csv(output_dir / "ranked_configs.csv", index=False)

    if "checkpoint_name" in summary.columns and "iou_mean" in summary.columns:
        ranked_by_checkpoint = (
            summary.sort_values(["checkpoint_name", "iou_mean"], ascending=[True, False])
            .groupby("checkpoint_name", group_keys=False)
            .apply(lambda g: g.assign(rank_within_checkpoint=np.arange(1, len(g) + 1)), include_groups=False)
            .reset_index(drop=True)
        )
        ranked_by_checkpoint.to_csv(output_dir / "ranked_configs_by_checkpoint.csv", index=False)

    all_delta_tables = []
    for baseline in baseline_presets:
        delta = paired_deltas_against_baseline(metrics, model_metadata, baseline_preset=baseline, metric="iou")
        if delta.empty:
            continue
        safe_baseline = baseline.replace("/", "_")
        delta.to_csv(output_dir / f"paired_deltas_vs_{safe_baseline}.csv", index=False)
        all_delta_tables.append(delta)

    if all_delta_tables:
        pd.concat(all_delta_tables, ignore_index=True).to_csv(output_dir / "paired_deltas_all_baselines.csv", index=False)

    tta_gains = summarize_tta_gains(summary)
    if not tta_gains.empty:
        tta_gains.to_csv(output_dir / "tta_gain_summary.csv", index=False)

    write_json(
        output_dir / "ranking_metadata.json",
        {
            "created_at": utc_now_iso(),
            "evaluation_dir": str(evaluation_dir),
            "model_metadata_csv": str(model_metadata_path),
            "metrics_per_scene_csv": str(metrics_path),
            "baseline_presets": baseline_presets,
            "primary_sort_metric": "iou_mean",
            "outputs": {
                "ranked_configs_csv": str(output_dir / "ranked_configs.csv"),
                "ranked_configs_by_checkpoint_csv": str(output_dir / "ranked_configs_by_checkpoint.csv"),
                "paired_deltas_all_baselines_csv": str(output_dir / "paired_deltas_all_baselines.csv"),
                "tta_gain_summary_csv": str(output_dir / "tta_gain_summary.csv"),
            },
        },
    )


def cleanup_heavy_outputs(manifest_path: Path, dry_run: bool = False) -> pd.DataFrame:
    manifest = read_csv_allow_empty(manifest_path)
    candidate_paths: set[Path] = set()

    for _, row in manifest.iterrows():
        for col in ["probability_geotiff", "shoreline_geojson"]:
            if col in row and pd.notna(row[col]) and str(row[col]):
                candidate_paths.add(Path(str(row[col])))
        if "scene_output_dir" in row and pd.notna(row["scene_output_dir"]):
            scene_dir = Path(str(row["scene_output_dir"]))
            candidate_paths.update(scene_dir.glob("*_probability_water.memmap"))
            candidate_paths.update(scene_dir.glob("*_probability_water_tta_*.memmap"))
            candidate_paths.update(scene_dir.glob("*_probability_water_tta_*.tif"))

    rows = []
    for path in sorted(candidate_paths):
        existed = path.exists()
        size_bytes = path.stat().st_size if existed and path.is_file() else 0
        deleted = False
        if existed and not dry_run:
            if path.is_file():
                path.unlink()
                deleted = True
            elif path.is_dir():
                shutil.rmtree(path)
                deleted = True
        rows.append({"path": str(path), "existed": existed, "deleted": deleted, "size_bytes": size_bytes})

    return pd.DataFrame(rows)


def copy_log_into_sweep_dir(source_log: Path | None, sweep_dir: Path, target_name: str) -> Path | None:
    if source_log is None or not source_log.exists():
        return None
    target = sweep_dir / target_name
    shutil.copy2(source_log, target)
    return target


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run an inference-config sweep, generate an Indonesia benchmark evaluation config, "
            "run the existing evaluator, and write ranked config summaries."
        )
    )
    parser.add_argument(
        "--sweep-config",
        default="configs/inference_sweep_indonesia_config_benchmark_v1.yaml",
        help="Path to an inference sweep config consumed by scripts/run_inference_sweep.py.",
    )
    parser.add_argument(
        "--evaluation-template",
        default="configs/evaluation/indonesia_inference_config_benchmark_template.yaml",
        help="Evaluation template whose evaluation.models block will be generated from the sweep manifest.",
    )
    parser.add_argument(
        "--sweep-dir",
        default=None,
        help="Existing sweep directory to evaluate. If provided, the wrapper does not run a new inference sweep.",
    )
    parser.add_argument(
        "--evaluation-dir",
        default=None,
        help="Existing evaluation output directory to summarize. If provided, the wrapper does not run the evaluator.",
    )
    parser.add_argument("--skip-inference", action="store_true", help="Do not run the inference sweep; require --sweep-dir.")
    parser.add_argument("--skip-evaluation", action="store_true", help="Generate evaluation config only; do not run evaluator or ranking.")
    parser.add_argument("--cleanup-heavy-outputs", action="store_true", help="Delete probability GeoTIFFs/memmaps after successful ranking.")
    parser.add_argument("--cleanup-dry-run", action="store_true", help="List cleanup targets without deleting them.")
    parser.add_argument(
        "--baseline-preset",
        action="append",
        default=None,
        help="Preset name to use for paired deltas. Can be provided multiple times.",
    )
    parser.add_argument("--log-commands", action="store_true", help="Write inference/evaluation stdout/stderr logs.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sweep_config = Path(args.sweep_config)
    evaluation_template = Path(args.evaluation_template)
    baseline_presets = args.baseline_preset or DEFAULT_BASELINE_PRESETS

    if args.skip_inference and not args.sweep_dir:
        raise ValueError("--skip-inference requires --sweep-dir")

    inference_log_path: Path | None = None
    copied_inference_log_path: Path | None = None
    if args.sweep_dir:
        sweep_dir = Path(args.sweep_dir)
    else:
        if args.log_commands:
            inference_log_path = Path("outputs/inference/sweeps") / f"benchmark_inference_sweep_{int(time.time())}.log"
        sweep_dir = run_inference_sweep(sweep_config, log_path=inference_log_path)
        copied_inference_log_path = copy_log_into_sweep_dir(inference_log_path, sweep_dir, "inference_sweep.log")

    if not sweep_dir.exists():
        raise FileNotFoundError(f"Sweep directory does not exist: {sweep_dir}")

    manifest_path = sweep_dir / "sweep_manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Sweep manifest not found: {manifest_path}")

    generated_eval_config = sweep_dir / "generated_evaluation_config.yaml"
    generated_eval_config, model_metadata_path = generate_evaluation_config(
        template_path=evaluation_template,
        manifest_path=manifest_path,
        output_path=generated_eval_config,
    )

    evaluation_log_path: Path | None = None
    if args.evaluation_dir:
        evaluation_dir = Path(args.evaluation_dir)
    elif args.skip_evaluation:
        evaluation_dir = None
    else:
        evaluation_log_path = (sweep_dir / "evaluation.log") if args.log_commands else None
        evaluation_dir = run_evaluation(generated_eval_config, log_path=evaluation_log_path)

    if evaluation_dir is not None:
        summary_dir = sweep_dir / "benchmark_summary"
        build_rankings(
            evaluation_dir=evaluation_dir,
            model_metadata_path=model_metadata_path,
            output_dir=summary_dir,
            baseline_presets=baseline_presets,
        )

        if args.cleanup_heavy_outputs or args.cleanup_dry_run:
            cleanup_df = cleanup_heavy_outputs(manifest_path, dry_run=args.cleanup_dry_run)
            cleanup_df.to_csv(summary_dir / "cleanup_heavy_outputs.csv", index=False)

    write_json(
        sweep_dir / "benchmark_wrapper_metadata.json",
        {
            "script": "scripts/benchmark_inference_config_sweep.py",
            "created_at": utc_now_iso(),
            "sweep_config": str(sweep_config),
            "evaluation_template": str(evaluation_template),
            "sweep_dir": str(sweep_dir),
            "sweep_manifest": str(manifest_path),
            "generated_evaluation_config": str(generated_eval_config),
            "generated_model_metadata_csv": str(model_metadata_path),
            "evaluation_dir": str(evaluation_dir) if evaluation_dir is not None else None,
            "baseline_presets": baseline_presets,
            "cleanup_heavy_outputs": bool(args.cleanup_heavy_outputs),
            "cleanup_dry_run": bool(args.cleanup_dry_run),
            "inference_log": str(copied_inference_log_path or inference_log_path) if (copied_inference_log_path or inference_log_path) else None,
            "evaluation_log": str(evaluation_log_path) if evaluation_log_path else None,
        },
    )

    print("Benchmark wrapper complete")
    print(f"Sweep directory: {sweep_dir}")
    print(f"Generated evaluation config: {generated_eval_config}")
    if evaluation_dir is not None:
        print(f"Evaluation directory: {evaluation_dir}")
        print(f"Benchmark summary: {sweep_dir / 'benchmark_summary'}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
