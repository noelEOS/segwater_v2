from __future__ import annotations

import argparse
import copy
import json
import re
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


def timestamp_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def sanitize_for_path(value: str) -> str:
    value = str(value).strip().replace(".", "p")
    value = re.sub(r"[^A-Za-z0-9_-]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("_")
    return value or "unknown"


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


def run_single_inference_sweep(sweep_config: Path, log_path: Path | None = None) -> Path:
    sweep_cfg = load_yaml(sweep_config)
    sweep_root = Path(sweep_cfg["sweep"].get("output_root", "outputs/inference/sweeps"))
    before = list_subdirs(sweep_root)
    run_command(["python", "scripts/run_inference_sweep.py", str(sweep_config)], log_path=log_path)
    return find_newest_subdir(sweep_root, before=before)


def run_evaluation(evaluation_config: Path, log_path: Path | None = None) -> Path:
    eval_cfg = load_yaml(evaluation_config)
    output_root = Path(eval_cfg.get("output", {}).get("root", "outputs/evaluation/indonesia_inference_run_benchmark"))
    before = list_subdirs(output_root)
    run_command(["python", "scripts/evaluate_indonesia_inference_run_benchmark.py", str(evaluation_config)], log_path=log_path)
    return find_newest_subdir(output_root, before=before)


def preset_model_input_size(preset: dict[str, Any]) -> int | None:
    overrides = preset.get("overrides", {})
    tile_size = overrides.get("inference.data.tile_size")
    buffer_size = overrides.get("inference.data.buffer_size", 0)
    if tile_size is None:
        return None
    return int(tile_size) + 2 * int(buffer_size or 0)


def check_checkpoint_preset_compatibility(checkpoint: dict[str, Any], preset: dict[str, Any]) -> tuple[bool, str, int | None, int | None]:
    preset_name = str(preset.get("name", ""))
    allowed_presets = checkpoint.get("allowed_presets")
    model_input_size = preset_model_input_size(preset)

    if allowed_presets is not None and preset_name not in set(map(str, allowed_presets)):
        return False, "not_in_allowed_presets", model_input_size, checkpoint.get("max_model_input_size")

    max_model_input_size = checkpoint.get("max_model_input_size")
    if max_model_input_size is not None and model_input_size is not None:
        max_model_input_size = int(max_model_input_size)
        if model_input_size > max_model_input_size:
            return False, "input_too_large", model_input_size, max_model_input_size

    return True, "", model_input_size, max_model_input_size


def build_compatibility_matrix(sweep_cfg: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    sweep = sweep_cfg["sweep"]
    for checkpoint in sweep.get("checkpoints", []):
        for preset in sweep.get("presets", []):
            compatible, reason, model_input_size, max_model_input_size = check_checkpoint_preset_compatibility(checkpoint, preset)
            rows.append({
                "checkpoint_name": checkpoint.get("name", ""),
                "checkpoint_path": checkpoint.get("checkpoint_path", ""),
                "model_arch": checkpoint.get("model", {}).get("arch", ""),
                "model_encoder": checkpoint.get("model", {}).get("encoder_name", ""),
                "preset_name": preset.get("name", ""),
                "tile_size": preset.get("overrides", {}).get("inference.data.tile_size", ""),
                "buffer_size": preset.get("overrides", {}).get("inference.data.buffer_size", ""),
                "model_input_size": model_input_size if model_input_size is not None else "",
                "max_model_input_size": max_model_input_size if max_model_input_size is not None else "",
                "compatible": bool(compatible),
                "skip_reason": reason,
            })
    return pd.DataFrame(rows)


def make_filtered_sweep_config(master_cfg: dict[str, Any], checkpoint: dict[str, Any], compatible_presets: list[dict[str, Any]], sweep_name_suffix: str) -> dict[str, Any]:
    generated = copy.deepcopy(master_cfg)
    generated["sweep"]["name"] = f"{master_cfg['sweep']['name']}__{sweep_name_suffix}"
    generated["sweep"]["checkpoints"] = [checkpoint]
    generated["sweep"]["presets"] = compatible_presets
    return generated


def run_compatible_inference_sweeps(sweep_config: Path, log_commands: bool = False) -> tuple[Path, Path, Path | None]:
    master_cfg = load_yaml(sweep_config)
    sweep = master_cfg["sweep"]
    sweep_root = Path(sweep.get("output_root", "outputs/inference/sweeps"))
    wrapper_dir = sweep_root / f"{sanitize_for_path(sweep['name'])}__{timestamp_id()}__benchmark_wrapper"
    generated_sweeps_dir = wrapper_dir / "generated_sweeps"
    logs_dir = wrapper_dir / "logs"
    wrapper_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(sweep_config, wrapper_dir / "master_sweep_config.yaml")

    compatibility = build_compatibility_matrix(master_cfg)
    compatibility.to_csv(wrapper_dir / "compatibility_matrix.csv", index=False)
    compatibility[~compatibility["compatible"]].to_csv(wrapper_dir / "skipped_incompatible_configs.csv", index=False)

    sweep_dirs: list[Path] = []
    manifest_frames: list[pd.DataFrame] = []
    run_rows: list[dict[str, Any]] = []

    for idx, checkpoint in enumerate(sweep.get("checkpoints", []), start=1):
        checkpoint_name = str(checkpoint.get("name", f"checkpoint_{idx:03d}"))
        compatible_presets = [
            preset for preset in sweep.get("presets", [])
            if check_checkpoint_preset_compatibility(checkpoint, preset)[0]
        ]
        if not compatible_presets:
            run_rows.append({
                "checkpoint_name": checkpoint_name,
                "generated_sweep_config": "",
                "sweep_dir": "",
                "num_presets": 0,
                "status": "skipped",
                "message": "no compatible presets",
            })
            continue

        suffix = f"{idx:03d}_{sanitize_for_path(checkpoint_name)}"
        generated_cfg = make_filtered_sweep_config(master_cfg, checkpoint, compatible_presets, suffix)
        generated_cfg_path = generated_sweeps_dir / f"sweep_{suffix}.yaml"
        write_yaml(generated_cfg_path, generated_cfg)

        log_path = logs_dir / f"sweep_{suffix}.log" if log_commands else None
        sweep_dir = run_single_inference_sweep(generated_cfg_path, log_path=log_path)
        sweep_dirs.append(sweep_dir)

        manifest_path = sweep_dir / "sweep_manifest.csv"
        if manifest_path.exists():
            manifest = read_csv_allow_empty(manifest_path)
            if not manifest.empty:
                manifest["source_sweep_dir"] = str(sweep_dir)
                manifest["source_sweep_manifest"] = str(manifest_path)
                manifest_frames.append(manifest)
        run_rows.append({
            "checkpoint_name": checkpoint_name,
            "generated_sweep_config": str(generated_cfg_path),
            "sweep_dir": str(sweep_dir),
            "num_presets": len(compatible_presets),
            "status": "completed",
            "message": "",
        })

    if not manifest_frames:
        raise RuntimeError(f"No sweep manifests were produced under {wrapper_dir}")

    combined_manifest = pd.concat(manifest_frames, ignore_index=True)
    combined_manifest_path = wrapper_dir / "sweep_manifest.csv"
    combined_manifest.to_csv(combined_manifest_path, index=False)
    pd.DataFrame(run_rows).to_csv(wrapper_dir / "generated_sweep_runs.csv", index=False)

    counts = combined_manifest["status"].value_counts(dropna=False).to_dict() if "status" in combined_manifest else {}
    write_json(wrapper_dir / "sweep_summary.json", {
        "sweep_id": wrapper_dir.name,
        "master_sweep_config": str(sweep_config),
        "generated_sweep_runs_csv": str(wrapper_dir / "generated_sweep_runs.csv"),
        "compatibility_matrix_csv": str(wrapper_dir / "compatibility_matrix.csv"),
        "combined_manifest_csv": str(combined_manifest_path),
        "source_sweep_dirs": [str(path) for path in sweep_dirs],
        "status_counts": counts,
    })
    return wrapper_dir, combined_manifest_path, logs_dir if log_commands else None


def normalize_model_name(value: str) -> str:
    return str(value).replace(" ", "_")


def _as_int_return_code(value: Any) -> int | None:
    if pd.isna(value) or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def build_model_entries_from_manifest(manifest: pd.DataFrame) -> tuple[dict[str, dict[str, str]], pd.DataFrame]:
    required = {"checkpoint_name", "preset_name", "run_name", "scene_output_dir"}
    missing = sorted(required - set(manifest.columns))
    if missing:
        raise ValueError(f"Sweep manifest is missing required columns: {missing}")

    models: dict[str, dict[str, str]] = {}
    rows: list[dict[str, Any]] = []
    skipped_rows: list[dict[str, Any]] = []
    group_cols = ["checkpoint_name", "preset_name", "run_name"]

    for (checkpoint_name, preset_name, run_name), group in manifest.groupby(group_cols, sort=False, dropna=False):
        first = group.iloc[0]
        statuses = set(str(v) for v in group["status"].dropna().unique()) if "status" in group else set()
        return_codes = {_as_int_return_code(v) for v in group["return_code"].dropna().unique()} if "return_code" in group else {0}
        all_success = (not statuses or statuses == {"success"}) and return_codes.issubset({0, None})

        run_dir = Path(str(first["scene_output_dir"])).parent
        model_name = normalize_model_name(f"{checkpoint_name}__{preset_name}")
        n_probability_geotiffs = len(list(run_dir.glob("*/*_probability_water.tif"))) if run_dir.exists() else 0

        base_row: dict[str, Any] = {
            "model_name": model_name,
            "checkpoint_name": checkpoint_name,
            "preset_name": preset_name,
            "run_name": run_name,
            "run_dir": str(run_dir),
            "num_manifest_rows": int(len(group)),
            "n_probability_geotiffs": int(n_probability_geotiffs),
        }
        for col in [
            "checkpoint_path", "model_arch", "model_encoder", "tile_size", "buffer_size",
            "stride", "edge_policy", "stitching_mode", "blend_window", "tta_enabled",
            "tta_transforms", "elapsed_minutes", "status", "return_code", "error_message",
            "source_sweep_dir", "source_sweep_manifest",
        ]:
            if col in first.index:
                base_row[col] = first[col]

        if not all_success:
            skipped_rows.append({**base_row, "skip_reason": "inference_group_not_successful"})
            continue
        if n_probability_geotiffs == 0:
            skipped_rows.append({**base_row, "skip_reason": "no_probability_geotiffs"})
            continue

        models[model_name] = {"run_dir": str(run_dir), "inference_mode": str(preset_name)}
        rows.append(base_row)

    metadata = pd.DataFrame(rows)
    skipped = pd.DataFrame(skipped_rows)
    metadata.attrs["skipped_model_metadata"] = skipped
    return models, metadata


def generate_evaluation_config(template_path: Path, manifest_path: Path, output_path: Path) -> tuple[Path, Path]:
    template = load_yaml(template_path)
    manifest = read_csv_allow_empty(manifest_path)
    models, model_metadata = build_model_entries_from_manifest(manifest)
    skipped_model_metadata = model_metadata.attrs.get("skipped_model_metadata", pd.DataFrame())
    if not models:
        skipped_path = output_path.with_name("generated_model_metadata_skipped.csv")
        skipped_model_metadata.to_csv(skipped_path, index=False)
        raise ValueError(
            f"No successful model entries with probability GeoTIFFs could be generated from {manifest_path}. "
            f"Skipped metadata written to {skipped_path}"
        )

    generated = dict(template)
    generated.setdefault("evaluation", {})
    generated["evaluation"]["models"] = models
    generated["evaluation"].pop("model_runs", None)
    generated.setdefault("output", {})["run_name"] = f"{manifest_path.parent.name}__evaluation"

    write_yaml(output_path, generated)
    metadata_path = output_path.with_name("generated_model_metadata.csv")
    skipped_path = output_path.with_name("generated_model_metadata_skipped.csv")
    model_metadata.to_csv(metadata_path, index=False)
    skipped_model_metadata.to_csv(skipped_path, index=False)
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
        for col in ["valid_pixels", "reference_water_pixels", "prediction_water_pixels"]:
            if col in group.columns:
                row[f"{col}_total"] = int(group[col].sum())
        for metric in METRICS:
            if metric not in group.columns:
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
    if "iou_mean" in summary.columns:
        summary = summary.sort_values("iou_mean", ascending=False).reset_index(drop=True)
        summary.insert(0, "rank_by_iou", np.arange(1, len(summary) + 1))
    return summary


def paired_deltas_against_baseline(metrics: pd.DataFrame, model_metadata: pd.DataFrame, baseline_preset: str, metric: str = "iou") -> pd.DataFrame:
    if metrics.empty or metric not in metrics.columns:
        return pd.DataFrame()
    metadata_cols = ["model_name", "checkpoint_name", "preset_name"]
    missing = [col for col in metadata_cols if col not in model_metadata.columns]
    if missing:
        raise ValueError(f"Model metadata missing required columns for paired deltas: {missing}")

    annotated = metrics.merge(model_metadata[metadata_cols], on="model_name", how="left")
    if "evaluation_id" not in annotated.columns:
        raise ValueError("metrics_per_scene.csv must contain evaluation_id for paired deltas")

    baseline = annotated[annotated["preset_name"] == baseline_preset]
    if baseline.empty:
        return pd.DataFrame()
    baseline_values = baseline[["checkpoint_name", "evaluation_id", metric]].rename(columns={metric: f"baseline_{metric}"})
    paired = annotated.merge(baseline_values, on=["checkpoint_name", "evaluation_id"], how="inner")
    paired[f"delta_{metric}"] = paired[metric] - paired[f"baseline_{metric}"]
    paired = paired[paired["preset_name"] != baseline_preset]

    rows: list[dict[str, Any]] = []
    for (checkpoint_name, model_name, preset_name), group in paired.groupby(["checkpoint_name", "model_name", "preset_name"], sort=False, dropna=False):
        values = group[f"delta_{metric}"].dropna().to_numpy(dtype=float)
        rows.append({
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
        })
    return pd.DataFrame(rows)


def summarize_tta_gains(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty or "preset_name" not in summary.columns or "checkpoint_name" not in summary.columns:
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


def add_rank_within_checkpoint(summary: pd.DataFrame) -> pd.DataFrame:
    ranked_groups = []
    sorted_summary = summary.sort_values(["checkpoint_name", "iou_mean"], ascending=[True, False])
    for _, group in sorted_summary.groupby("checkpoint_name", sort=False, dropna=False):
        ranked_groups.append(group.assign(rank_within_checkpoint=np.arange(1, len(group) + 1)))
    return pd.concat(ranked_groups, ignore_index=True) if ranked_groups else pd.DataFrame()


def build_rankings(evaluation_dir: Path, model_metadata_path: Path, output_dir: Path, baseline_presets: list[str]) -> None:
    metrics_path = evaluation_dir / "metrics_per_scene.csv"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Expected evaluator output not found: {metrics_path}")
    metrics = read_csv_allow_empty(metrics_path)
    model_metadata = read_csv_allow_empty(model_metadata_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = summarize_by_model(metrics, model_metadata)
    summary.to_csv(output_dir / "ranked_configs.csv", index=False)
    if "checkpoint_name" in summary.columns and "iou_mean" in summary.columns:
        add_rank_within_checkpoint(summary).to_csv(output_dir / "ranked_configs_by_checkpoint.csv", index=False)

    deltas = []
    for baseline in baseline_presets:
        delta = paired_deltas_against_baseline(metrics, model_metadata, baseline_preset=baseline, metric="iou")
        if delta.empty:
            continue
        delta.to_csv(output_dir / f"paired_deltas_vs_{baseline.replace('/', '_')}.csv", index=False)
        deltas.append(delta)
    if deltas:
        pd.concat(deltas, ignore_index=True).to_csv(output_dir / "paired_deltas_all_baselines.csv", index=False)

    tta_gains = summarize_tta_gains(summary)
    if not tta_gains.empty:
        tta_gains.to_csv(output_dir / "tta_gain_summary.csv", index=False)

    write_json(output_dir / "ranking_metadata.json", {
        "created_at": utc_now_iso(),
        "evaluation_dir": str(evaluation_dir),
        "model_metadata_csv": str(model_metadata_path),
        "metrics_per_scene_csv": str(metrics_path),
        "baseline_presets": baseline_presets,
        "primary_sort_metric": "iou_mean",
    })


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
            candidate_paths.update(scene_dir.glob("*_probability_water.memmap.sum.float32.memmap"))
            candidate_paths.update(scene_dir.glob("*_probability_water.memmap.weight.float32.memmap"))
            candidate_paths.update(scene_dir.glob("*_probability_water_tta_*.memmap"))
            candidate_paths.update(scene_dir.glob("*_probability_water_tta_*.memmap.sum.float32.memmap"))
            candidate_paths.update(scene_dir.glob("*_probability_water_tta_*.memmap.weight.float32.memmap"))
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run an inference-config sweep, evaluate it, and write ranked config summaries.")
    parser.add_argument("--sweep-config", default="configs/inference_sweep_indonesia_config_benchmark_v1.yaml")
    parser.add_argument("--evaluation-template", default="configs/evaluation/indonesia_inference_config_benchmark_template.yaml")
    parser.add_argument("--sweep-dir", default=None, help="Existing sweep directory to evaluate. If provided, do not run a new inference sweep.")
    parser.add_argument("--evaluation-dir", default=None, help="Existing evaluation output directory to summarize. If provided, do not run evaluator.")
    parser.add_argument("--skip-inference", action="store_true", help="Do not run inference; require --sweep-dir.")
    parser.add_argument("--skip-evaluation", action="store_true", help="Generate evaluation config only; do not run evaluator or ranking.")
    parser.add_argument("--cleanup-heavy-outputs", action="store_true", help="Delete probability GeoTIFFs/memmaps after successful ranking.")
    parser.add_argument("--cleanup-dry-run", action="store_true", help="List cleanup targets without deleting them.")
    parser.add_argument("--baseline-preset", action="append", default=None, help="Preset name for paired deltas. Can be provided multiple times.")
    parser.add_argument("--log-commands", action="store_true", help="Write inference/evaluation stdout/stderr logs.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sweep_config = Path(args.sweep_config)
    evaluation_template = Path(args.evaluation_template)
    baseline_presets = args.baseline_preset or DEFAULT_BASELINE_PRESETS
    if args.skip_inference and not args.sweep_dir:
        raise ValueError("--skip-inference requires --sweep-dir")

    inference_log_path = None
    copied_inference_log_path = None
    if args.sweep_dir:
        sweep_dir = Path(args.sweep_dir)
    else:
        sweep_dir, manifest_path, logs_dir = run_compatible_inference_sweeps(sweep_config, log_commands=args.log_commands)
        inference_log_path = logs_dir if logs_dir is not None else None
    if args.sweep_dir:
        if not sweep_dir.exists():
            raise FileNotFoundError(f"Sweep directory does not exist: {sweep_dir}")
        manifest_path = sweep_dir / "sweep_manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Sweep manifest not found: {manifest_path}")

    generated_eval_config, model_metadata_path = generate_evaluation_config(
        template_path=evaluation_template,
        manifest_path=manifest_path,
        output_path=sweep_dir / "generated_evaluation_config.yaml",
    )

    evaluation_log_path = None
    if args.evaluation_dir:
        evaluation_dir = Path(args.evaluation_dir)
    elif args.skip_evaluation:
        evaluation_dir = None
    else:
        evaluation_log_path = (sweep_dir / "evaluation.log") if args.log_commands else None
        evaluation_dir = run_evaluation(generated_eval_config, log_path=evaluation_log_path)

    if evaluation_dir is not None:
        summary_dir = sweep_dir / "benchmark_summary"
        build_rankings(evaluation_dir, model_metadata_path, summary_dir, baseline_presets)
        if args.cleanup_heavy_outputs or args.cleanup_dry_run:
            cleanup_heavy_outputs(manifest_path, dry_run=args.cleanup_dry_run).to_csv(summary_dir / "cleanup_heavy_outputs.csv", index=False)

    write_json(sweep_dir / "benchmark_wrapper_metadata.json", {
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
        "inference_log": str(inference_log_path) if inference_log_path else None,
        "evaluation_log": str(evaluation_log_path) if evaluation_log_path else None,
    })

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
