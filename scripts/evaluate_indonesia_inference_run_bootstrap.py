import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

from evaluation.metrics import METRIC_NAMES, compute_binary_metrics, summarize_bootstrap_samples, summarize_model_from_pair_means
from inference_overlap_utils import load_overlap_reference_and_prediction


DEFAULT_S1_ID_REGEX = r"(S1_\d{8}_\d{6}_\d+_\d+_\d+)"
DEFAULT_S2_ID_REGEX = r"(S2_\d{8}_\d{6})"
SUPPORTED_SPATIAL_POLICIES = {"evaluate_geospatial_overlap"}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def timestamp_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bootstrap evaluation for one inference run with probability maps.")
    parser.add_argument(
        "config",
        nargs="?",
        default="configs/evaluation/indonesia_inference_run_bootstrap_semarang.yaml",
        help="Path to the YAML config file.",
    )
    parser.add_argument("--run-dir", default=None, help="Override evaluation.run_dir from the config.")
    parser.add_argument("--run-name", default=None, help="Override output.run_name from the config.")
    parser.add_argument("--threshold", type=float, default=None, help="Override evaluation.prediction.threshold.")
    parser.add_argument(
        "--comparison",
        choices=["greater_than", "greater_equal"],
        default=None,
        help="Override evaluation.prediction.comparison.",
    )
    return parser.parse_args()


def load_config(config_path: Path) -> dict[str, Any]:
    return OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)


def apply_cli_overrides(cfg: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    cfg = dict(cfg)
    cfg["evaluation"] = dict(cfg["evaluation"])
    cfg["evaluation"]["prediction"] = dict(cfg["evaluation"].get("prediction", {}))
    cfg["output"] = dict(cfg.get("output", {}))

    if args.run_dir is not None:
        cfg["evaluation"]["run_dir"] = args.run_dir
    if args.run_name is not None:
        cfg["output"]["run_name"] = args.run_name
    if args.threshold is not None:
        cfg["evaluation"]["prediction"]["threshold"] = args.threshold
    if args.comparison is not None:
        cfg["evaluation"]["prediction"]["comparison"] = args.comparison
    return cfg


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=False), encoding="utf-8")


def write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config=OmegaConf.create(payload), f=str(path))


def extract_id(filename: str, regex: str) -> str | None:
    match = re.search(regex, filename)
    return match.group(1) if match else None


def extract_s1_id(path_or_name: str, regex: str = DEFAULT_S1_ID_REGEX) -> str:
    return extract_id(Path(path_or_name).name, regex) or ""


def extract_s2_id(path_or_name: str, regex: str = DEFAULT_S2_ID_REGEX) -> str:
    return extract_id(Path(path_or_name).name, regex) or ""


def infer_inference_mode(value: str) -> str:
    lower_value = value.lower()
    if "large_crop_only" in lower_value:
        return "large_crop_only"
    if "native224" in lower_value:
        return "native224"
    if "whole" in lower_value:
        return "whole"
    return ""


def normalize_prediction_type(prediction_type: str | None) -> str:
    normalized = str(prediction_type or "probability_map").lower()
    if normalized in {"probability", "probability_map", "prob_map"}:
        return "probability_map"
    raise ValueError(f"This evaluator only supports prediction.type='probability_map'. Received: {prediction_type}")


def resolve_spatial_policy(evaluation: dict[str, Any]) -> str:
    spatial_policy = evaluation.get("spatial_policy")
    if spatial_policy is None:
        legacy_alignment_policy = evaluation.get("alignment_policy", "evaluate_overlap")
        if legacy_alignment_policy == "evaluate_overlap":
            spatial_policy = "evaluate_geospatial_overlap"
        else:
            spatial_policy = legacy_alignment_policy

    if spatial_policy not in SUPPORTED_SPATIAL_POLICIES:
        raise ValueError(
            "evaluate_indonesia_inference_run_bootstrap.py only supports "
            f"spatial_policy in {sorted(SUPPORTED_SPATIAL_POLICIES)}. Received: {spatial_policy}"
        )
    return str(spatial_policy)


def resolve_pair_path(pair: dict[str, Any], new_key: str, legacy_key: str) -> str:
    path = pair.get(new_key) or pair.get(legacy_key)
    if not path:
        raise ValueError(f"Each file pair must define {new_key} or {legacy_key}.")
    return str(path)


def resolve_run_dir(evaluation: dict[str, Any]) -> str:
    run_dir = evaluation.get("run_dir")
    if run_dir:
        return str(run_dir)

    model_runs = evaluation.get("models") or evaluation.get("model_runs")
    if model_runs and len(model_runs) == 1:
        _, model_info = next(iter(model_runs.items()))
        if isinstance(model_info, dict):
            candidate = model_info.get("run_dir") or model_info.get("path") or model_info.get("dir")
            if candidate:
                return str(candidate)
        return str(model_info)

    raise ValueError("Config must define evaluation.run_dir, or one model entry with run_dir/path/dir.")


def resolve_model_name(evaluation: dict[str, Any], run_dir: str) -> str:
    prediction_cfg = evaluation.get("prediction", {})
    configured = evaluation.get("model_name") or prediction_cfg.get("model_name")
    if configured:
        return str(configured)

    model_runs = evaluation.get("models") or evaluation.get("model_runs")
    if model_runs and len(model_runs) == 1:
        return str(next(iter(model_runs.keys())))

    return Path(run_dir).name or "probability_map_model"


def resolve_inference_mode(evaluation: dict[str, Any], model_name: str, run_dir: str) -> str:
    prediction_cfg = evaluation.get("prediction", {})
    return str(
        evaluation.get("inference_mode")
        or prediction_cfg.get("inference_mode")
        or infer_inference_mode(model_name)
        or infer_inference_mode(run_dir)
    )


def resolve_prediction_path(run_dir: str, s1_id: str, prediction_cfg: dict[str, Any]) -> Path:
    path_template = prediction_cfg.get("path_template", "{run_dir}/{s1_id}/{s1_id}_probability_water.tif")
    return Path(str(path_template).format(run_dir=run_dir, s1_id=s1_id))


def bootstrap_pair(
    row_context: dict[str, Any],
    y_true_all: np.ndarray,
    y_pred_all: np.ndarray,
    sample_size: int,
    n_bootstraps: int,
    rng: np.random.Generator,
) -> list[dict[str, Any]]:
    evaluation_id = row_context["evaluation_id"]
    n_pixels = len(y_true_all)
    if n_pixels != len(y_pred_all):
        raise ValueError(f"{evaluation_id}: y_true/y_pred length mismatch")
    if sample_size > n_pixels:
        raise ValueError(f"{evaluation_id}: sample_size ({sample_size}) > available pixels ({n_pixels})")
    if n_pixels == 0:
        raise ValueError(f"{evaluation_id}: no valid pixels available after masking")

    rows = []
    for bootstrap_idx in range(n_bootstraps):
        selected = rng.choice(n_pixels, sample_size, replace=True)
        rows.append({
            **row_context,
            "bootstrap_idx": bootstrap_idx,
            "sample_size": sample_size,
            "available_valid_pixels": n_pixels,
            **compute_binary_metrics(y_true_all[selected], y_pred_all[selected], include_counts=True),
        })
    return rows


def resolve_output_dir(cfg: dict[str, Any], config_path: Path) -> Path:
    output_cfg = cfg.get("output", {})
    output_root = Path(output_cfg.get("root", "outputs/evaluation/indonesia_inference_run_bootstrap"))
    run_name = output_cfg.get("run_name") or Path(config_path).stem
    run_id = f"{run_name}__{timestamp_id()}" if bool(output_cfg.get("add_timestamp", True)) else run_name
    return output_root / run_id


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    cfg = apply_cli_overrides(load_config(config_path), args)
    evaluation = cfg["evaluation"]

    sample_size = int(evaluation.get("sample_size", 400))
    n_bootstraps = int(evaluation.get("n_bootstraps", 1000))
    seed = int(evaluation.get("seed", 42))

    reference_cfg = evaluation.get("reference", {})
    prediction_cfg = evaluation.get("prediction", {})
    valid_mask_cfg = evaluation.get("valid_mask", {})

    prediction_type = normalize_prediction_type(prediction_cfg.get("type", "probability_map"))
    spatial_policy = resolve_spatial_policy(evaluation)
    reference_water_values = list(reference_cfg.get("water_values", evaluation.get("reference_water_values", [1])))
    reference_nodata_values = reference_cfg.get("nodata_values", evaluation.get("reference_nodata_values", [255]))
    valid_mask_path = valid_mask_cfg.get("path", evaluation.get("valid_mask_path", None))
    valid_mask_value = valid_mask_cfg.get("value", evaluation.get("valid_mask_value", 1))
    resolution_atol = float(evaluation.get("resolution_atol", 1.0e-12))

    s1_id_regex = evaluation.get("s1_id_regex", DEFAULT_S1_ID_REGEX)
    s2_id_regex = evaluation.get("s2_id_regex", DEFAULT_S2_ID_REGEX)
    probability_threshold = float(prediction_cfg.get("threshold", 0.5))
    probability_comparison = prediction_cfg.get("comparison", "greater_than")
    run_dir = resolve_run_dir(evaluation)
    model_name = resolve_model_name(evaluation, run_dir)
    inference_mode = resolve_inference_mode(evaluation, model_name, run_dir)
    file_pairs = evaluation["file_pairs"]

    output_dir = resolve_output_dir(cfg, config_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_yaml(output_dir / "config.yaml", cfg)

    print(f"Config: {config_path}")
    print(f"Output directory: {output_dir}")
    print(f"Run directory: {run_dir}")
    print(f"Pairs: {len(file_pairs)}")
    print(f"Model name: {model_name}")
    print(f"Prediction type: {prediction_type}")
    print(f"Spatial policy: {spatial_policy}")
    print(f"Bootstraps per pair: {n_bootstraps}")
    print(f"Sample size: {sample_size}")
    print(f"Seed: {seed}")
    print(f"Probability threshold: {probability_comparison} {probability_threshold}")
    print(f"External valid mask: {valid_mask_path if valid_mask_path else 'None'}")

    started_at = utc_now_iso()
    rng = np.random.default_rng(seed)
    all_rows: list[dict[str, Any]] = []
    scene_metadata: list[dict[str, Any]] = []
    spatial_rows: list[dict[str, Any]] = []

    for pair_idx, pair in enumerate(file_pairs, start=1):
        evaluation_id = pair.get("evaluation_id") or pair.get("name", f"PAIR_{pair_idx}")
        reference_path = resolve_pair_path(pair, "reference_path", "s2_reference_path")
        s1_id = pair.get("s1_id") or extract_s1_id(reference_path, regex=s1_id_regex)
        s2_id = pair.get("s2_id") or extract_s2_id(reference_path, regex=s2_id_regex)
        if not s1_id:
            raise ValueError(f"{evaluation_id}: could not resolve s1_id from pair config or reference path")

        prediction_path = resolve_prediction_path(run_dir, s1_id, prediction_cfg)
        print(f"[{pair_idx}/{len(file_pairs)}] {evaluation_id}")

        if not prediction_path.exists():
            raise FileNotFoundError(f"Missing probability prediction: {prediction_path}")

        row_context = {
            "evaluation_id": evaluation_id,
            "s1_id": s1_id,
            "s2_id": s2_id,
            "model_name": model_name,
            "prediction_type": prediction_type,
            "inference_mode": inference_mode,
            "reference_path": reference_path,
            "prediction_path": str(prediction_path),
        }

        y_true_all, y_pred_all, diagnostics = load_overlap_reference_and_prediction(
            reference_path=reference_path,
            prediction_path=str(prediction_path),
            reference_water_values=reference_water_values,
            reference_nodata_values=reference_nodata_values,
            probability_threshold=probability_threshold,
            probability_comparison=probability_comparison,
            resolution_atol=resolution_atol,
            valid_mask_path=valid_mask_path,
            valid_mask_value=valid_mask_value,
        )

        scene_metadata.append({
            **row_context,
            "valid_pixels": int(len(y_true_all)),
            "reference_water_pixels": int(y_true_all.sum()),
            "prediction_water_pixels": int(y_pred_all.sum()),
        })
        spatial_rows.append({**row_context, **diagnostics})
        all_rows.extend(bootstrap_pair(row_context, y_true_all, y_pred_all, sample_size, n_bootstraps, rng))

    df_bootstrap = pd.DataFrame(all_rows)
    df_bootstrap_summary = summarize_bootstrap_samples(df_bootstrap, METRIC_NAMES)
    df_scene_metadata = pd.DataFrame(scene_metadata)
    df_spatial = pd.DataFrame(spatial_rows)

    identity_cols = ["evaluation_id", "s1_id", "s2_id", "model_name", "prediction_type", "inference_mode", "reference_path", "prediction_path"]
    df_model_pair_summary = df_bootstrap_summary.merge(df_scene_metadata, on=identity_cols, how="left") if not df_bootstrap_summary.empty else pd.DataFrame()
    df_metrics_summary = summarize_model_from_pair_means(df_model_pair_summary, METRIC_NAMES)

    samples_parquet = output_dir / "bootstrap_samples.parquet"
    samples_csv = output_dir / "bootstrap_samples.csv"
    bootstrap_summary_csv = output_dir / "bootstrap_summary.csv"
    model_pair_summary_csv = output_dir / "model_pair_summary.csv"
    metrics_summary_csv = output_dir / "metrics_summary.csv"
    scene_metadata_csv = output_dir / "scene_metadata.csv"
    spatial_csv = output_dir / "spatial_diagnostics.csv"

    legacy_pair_metadata_csv = output_dir / "pair_metadata.csv"
    legacy_overlap_csv = output_dir / "overlap_diagnostics.csv"

    df_bootstrap.to_parquet(samples_parquet, index=False)
    df_bootstrap.to_csv(samples_csv, index=False)
    df_bootstrap_summary.to_csv(bootstrap_summary_csv, index=False)
    df_model_pair_summary.to_csv(model_pair_summary_csv, index=False)
    df_metrics_summary.to_csv(metrics_summary_csv, index=False)
    df_scene_metadata.to_csv(scene_metadata_csv, index=False)
    df_spatial.to_csv(spatial_csv, index=False)

    df_scene_metadata.to_csv(legacy_pair_metadata_csv, index=False)
    df_spatial.to_csv(legacy_overlap_csv, index=False)

    metadata = {
        "script": "scripts/evaluate_indonesia_inference_run_bootstrap.py",
        "config_path": str(config_path),
        "started_at": started_at,
        "finished_at": utc_now_iso(),
        "output_dir": str(output_dir),
        "run_dir": run_dir,
        "valid_mask_path": valid_mask_path,
        "valid_mask_value": valid_mask_value,
        "sample_size": sample_size,
        "n_bootstraps": n_bootstraps,
        "seed": seed,
        "num_pairs": len(file_pairs),
        "model_name": model_name,
        "inference_mode": inference_mode,
        "spatial_policy": spatial_policy,
        "alignment_policy": evaluation.get("alignment_policy", None),
        "resolution_atol": resolution_atol,
        "prediction_type": prediction_type,
        "probability_threshold": probability_threshold,
        "probability_comparison": probability_comparison,
        "metrics": METRIC_NAMES,
        "bootstrap_type": "pixel_level_bootstrap",
        "model_summary_type": "macro_average_across_pair_means",
        "cli_overrides": {
            "run_dir": args.run_dir,
            "run_name": args.run_name,
            "threshold": args.threshold,
            "comparison": args.comparison,
        },
        "outputs": {
            "effective_config_yaml": str(output_dir / "config.yaml"),
            "bootstrap_samples_parquet": str(samples_parquet),
            "bootstrap_samples_csv": str(samples_csv),
            "bootstrap_summary_csv": str(bootstrap_summary_csv),
            "model_pair_summary_csv": str(model_pair_summary_csv),
            "metrics_summary_csv": str(metrics_summary_csv),
            "scene_metadata_csv": str(scene_metadata_csv),
            "spatial_diagnostics_csv": str(spatial_csv),
            "legacy_pair_metadata_csv": str(legacy_pair_metadata_csv),
            "legacy_overlap_diagnostics_csv": str(legacy_overlap_csv),
        },
    }
    write_json(output_dir / "run_metadata.json", metadata)

    print("Evaluation complete")
    print(f"Bootstrap samples Parquet: {samples_parquet}")
    print(f"Bootstrap samples CSV: {samples_csv}")
    print(f"Bootstrap summary: {bootstrap_summary_csv}")
    print(f"Model-pair summary: {model_pair_summary_csv}")
    print(f"Metrics summary: {metrics_summary_csv}")
    print(f"Scene metadata: {scene_metadata_csv}")
    print(f"Spatial diagnostics CSV: {spatial_csv}")


if __name__ == "__main__":
    main()
