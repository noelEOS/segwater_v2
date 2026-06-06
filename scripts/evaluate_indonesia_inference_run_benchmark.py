import json
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from omegaconf import OmegaConf

from evaluation.metrics import METRIC_NAMES, compute_binary_metrics, deterministic_model_pair_summary, summarize_metric_groups
from inference_overlap_utils import load_overlap_reference_and_prediction


DEFAULT_S1_ID_REGEX = r"(S1_\d{8}_\d{6}_\d+_\d+_\d+)"
DEFAULT_S2_ID_REGEX = r"(S2_\d{8}_\d{6})"
SUPPORTED_SPATIAL_POLICIES = {"evaluate_geospatial_overlap"}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def timestamp_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def load_config(config_path: Path) -> dict[str, Any]:
    return OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=False), encoding="utf-8")


def extract_id(filename: str, regex: str) -> str | None:
    match = re.search(regex, filename)
    return match.group(1) if match else None


def extract_s1_id(reference_filename: str, regex: str = DEFAULT_S1_ID_REGEX) -> str | None:
    return extract_id(reference_filename, regex)


def extract_s2_id(reference_filename: str, regex: str = DEFAULT_S2_ID_REGEX) -> str:
    return extract_id(reference_filename, regex) or ""


def infer_inference_mode(model_name: str) -> str:
    lower_name = model_name.lower()
    if "large_crop_only" in lower_name:
        return "large_crop_only"
    if "native224" in lower_name:
        return "native224"
    if "whole" in lower_name:
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
        # Backward compatibility with the previous config key.
        legacy_alignment_policy = evaluation.get("alignment_policy", "evaluate_overlap")
        if legacy_alignment_policy == "evaluate_overlap":
            spatial_policy = "evaluate_geospatial_overlap"
        elif legacy_alignment_policy == "fail":
            spatial_policy = "strict_grid_match"
        else:
            spatial_policy = legacy_alignment_policy

    if spatial_policy not in SUPPORTED_SPATIAL_POLICIES:
        raise ValueError(
            "evaluate_indonesia_inference_run_benchmark.py only supports "
            f"spatial_policy in {sorted(SUPPORTED_SPATIAL_POLICIES)}. Received: {spatial_policy}"
        )
    return str(spatial_policy)


def resolve_model_entries(evaluation: dict[str, Any]) -> dict[str, dict[str, str]]:
    """Resolve old model_runs and new models config structures.

    Supported old format:
      model_runs:
        ModelName:
          run_dir: /path/to/run

    Supported new format:
      models:
        ModelName:
          run_dir: /path/to/run
          inference_mode: native224
    """
    raw_models = evaluation.get("models") or evaluation.get("model_runs")
    if not raw_models:
        raise ValueError("Config must define evaluation.models or evaluation.model_runs.")

    model_entries: dict[str, dict[str, str]] = {}
    for model_name, model_info in raw_models.items():
        if isinstance(model_info, dict):
            run_dir = model_info.get("run_dir") or model_info.get("path") or model_info.get("dir")
            if not run_dir:
                raise ValueError(f"Model entry for {model_name} must define run_dir, path, or dir.")
            inference_mode = model_info.get("inference_mode") or infer_inference_mode(model_name)
        else:
            run_dir = str(model_info)
            inference_mode = infer_inference_mode(model_name)

        model_entries[str(model_name)] = {
            "run_dir": str(run_dir),
            "inference_mode": str(inference_mode),
        }
    return model_entries


def resolve_reference_files(reference_dir: str, reference_glob: str) -> list[Path]:
    return sorted(Path(reference_dir).glob(reference_glob))


def resolve_prediction_path(run_dir: str, s1_id: str, prediction_cfg: dict[str, Any]) -> Path:
    path_template = prediction_cfg.get("path_template", "{run_dir}/{s1_id}/{s1_id}_probability_water.tif")
    return Path(str(path_template).format(run_dir=run_dir, s1_id=s1_id))


def resolve_output_dir(cfg: dict[str, Any], config_path: Path) -> Path:
    output_cfg = cfg.get("output", {})
    output_root = Path(output_cfg.get("root", "outputs/evaluation/indonesia_inference_run_benchmark"))
    run_name = output_cfg.get("run_name") or Path(config_path).stem
    run_id = f"{run_name}__{timestamp_id()}" if bool(output_cfg.get("add_timestamp", True)) else run_name
    return output_root / run_id


def main() -> None:
    config_path = Path(sys.argv[1] if len(sys.argv) > 1 else "configs/evaluation/indonesia_inference_run_benchmark_semarang.yaml")
    cfg = load_config(config_path)
    evaluation = cfg["evaluation"]

    reference_cfg = evaluation.get("reference", {})
    reference_dir = reference_cfg.get("dir") or evaluation.get("reference_dir")
    if reference_dir is None:
        raise ValueError("Config must define evaluation.reference.dir or evaluation.reference_dir.")
    reference_glob = reference_cfg.get("glob") or evaluation.get("reference_glob", "*.tif")
    s1_id_regex = reference_cfg.get("s1_id_regex") or evaluation.get("s1_id_regex", DEFAULT_S1_ID_REGEX)
    s2_id_regex = reference_cfg.get("s2_id_regex") or evaluation.get("s2_id_regex", DEFAULT_S2_ID_REGEX)
    reference_water_values = list(reference_cfg.get("water_values", evaluation.get("reference_water_values", [1])))
    reference_nodata_values = reference_cfg.get("nodata_values", evaluation.get("reference_nodata_values", [255]))

    valid_mask_cfg = evaluation.get("valid_mask", {})
    valid_mask_path = valid_mask_cfg.get("path", evaluation.get("valid_mask_path", None))
    valid_mask_value = valid_mask_cfg.get("value", evaluation.get("valid_mask_value", 1))

    spatial_policy = resolve_spatial_policy(evaluation)
    resolution_atol = float(evaluation.get("resolution_atol", 1.0e-12))

    prediction_cfg = evaluation.get("prediction", {})
    prediction_type = normalize_prediction_type(prediction_cfg.get("type", "probability_map"))
    probability_threshold = float(prediction_cfg.get("threshold", 0.5))
    probability_comparison = prediction_cfg.get("comparison", "greater_than")

    model_entries = resolve_model_entries(evaluation)
    missing_prediction_policy = evaluation.get("missing_prediction_policy", "record_and_continue")
    if missing_prediction_policy not in {"record_and_continue", "fail"}:
        raise ValueError("missing_prediction_policy must be 'record_and_continue' or 'fail'.")

    reference_files = resolve_reference_files(reference_dir, reference_glob)
    if not reference_files:
        raise ValueError(f"No reference files found in {reference_dir} with glob {reference_glob}")

    output_dir = resolve_output_dir(cfg, config_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(config_path, output_dir / "config.yaml")

    print(f"Config: {config_path}")
    print(f"Output directory: {output_dir}")
    print(f"Reference files: {len(reference_files)}")
    print(f"Model runs: {len(model_entries)}")
    print(f"Spatial policy: {spatial_policy}")
    print(f"Prediction type: {prediction_type}")
    print(f"Probability threshold: {probability_comparison} {probability_threshold}")
    print(f"External valid mask: {valid_mask_path if valid_mask_path else 'None'}")

    started_at = utc_now_iso()
    metrics_rows: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    spatial_rows: list[dict[str, Any]] = []

    for model_name, model_info in model_entries.items():
        run_dir = model_info["run_dir"]
        inference_mode = model_info["inference_mode"]
        print(f"Processing model run: {model_name}")

        for reference_path in reference_files:
            s1_id = extract_s1_id(reference_path.name, regex=s1_id_regex)
            s2_id = extract_s2_id(reference_path.name, regex=s2_id_regex)
            evaluation_id = s1_id or reference_path.stem

            base_manifest = {
                "evaluation_id": evaluation_id,
                "s1_id": s1_id or "",
                "s2_id": s2_id,
                "model_name": model_name,
                "prediction_type": prediction_type,
                "inference_mode": inference_mode,
                "reference_file": reference_path.name,
                "reference_path": str(reference_path),
                "prediction_path": "",
                "status": "",
                "error_message": "",
            }

            if not s1_id:
                manifest_rows.append({**base_manifest, "status": "invalid_reference_filename"})
                continue

            prediction_path = resolve_prediction_path(run_dir, s1_id, prediction_cfg)
            base_manifest["prediction_path"] = str(prediction_path)

            if not prediction_path.exists():
                manifest_rows.append({**base_manifest, "status": "missing_prediction"})
                if missing_prediction_policy == "fail":
                    raise FileNotFoundError(f"Missing prediction: {prediction_path}")
                continue

            try:
                y_true, y_pred, diagnostics = load_overlap_reference_and_prediction(
                    reference_path=str(reference_path),
                    prediction_path=str(prediction_path),
                    reference_water_values=reference_water_values,
                    reference_nodata_values=reference_nodata_values,
                    probability_threshold=probability_threshold,
                    probability_comparison=probability_comparison,
                    resolution_atol=resolution_atol,
                    valid_mask_path=valid_mask_path,
                    valid_mask_value=valid_mask_value,
                )
                metrics = compute_binary_metrics(y_true=y_true, y_pred=y_pred, include_counts=True)

                metrics_rows.append({
                    "evaluation_id": evaluation_id,
                    "s1_id": s1_id,
                    "s2_id": s2_id,
                    "model_name": model_name,
                    "prediction_type": prediction_type,
                    "inference_mode": inference_mode,
                    "reference_file": reference_path.name,
                    "reference_path": str(reference_path),
                    "prediction_path": str(prediction_path),
                    "valid_pixels": int(len(y_true)),
                    "reference_water_pixels": int(y_true.sum()),
                    "prediction_water_pixels": int(y_pred.sum()),
                    **metrics,
                })
                spatial_rows.append({
                    "evaluation_id": evaluation_id,
                    "s1_id": s1_id,
                    "s2_id": s2_id,
                    "model_name": model_name,
                    "prediction_type": prediction_type,
                    "inference_mode": inference_mode,
                    "reference_file": reference_path.name,
                    "reference_path": str(reference_path),
                    "prediction_path": str(prediction_path),
                    **diagnostics,
                })
                manifest_rows.append({**base_manifest, "status": "success"})
            except Exception as exc:
                manifest_rows.append({**base_manifest, "status": "error", "error_message": str(exc)})
                raise

    df_metrics = pd.DataFrame(metrics_rows)
    df_summary = summarize_metric_groups(df_metrics, "model_name", METRIC_NAMES)
    df_model_pair_summary = deterministic_model_pair_summary(df_metrics, METRIC_NAMES)
    df_manifest = pd.DataFrame(manifest_rows)
    df_spatial = pd.DataFrame(spatial_rows)

    metrics_per_scene_csv = output_dir / "metrics_per_scene.csv"
    metrics_summary_csv = output_dir / "metrics_summary.csv"
    model_pair_summary_csv = output_dir / "model_pair_summary.csv"
    manifest_csv = output_dir / "evaluation_manifest.csv"
    spatial_csv = output_dir / "spatial_diagnostics.csv"

    # Temporary backward-compatible aliases for existing notebooks/scripts.
    legacy_per_scene_csv = output_dir / "per_scene_metrics.csv"
    legacy_model_summary_csv = output_dir / "model_summary.csv"
    legacy_overlap_csv = output_dir / "overlap_diagnostics.csv"

    df_metrics.to_csv(metrics_per_scene_csv, index=False)
    df_summary.to_csv(metrics_summary_csv, index=False)
    df_model_pair_summary.to_csv(model_pair_summary_csv, index=False)
    df_manifest.to_csv(manifest_csv, index=False)
    df_spatial.to_csv(spatial_csv, index=False)

    df_metrics.to_csv(legacy_per_scene_csv, index=False)
    df_summary.to_csv(legacy_model_summary_csv, index=False)
    df_spatial.to_csv(legacy_overlap_csv, index=False)

    status_counts = df_manifest["status"].value_counts(dropna=False).to_dict() if not df_manifest.empty else {}
    metadata = {
        "script": "scripts/evaluate_indonesia_inference_run_benchmark.py",
        "config_path": str(config_path),
        "started_at": started_at,
        "finished_at": utc_now_iso(),
        "output_dir": str(output_dir),
        "reference_dir": reference_dir,
        "reference_glob": reference_glob,
        "valid_mask_path": valid_mask_path,
        "valid_mask_value": valid_mask_value,
        "num_reference_files": len(reference_files),
        "num_model_runs": len(model_entries),
        "spatial_policy": spatial_policy,
        "alignment_policy": evaluation.get("alignment_policy", None),
        "resolution_atol": resolution_atol,
        "prediction_type": prediction_type,
        "probability_threshold": probability_threshold,
        "probability_comparison": probability_comparison,
        "missing_prediction_policy": missing_prediction_policy,
        "metrics": METRIC_NAMES,
        "status_counts": status_counts,
        "outputs": {
            "metrics_per_scene_csv": str(metrics_per_scene_csv),
            "metrics_summary_csv": str(metrics_summary_csv),
            "model_pair_summary_csv": str(model_pair_summary_csv),
            "evaluation_manifest_csv": str(manifest_csv),
            "spatial_diagnostics_csv": str(spatial_csv),
            "legacy_per_scene_metrics_csv": str(legacy_per_scene_csv),
            "legacy_model_summary_csv": str(legacy_model_summary_csv),
            "legacy_overlap_diagnostics_csv": str(legacy_overlap_csv),
        },
    }
    write_json(output_dir / "run_metadata.json", metadata)

    print("Evaluation complete")
    print(f"Metrics per scene: {metrics_per_scene_csv}")
    print(f"Metrics summary: {metrics_summary_csv}")
    print(f"Model-pair summary: {model_pair_summary_csv}")
    print(f"Manifest: {manifest_csv}")
    print(f"Spatial diagnostics CSV: {spatial_csv}")
    print(f"Status counts: {status_counts}")


if __name__ == "__main__":
    main()
