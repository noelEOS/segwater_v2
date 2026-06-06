import json
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import rasterio
from omegaconf import OmegaConf

from evaluation.metrics import METRIC_NAMES, compute_binary_metrics, summarize_bootstrap_samples, summarize_model_from_pair_means
from evaluation_alignment import validate_triplet_alignment


DEFAULT_S1_ID_REGEX = r"(S1_\d{8}_\d{6}_\d+_\d+_\d+)"
DEFAULT_S2_ID_REGEX = r"(S2_\d{8}_\d{6})"
SUPPORTED_SPATIAL_POLICIES = {"strict_grid_match", "assume_aligned"}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def timestamp_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def load_config(config_path: Path) -> dict[str, Any]:
    return OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=False), encoding="utf-8")


def load_raster_array(path: str) -> np.ndarray:
    with rasterio.open(path) as src:
        return src.read(1)


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
    normalized = str(prediction_type or "binary_mask").lower()
    if normalized in {"binary", "mask", "binary_mask"}:
        return "binary_mask"
    raise ValueError(f"This evaluator only supports prediction.type='binary_mask'. Received: {prediction_type}")


def resolve_spatial_policy(evaluation: dict[str, Any]) -> str:
    spatial_policy = evaluation.get("spatial_policy")
    if spatial_policy is None:
        spatial_policy = "strict_grid_match" if bool(evaluation.get("check_alignment", True)) else "assume_aligned"

    if spatial_policy not in SUPPORTED_SPATIAL_POLICIES:
        raise ValueError(
            "evaluate_indonesia_s1_s2_bootstrap.py only supports "
            f"spatial_policy in {sorted(SUPPORTED_SPATIAL_POLICIES)}. Received: {spatial_policy}"
        )
    return str(spatial_policy)


def resolve_pair_path(pair: dict[str, Any], new_key: str, legacy_key: str) -> str:
    path = pair.get(new_key) or pair.get(legacy_key)
    if not path:
        raise ValueError(f"Each file pair must define {new_key} or {legacy_key}.")
    return str(path)


def resolve_model_name(evaluation: dict[str, Any], file_pairs: list[dict[str, Any]]) -> str:
    prediction_cfg = evaluation.get("prediction", {})
    configured = evaluation.get("model_name") or prediction_cfg.get("model_name")
    if configured:
        return str(configured)

    if file_pairs:
        first_prediction_path = file_pairs[0].get("prediction_path") or file_pairs[0].get("s1_prediction_path")
        if first_prediction_path:
            return Path(str(first_prediction_path)).parent.name
    return "binary_mask_model"


def resolve_inference_mode(evaluation: dict[str, Any], model_name: str) -> str:
    prediction_cfg = evaluation.get("prediction", {})
    return str(evaluation.get("inference_mode") or prediction_cfg.get("inference_mode") or infer_inference_mode(model_name))


def get_valid_indices(valid_mask_path: str, valid_value: int | float = 1) -> tuple[np.ndarray, np.ndarray]:
    return np.where(load_raster_array(valid_mask_path) == valid_value)


def apply_optional_nodata_filter(
    y_ref: np.ndarray,
    y_pred: np.ndarray,
    reference_nodata_values: list[int | float] | None,
    prediction_nodata_values: list[int | float] | None,
) -> tuple[np.ndarray, np.ndarray]:
    keep = np.ones(y_ref.shape, dtype=bool)
    if reference_nodata_values:
        keep &= ~np.isin(y_ref, reference_nodata_values)
    if prediction_nodata_values:
        keep &= ~np.isin(y_pred, prediction_nodata_values)
    return y_ref[keep], y_pred[keep]


def extract_valid_binary_pixels(
    reference_path: str,
    prediction_path: str,
    valid_indices: tuple[np.ndarray, np.ndarray],
    reference_water_values: list[int | float],
    prediction_water_values: list[int | float],
    reference_nodata_values: list[int | float] | None = None,
    prediction_nodata_values: list[int | float] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    reference = load_raster_array(reference_path)[valid_indices]
    prediction = load_raster_array(prediction_path)[valid_indices]
    if reference.shape != prediction.shape:
        raise ValueError(f"Reference/prediction valid-pixel shape mismatch: reference={reference.shape}, prediction={prediction.shape}")

    reference, prediction = apply_optional_nodata_filter(reference, prediction, reference_nodata_values, prediction_nodata_values)
    y_true = np.isin(reference, reference_water_values).astype(np.uint8)
    y_pred = np.isin(prediction, prediction_water_values).astype(np.uint8)
    return y_true, y_pred


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
    output_root = Path(output_cfg.get("root", "outputs/evaluation/indonesia_s1_s2_bootstrap"))
    run_name = output_cfg.get("run_name") or Path(config_path).stem
    run_id = f"{run_name}__{timestamp_id()}" if bool(output_cfg.get("add_timestamp", True)) else run_name
    return output_root / run_id


def main() -> None:
    config_path = Path(sys.argv[1] if len(sys.argv) > 1 else "configs/evaluation/indonesia_s1_s2_bootstrap_semarang.yaml")
    cfg = load_config(config_path)
    evaluation = cfg["evaluation"]

    valid_mask_cfg = evaluation.get("valid_mask", {})
    valid_mask_path = valid_mask_cfg.get("path", evaluation.get("valid_mask_path", None))
    if valid_mask_path is None:
        raise ValueError("Config must define evaluation.valid_mask.path or evaluation.valid_mask_path.")
    valid_mask_value = valid_mask_cfg.get("value", evaluation.get("valid_mask_value", 1))

    sample_size = int(evaluation.get("sample_size", 400))
    n_bootstraps = int(evaluation.get("n_bootstraps", 1000))
    seed = int(evaluation.get("seed", 42))

    spatial_policy = resolve_spatial_policy(evaluation)
    check_alignment = spatial_policy == "strict_grid_match"
    transform_atol = float(evaluation.get("transform_atol", 1.0e-9))

    reference_cfg = evaluation.get("reference", {})
    prediction_cfg = evaluation.get("prediction", {})
    prediction_type = normalize_prediction_type(prediction_cfg.get("type", "binary_mask"))

    s1_id_regex = evaluation.get("s1_id_regex", DEFAULT_S1_ID_REGEX)
    s2_id_regex = evaluation.get("s2_id_regex", DEFAULT_S2_ID_REGEX)
    reference_water_values = list(reference_cfg.get("water_values", evaluation.get("reference_water_values", [1])))
    prediction_water_values = list(prediction_cfg.get("water_values", evaluation.get("prediction_water_values", [1])))
    reference_nodata_values = reference_cfg.get("nodata_values", evaluation.get("reference_nodata_values", [255]))
    prediction_nodata_values = prediction_cfg.get("nodata_values", evaluation.get("prediction_nodata_values", None))
    file_pairs = evaluation["file_pairs"]

    model_name = resolve_model_name(evaluation, file_pairs)
    inference_mode = resolve_inference_mode(evaluation, model_name)

    output_dir = resolve_output_dir(cfg, config_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(config_path, output_dir / "config.yaml")

    print(f"Config: {config_path}")
    print(f"Output directory: {output_dir}")
    print(f"Pairs: {len(file_pairs)}")
    print(f"Model name: {model_name}")
    print(f"Prediction type: {prediction_type}")
    print(f"Spatial policy: {spatial_policy}")
    print(f"Bootstraps per pair: {n_bootstraps}")
    print(f"Sample size: {sample_size}")
    print(f"Seed: {seed}")
    print(f"Transform tolerance: {transform_atol:.12g}")

    started_at = utc_now_iso()
    rng = np.random.default_rng(seed)
    valid_indices = get_valid_indices(valid_mask_path, valid_mask_value)
    all_rows: list[dict[str, Any]] = []
    scene_metadata: list[dict[str, Any]] = []
    spatial_rows: list[dict[str, Any]] = []

    for pair_idx, pair in enumerate(file_pairs, start=1):
        evaluation_id = pair.get("evaluation_id") or pair.get("name", f"PAIR_{pair_idx}")
        reference_path = resolve_pair_path(pair, "reference_path", "s2_reference_path")
        prediction_path = resolve_pair_path(pair, "prediction_path", "s1_prediction_path")
        s1_id = pair.get("s1_id") or extract_s1_id(prediction_path, regex=s1_id_regex) or extract_s1_id(reference_path, regex=s1_id_regex)
        s2_id = pair.get("s2_id") or extract_s2_id(reference_path, regex=s2_id_regex)
        print(f"[{pair_idx}/{len(file_pairs)}] {evaluation_id}")

        row_context = {
            "evaluation_id": evaluation_id,
            "s1_id": s1_id,
            "s2_id": s2_id,
            "model_name": model_name,
            "prediction_type": prediction_type,
            "inference_mode": inference_mode,
            "reference_path": reference_path,
            "prediction_path": prediction_path,
        }

        if check_alignment:
            for diagnostic in validate_triplet_alignment(reference_path, prediction_path, valid_mask_path, transform_atol):
                spatial_rows.append({**row_context, **diagnostic})

        y_true_all, y_pred_all = extract_valid_binary_pixels(
            reference_path,
            prediction_path,
            valid_indices,
            reference_water_values,
            prediction_water_values,
            reference_nodata_values,
            prediction_nodata_values,
        )
        scene_metadata.append({
            **row_context,
            "valid_pixels": int(len(y_true_all)),
            "reference_water_pixels": int(y_true_all.sum()),
            "prediction_water_pixels": int(y_pred_all.sum()),
        })
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
    legacy_alignment_csv = output_dir / "alignment_diagnostics.csv"

    df_bootstrap.to_parquet(samples_parquet, index=False)
    df_bootstrap.to_csv(samples_csv, index=False)
    df_bootstrap_summary.to_csv(bootstrap_summary_csv, index=False)
    df_model_pair_summary.to_csv(model_pair_summary_csv, index=False)
    df_metrics_summary.to_csv(metrics_summary_csv, index=False)
    df_scene_metadata.to_csv(scene_metadata_csv, index=False)
    df_spatial.to_csv(spatial_csv, index=False)

    df_scene_metadata.to_csv(legacy_pair_metadata_csv, index=False)
    df_spatial.to_csv(legacy_alignment_csv, index=False)

    metadata = {
        "script": "scripts/evaluate_indonesia_s1_s2_bootstrap.py",
        "config_path": str(config_path),
        "started_at": started_at,
        "finished_at": utc_now_iso(),
        "output_dir": str(output_dir),
        "valid_mask_path": valid_mask_path,
        "valid_mask_value": valid_mask_value,
        "sample_size": sample_size,
        "n_bootstraps": n_bootstraps,
        "seed": seed,
        "num_pairs": len(file_pairs),
        "model_name": model_name,
        "inference_mode": inference_mode,
        "prediction_type": prediction_type,
        "metrics": METRIC_NAMES,
        "bootstrap_type": "pixel_level_bootstrap",
        "spatial_policy": spatial_policy,
        "check_alignment": check_alignment,
        "transform_atol": transform_atol,
        "model_summary_type": "macro_average_across_pair_means",
        "outputs": {
            "bootstrap_samples_parquet": str(samples_parquet),
            "bootstrap_samples_csv": str(samples_csv),
            "bootstrap_summary_csv": str(bootstrap_summary_csv),
            "model_pair_summary_csv": str(model_pair_summary_csv),
            "metrics_summary_csv": str(metrics_summary_csv),
            "scene_metadata_csv": str(scene_metadata_csv),
            "spatial_diagnostics_csv": str(spatial_csv),
            "legacy_pair_metadata_csv": str(legacy_pair_metadata_csv),
            "legacy_alignment_diagnostics_csv": str(legacy_alignment_csv),
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
