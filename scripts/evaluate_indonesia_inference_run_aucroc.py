import json
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from sklearn.metrics import average_precision_score, roc_auc_score

from evaluation.metrics import METRIC_NAMES, compute_binary_metrics
from inference_overlap_utils import load_overlap_reference_and_score, threshold_probability


AUC_METRIC_NAMES = ["roc_auc", "average_precision"]
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
        legacy_alignment_policy = evaluation.get("alignment_policy", "evaluate_overlap")
        spatial_policy = "evaluate_geospatial_overlap" if legacy_alignment_policy == "evaluate_overlap" else legacy_alignment_policy

    if spatial_policy not in SUPPORTED_SPATIAL_POLICIES:
        raise ValueError(
            "evaluate_indonesia_inference_run_aucroc.py only supports "
            f"spatial_policy in {sorted(SUPPORTED_SPATIAL_POLICIES)}. Received: {spatial_policy}"
        )
    return str(spatial_policy)


def resolve_reference_files(reference_dir: str, reference_glob: str) -> list[Path]:
    return sorted(Path(reference_dir).glob(reference_glob))


def resolve_model_entries(evaluation: dict[str, Any]) -> dict[str, dict[str, str]]:
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


def resolve_prediction_path(run_dir: str, s1_id: str, prediction_cfg: dict[str, Any]) -> Path:
    path_template = prediction_cfg.get("path_template", "{run_dir}/{s1_id}/{s1_id}_probability_water.tif")
    return Path(str(path_template).format(run_dir=run_dir, s1_id=s1_id))


def resolve_output_dir(cfg: dict[str, Any], config_path: Path) -> Path:
    output_cfg = cfg.get("output", {})
    output_root = Path(output_cfg.get("root", "outputs/evaluation/indonesia_inference_run_aucroc"))
    run_name = output_cfg.get("run_name") or Path(config_path).stem
    run_id = f"{run_name}__{timestamp_id()}" if bool(output_cfg.get("add_timestamp", True)) else run_name
    return output_root / run_id


def compute_auc_metrics(y_true: np.ndarray, y_score: np.ndarray) -> dict[str, float | str]:
    n_water = int(y_true.sum())
    n_land = int(len(y_true) - n_water)
    if n_water == 0 or n_land == 0:
        return {"roc_auc": np.nan, "average_precision": np.nan, "auc_status": "undefined_single_class_reference"}

    return {
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "average_precision": float(average_precision_score(y_true, y_score)),
        "auc_status": "success",
    }


def summarize_metric_values_by_model(df: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    group_cols = [col for col in ["model_name", "prediction_type", "inference_mode"] if col in df.columns]
    rows: list[dict[str, Any]] = []
    for group_key, group in df.groupby(group_cols, sort=False, dropna=False):
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        row: dict[str, Any] = {col: value for col, value in zip(group_cols, group_key)}
        row["estimate_type"] = "macro_average_across_pairs"
        row["n_pairs"] = int(group["evaluation_id"].nunique())
        if "auc_status" in group.columns:
            row["n_success_pairs"] = int((group["auc_status"] == "success").sum())
        if "valid_pixels" in group.columns:
            row["valid_pixels_total"] = int(group["valid_pixels"].sum())
        if "reference_water_pixels" in group.columns:
            row["reference_water_pixels_total"] = int(group["reference_water_pixels"].sum())
        if "reference_land_pixels" in group.columns:
            row["reference_land_pixels_total"] = int(group["reference_land_pixels"].sum())

        for metric in metrics:
            values = group[metric].dropna().to_numpy() if metric in group.columns else np.array([])
            row[f"{metric}_mean"] = float(np.mean(values)) if len(values) else np.nan
            row[f"{metric}_std_across_pairs"] = float(np.std(values, ddof=1)) if len(values) > 1 else np.nan
            row[f"{metric}_ci_lower"] = float(np.percentile(values, 2.5)) if len(values) else np.nan
            row[f"{metric}_ci_upper"] = float(np.percentile(values, 97.5)) if len(values) else np.nan
            row[f"{metric}_min_pair"] = float(np.min(values)) if len(values) else np.nan
            row[f"{metric}_max_pair"] = float(np.max(values)) if len(values) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def resolve_threshold_sweep_config(evaluation: dict[str, Any]) -> dict[str, Any]:
    cfg = dict(evaluation.get("threshold_sweep", {}))
    cfg.setdefault("enabled", False)
    cfg.setdefault("comparison", "greater_than")
    cfg.setdefault("optimize_metrics", ["iou", "f1", "mcc"])
    return cfg


def generate_thresholds(threshold_cfg: dict[str, Any]) -> list[float]:
    if "values" in threshold_cfg and threshold_cfg["values"] is not None:
        return [float(value) for value in threshold_cfg["values"]]

    start = float(threshold_cfg.get("start", 0.0))
    stop = float(threshold_cfg.get("stop", 1.0))
    step = float(threshold_cfg.get("step", 0.01))
    if step <= 0:
        raise ValueError("threshold_sweep.step must be > 0")
    if stop < start:
        raise ValueError("threshold_sweep.stop must be >= threshold_sweep.start")

    n_steps = int(np.floor((stop - start) / step))
    thresholds = [start + idx * step for idx in range(n_steps + 1)]
    if thresholds[-1] < stop and not np.isclose(thresholds[-1], stop):
        thresholds.append(stop)
    return [float(np.round(value, 10)) for value in thresholds]


def compute_threshold_sweep_rows(
    row_context: dict[str, Any],
    y_true: np.ndarray,
    y_score: np.ndarray,
    thresholds: list[float],
    comparison: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    valid_pixels = int(len(y_true))
    reference_water_pixels = int(y_true.sum())
    reference_land_pixels = int(valid_pixels - reference_water_pixels)

    for threshold in thresholds:
        y_pred = threshold_probability(y_score, threshold, comparison)
        rows.append({
            **row_context,
            "threshold": threshold,
            "comparison": comparison,
            "valid_pixels": valid_pixels,
            "reference_water_pixels": reference_water_pixels,
            "reference_land_pixels": reference_land_pixels,
            "prediction_water_pixels": int(y_pred.sum()),
            **compute_binary_metrics(y_true=y_true, y_pred=y_pred, include_counts=True),
        })
    return rows


def summarize_threshold_sweep(df_sweep: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    if df_sweep.empty:
        return pd.DataFrame()

    group_cols = ["model_name", "prediction_type", "inference_mode", "threshold", "comparison"]
    rows: list[dict[str, Any]] = []
    for group_key, group in df_sweep.groupby(group_cols, sort=False, dropna=False):
        row: dict[str, Any] = {col: value for col, value in zip(group_cols, group_key)}
        row["estimate_type"] = "macro_average_across_pairs_at_threshold"
        row["n_pairs"] = int(group["evaluation_id"].nunique())
        row["valid_pixels_total"] = int(group["valid_pixels"].sum())
        row["reference_water_pixels_total"] = int(group["reference_water_pixels"].sum())
        row["reference_land_pixels_total"] = int(group["reference_land_pixels"].sum())
        row["prediction_water_pixels_total"] = int(group["prediction_water_pixels"].sum())

        for metric in metrics:
            values = group[metric].dropna().to_numpy()
            row[f"{metric}_mean"] = float(np.mean(values)) if len(values) else np.nan
            row[f"{metric}_std_across_pairs"] = float(np.std(values, ddof=1)) if len(values) > 1 else np.nan
            row[f"{metric}_ci_lower"] = float(np.percentile(values, 2.5)) if len(values) else np.nan
            row[f"{metric}_ci_upper"] = float(np.percentile(values, 97.5)) if len(values) else np.nan
            row[f"{metric}_min_pair"] = float(np.min(values)) if len(values) else np.nan
            row[f"{metric}_max_pair"] = float(np.max(values)) if len(values) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def select_best_thresholds(df_threshold_summary: pd.DataFrame, optimize_metrics: list[str]) -> pd.DataFrame:
    if df_threshold_summary.empty:
        return pd.DataFrame()

    group_cols = ["model_name", "prediction_type", "inference_mode"]
    rows: list[dict[str, Any]] = []
    for group_key, group in df_threshold_summary.groupby(group_cols, sort=False, dropna=False):
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        base = {col: value for col, value in zip(group_cols, group_key)}
        for metric in optimize_metrics:
            metric_col = f"{metric}_mean"
            if metric_col not in group.columns:
                continue
            valid_group = group.dropna(subset=[metric_col])
            if valid_group.empty:
                continue
            best_row = valid_group.sort_values([metric_col, "threshold"], ascending=[False, True]).iloc[0]
            rows.append({
                **base,
                "optimize_metric": metric,
                "best_threshold": float(best_row["threshold"]),
                "comparison": best_row["comparison"],
                "best_metric_mean": float(best_row[metric_col]),
                "best_metric_ci_lower": float(best_row.get(f"{metric}_ci_lower", np.nan)),
                "best_metric_ci_upper": float(best_row.get(f"{metric}_ci_upper", np.nan)),
                "best_metric_std_across_pairs": float(best_row.get(f"{metric}_std_across_pairs", np.nan)),
                "n_pairs": int(best_row["n_pairs"]),
            })
    return pd.DataFrame(rows)


def compute_loo_threshold_evaluation(df_sweep: pd.DataFrame, optimize_metrics: list[str]) -> pd.DataFrame:
    """For each held-out scene, select the best threshold on the remaining N-1 scenes, then evaluate on the held-out scene.

    Output mirrors threshold_sweep.csv columns, plus optimize_metric and loo_selected_threshold.
    """
    if df_sweep.empty:
        return pd.DataFrame()

    model_cols = ["model_name", "prediction_type", "inference_mode"]
    rows: list[dict[str, Any]] = []

    for model_key, model_group in df_sweep.groupby(model_cols, sort=False, dropna=False):
        if not isinstance(model_key, tuple):
            model_key = (model_key,)
        evaluation_ids = model_group["evaluation_id"].unique()

        if len(evaluation_ids) < 2:
            continue

        for held_out_id in evaluation_ids:
            train_group = model_group[model_group["evaluation_id"] != held_out_id]
            held_out_group = model_group[model_group["evaluation_id"] == held_out_id]
            train_by_threshold = train_group.groupby("threshold", sort=False)

            for optimize_metric in optimize_metrics:
                if optimize_metric not in train_group.columns:
                    continue

                train_means = (
                    train_by_threshold[optimize_metric]
                    .mean()
                    .reset_index()
                    .rename(columns={optimize_metric: "_mean"})
                )
                if train_means.empty:
                    continue

                best_threshold = float(
                    train_means.sort_values(["_mean", "threshold"], ascending=[False, True]).iloc[0]["threshold"]
                )

                held_at_best = held_out_group[np.isclose(held_out_group["threshold"], best_threshold)]
                if held_at_best.empty:
                    continue

                rows.append({
                    **held_at_best.iloc[0].to_dict(),
                    "optimize_metric": optimize_metric,
                    "loo_selected_threshold": best_threshold,
                })

    return pd.DataFrame(rows)


def summarize_loo_threshold_evaluation(df_loo: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    """Macro-average LOO metrics per model x optimize_metric.

    Output mirrors threshold_summary.csv columns, with optimize_metric in place of threshold.
    """
    if df_loo.empty:
        return pd.DataFrame()

    group_cols = ["model_name", "prediction_type", "inference_mode", "optimize_metric", "comparison"]
    rows: list[dict[str, Any]] = []
    for group_key, group in df_loo.groupby(group_cols, sort=False, dropna=False):
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        row: dict[str, Any] = {col: value for col, value in zip(group_cols, group_key)}
        row["estimate_type"] = "loo_macro_average"
        row["n_pairs"] = int(group["evaluation_id"].nunique())
        row["loo_selected_threshold_mean"] = float(group["loo_selected_threshold"].mean())
        row["loo_selected_threshold_std"] = float(group["loo_selected_threshold"].std(ddof=1)) if len(group) > 1 else np.nan
        if "valid_pixels" in group.columns:
            row["valid_pixels_total"] = int(group["valid_pixels"].sum())
        if "reference_water_pixels" in group.columns:
            row["reference_water_pixels_total"] = int(group["reference_water_pixels"].sum())
        if "reference_land_pixels" in group.columns:
            row["reference_land_pixels_total"] = int(group["reference_land_pixels"].sum())
        if "prediction_water_pixels" in group.columns:
            row["prediction_water_pixels_total"] = int(group["prediction_water_pixels"].sum())

        for metric in metrics:
            if metric not in group.columns:
                continue
            values = group[metric].dropna().to_numpy()
            row[f"{metric}_mean"] = float(np.mean(values)) if len(values) else np.nan
            row[f"{metric}_std_across_pairs"] = float(np.std(values, ddof=1)) if len(values) > 1 else np.nan
            row[f"{metric}_ci_lower"] = float(np.percentile(values, 2.5)) if len(values) else np.nan
            row[f"{metric}_ci_upper"] = float(np.percentile(values, 97.5)) if len(values) else np.nan
            row[f"{metric}_min_pair"] = float(np.min(values)) if len(values) else np.nan
            row[f"{metric}_max_pair"] = float(np.max(values)) if len(values) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    config_path = Path(sys.argv[1] if len(sys.argv) > 1 else "configs/evaluation/indonesia_inference_run_aucroc_semarang.yaml")
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

    prediction_cfg = evaluation.get("prediction", {})
    prediction_type = normalize_prediction_type(prediction_cfg.get("type", "probability_map"))
    spatial_policy = resolve_spatial_policy(evaluation)
    resolution_atol = float(evaluation.get("resolution_atol", 1.0e-12))
    missing_prediction_policy = evaluation.get("missing_prediction_policy", "record_and_continue")
    if missing_prediction_policy not in {"record_and_continue", "fail"}:
        raise ValueError("missing_prediction_policy must be 'record_and_continue' or 'fail'.")

    threshold_cfg = resolve_threshold_sweep_config(evaluation)
    threshold_sweep_enabled = bool(threshold_cfg.get("enabled", False))
    thresholds = generate_thresholds(threshold_cfg) if threshold_sweep_enabled else []
    threshold_comparison = str(threshold_cfg.get("comparison", "greater_than"))
    optimize_metrics = [str(metric) for metric in threshold_cfg.get("optimize_metrics", ["iou", "f1", "mcc"])]

    model_entries = resolve_model_entries(evaluation)
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
    print(f"External valid mask: {valid_mask_path if valid_mask_path else 'None'}")
    print(f"Threshold sweep: {'enabled' if threshold_sweep_enabled else 'disabled'}")
    if threshold_sweep_enabled:
        print(f"Thresholds: {len(thresholds)} values from {min(thresholds):.4f} to {max(thresholds):.4f}")

    started_at = utc_now_iso()
    auc_rows: list[dict[str, Any]] = []
    threshold_rows: list[dict[str, Any]] = []
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
                y_true, y_score, diagnostics = load_overlap_reference_and_score(
                    reference_path=str(reference_path),
                    prediction_path=str(prediction_path),
                    reference_water_values=reference_water_values,
                    reference_nodata_values=reference_nodata_values,
                    resolution_atol=resolution_atol,
                    valid_mask_path=valid_mask_path,
                    valid_mask_value=valid_mask_value,
                )

                reference_water_pixels = int(y_true.sum())
                valid_pixels = int(len(y_true))
                reference_land_pixels = int(valid_pixels - reference_water_pixels)
                score_min = float(np.nanmin(y_score)) if len(y_score) else np.nan
                score_max = float(np.nanmax(y_score)) if len(y_score) else np.nan
                score_mean = float(np.nanmean(y_score)) if len(y_score) else np.nan

                row_context = {
                    "evaluation_id": evaluation_id,
                    "s1_id": s1_id,
                    "s2_id": s2_id,
                    "model_name": model_name,
                    "prediction_type": prediction_type,
                    "inference_mode": inference_mode,
                    "reference_file": reference_path.name,
                    "reference_path": str(reference_path),
                    "prediction_path": str(prediction_path),
                }

                auc_rows.append({
                    **row_context,
                    "valid_pixels": valid_pixels,
                    "reference_water_pixels": reference_water_pixels,
                    "reference_land_pixels": reference_land_pixels,
                    "score_min": score_min,
                    "score_max": score_max,
                    "score_mean": score_mean,
                    **compute_auc_metrics(y_true, y_score),
                })

                if threshold_sweep_enabled:
                    threshold_rows.extend(compute_threshold_sweep_rows(row_context, y_true, y_score, thresholds, threshold_comparison))

                spatial_rows.append({**row_context, **diagnostics})
                manifest_rows.append({**base_manifest, "status": "success"})
            except Exception as exc:
                manifest_rows.append({**base_manifest, "status": "error", "error_message": str(exc)})
                raise

    df_auc = pd.DataFrame(auc_rows)
    df_summary = summarize_metric_values_by_model(df_auc, AUC_METRIC_NAMES)
    df_manifest = pd.DataFrame(manifest_rows)
    df_spatial = pd.DataFrame(spatial_rows)
    df_threshold_sweep = pd.DataFrame(threshold_rows)
    df_threshold_summary = summarize_threshold_sweep(df_threshold_sweep, METRIC_NAMES) if threshold_sweep_enabled else pd.DataFrame()
    df_threshold_best = select_best_thresholds(df_threshold_summary, optimize_metrics) if threshold_sweep_enabled else pd.DataFrame()
    df_loo = compute_loo_threshold_evaluation(df_threshold_sweep, optimize_metrics) if threshold_sweep_enabled else pd.DataFrame()
    df_loo_summary = summarize_loo_threshold_evaluation(df_loo, METRIC_NAMES) if threshold_sweep_enabled else pd.DataFrame()

    auc_per_pair_csv = output_dir / "auc_per_pair.csv"
    metrics_summary_csv = output_dir / "metrics_summary.csv"
    manifest_csv = output_dir / "evaluation_manifest.csv"
    spatial_csv = output_dir / "spatial_diagnostics.csv"
    threshold_sweep_csv = output_dir / "threshold_sweep.csv"
    threshold_summary_csv = output_dir / "threshold_summary.csv"
    threshold_best_summary_csv = output_dir / "threshold_best_summary.csv"
    loo_evaluation_csv = output_dir / "loo_threshold_evaluation.csv"
    loo_summary_csv = output_dir / "loo_threshold_summary.csv"

    df_auc.to_csv(auc_per_pair_csv, index=False)
    df_summary.to_csv(metrics_summary_csv, index=False)
    df_manifest.to_csv(manifest_csv, index=False)
    df_spatial.to_csv(spatial_csv, index=False)
    if threshold_sweep_enabled:
        df_threshold_sweep.to_csv(threshold_sweep_csv, index=False)
        df_threshold_summary.to_csv(threshold_summary_csv, index=False)
        df_threshold_best.to_csv(threshold_best_summary_csv, index=False)
        df_loo.to_csv(loo_evaluation_csv, index=False)
        df_loo_summary.to_csv(loo_summary_csv, index=False)

    status_counts = df_manifest["status"].value_counts(dropna=False).to_dict() if not df_manifest.empty else {}
    auc_status_counts = df_auc["auc_status"].value_counts(dropna=False).to_dict() if not df_auc.empty and "auc_status" in df_auc.columns else {}
    outputs = {
        "auc_per_pair_csv": str(auc_per_pair_csv),
        "metrics_summary_csv": str(metrics_summary_csv),
        "evaluation_manifest_csv": str(manifest_csv),
        "spatial_diagnostics_csv": str(spatial_csv),
    }
    if threshold_sweep_enabled:
        outputs.update({
            "threshold_sweep_csv": str(threshold_sweep_csv),
            "threshold_summary_csv": str(threshold_summary_csv),
            "threshold_best_summary_csv": str(threshold_best_summary_csv),
            "loo_evaluation_csv": str(loo_evaluation_csv),
            "loo_summary_csv": str(loo_summary_csv),
        })

    metadata = {
        "script": "scripts/evaluate_indonesia_inference_run_aucroc.py",
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
        "metrics": AUC_METRIC_NAMES,
        "model_summary_type": "macro_average_across_pair_auc_values",
        "missing_prediction_policy": missing_prediction_policy,
        "threshold_sweep": {
            "enabled": threshold_sweep_enabled,
            "num_thresholds": len(thresholds),
            "comparison": threshold_comparison,
            "optimize_metrics": optimize_metrics,
        },
        "status_counts": status_counts,
        "auc_status_counts": auc_status_counts,
        "outputs": outputs,
    }
    write_json(output_dir / "run_metadata.json", metadata)

    print("AUC evaluation complete")
    print(f"AUC per pair: {auc_per_pair_csv}")
    print(f"Metrics summary: {metrics_summary_csv}")
    print(f"Manifest: {manifest_csv}")
    print(f"Spatial diagnostics CSV: {spatial_csv}")
    if threshold_sweep_enabled:
        print(f"Threshold sweep: {threshold_sweep_csv}")
        print(f"Threshold summary: {threshold_summary_csv}")
        print(f"Threshold best summary (in-sample): {threshold_best_summary_csv}")
        print(f"LOO threshold evaluation: {loo_evaluation_csv}")
        print(f"LOO threshold summary: {loo_summary_csv}")
    print(f"Status counts: {status_counts}")
    print(f"AUC status counts: {auc_status_counts}")


if __name__ == "__main__":
    main()
