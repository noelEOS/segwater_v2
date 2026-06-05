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

from evaluation_alignment import validate_pair_alignment


METRIC_NAMES = ["iou", "precision", "recall"]
DEFAULT_S1_ID_REGEX = r"(S1_\d{8}_\d{6}_\d+_\d+_\d+)"


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


def extract_s1_id(reference_filename: str, regex: str = DEFAULT_S1_ID_REGEX) -> str | None:
    match = re.search(regex, reference_filename)
    return match.group(1) if match else None


def calculate_iou_precision_recall(y_pred: np.ndarray, y_true: np.ndarray) -> dict[str, float]:
    pred = y_pred.flatten().astype(bool)
    true = y_true.flatten().astype(bool)
    intersection = np.logical_and(pred, true).sum()
    union = np.logical_or(pred, true).sum()
    if union == 0:
        return {"iou": np.nan, "precision": np.nan, "recall": np.nan}

    predicted_water = pred.sum()
    reference_water = true.sum()
    return {
        "iou": intersection / union,
        "precision": intersection / predicted_water if predicted_water > 0 else 0.0,
        "recall": intersection / reference_water if reference_water > 0 else 0.0,
    }


def apply_validity_filters(
    reference: np.ndarray,
    prediction: np.ndarray,
    reference_nodata_values: list[int | float] | None,
    prediction_nodata_values: list[int | float] | None,
    external_valid_mask: np.ndarray | None = None,
    external_valid_value: int | float = 1,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    valid_mask = np.ones(reference.shape, dtype=bool)
    if reference_nodata_values:
        valid_mask &= ~np.isin(reference, reference_nodata_values)
    if prediction_nodata_values:
        valid_mask &= ~np.isin(prediction, prediction_nodata_values)
    if external_valid_mask is not None:
        if external_valid_mask.shape != reference.shape:
            raise ValueError(
                f"External valid mask shape mismatch: reference={reference.shape}, valid_mask={external_valid_mask.shape}"
            )
        valid_mask &= external_valid_mask == external_valid_value

    diagnostics = {
        "valid_pixels_after_all_masks": int(valid_mask.sum()),
        "valid_pixels_after_reference_nodata": int((~np.isin(reference, reference_nodata_values)).sum()) if reference_nodata_values else int(reference.size),
        "valid_pixels_after_external_mask": int((external_valid_mask == external_valid_value).sum()) if external_valid_mask is not None else int(reference.size),
    }
    return reference[valid_mask], prediction[valid_mask], diagnostics


def evaluate_pair(
    reference_path: str,
    prediction_path: str,
    reference_water_values: list[int | float],
    prediction_water_values: list[int | float],
    reference_nodata_values: list[int | float] | None,
    prediction_nodata_values: list[int | float] | None,
    check_alignment: bool,
    transform_atol: float,
    valid_mask_path: str | None = None,
    valid_mask_value: int | float = 1,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    alignment_diagnostics = []
    if check_alignment:
        alignment_diagnostics = validate_pair_alignment(reference_path, prediction_path, transform_atol)

    reference = load_raster_array(reference_path)
    prediction = load_raster_array(prediction_path)
    if reference.shape != prediction.shape:
        raise ValueError(
            f"Shape mismatch after loading reference={reference.shape}, prediction={prediction.shape}. "
            "Official evaluation does not crop rasters."
        )

    external_valid_mask = load_raster_array(valid_mask_path) if valid_mask_path else None
    reference_valid, prediction_valid, mask_diagnostics = apply_validity_filters(
        reference,
        prediction,
        reference_nodata_values,
        prediction_nodata_values,
        external_valid_mask=external_valid_mask,
        external_valid_value=valid_mask_value,
    )
    if len(reference_valid) == 0:
        raise ValueError("No valid pixels remain after applying masks.")

    y_true = np.isin(reference_valid, reference_water_values).astype(np.uint8)
    y_pred = np.isin(prediction_valid, prediction_water_values).astype(np.uint8)
    metrics = calculate_iou_precision_recall(y_pred=y_pred, y_true=y_true)
    return {
        **metrics,
        **mask_diagnostics,
        "valid_pixels": int(len(y_true)),
        "reference_water_pixels": int(y_true.sum()),
        "prediction_water_pixels": int(y_pred.sum()),
    }, alignment_diagnostics


def resolve_reference_files(reference_dir: str, reference_glob: str) -> list[Path]:
    return sorted(Path(reference_dir).glob(reference_glob))


def resolve_output_dir(cfg: dict[str, Any], config_path: Path) -> Path:
    output_cfg = cfg.get("output", {})
    output_root = Path(output_cfg.get("root", "outputs/evaluation/indonesia_model_benchmark"))
    run_name = output_cfg.get("run_name") or Path(config_path).stem
    run_id = f"{run_name}__{timestamp_id()}" if bool(output_cfg.get("add_timestamp", True)) else run_name
    return output_root / run_id


def summarize_by_model(df_metrics: pd.DataFrame) -> pd.DataFrame:
    if df_metrics.empty:
        return pd.DataFrame()

    rows = []
    for model_name, group in df_metrics.groupby("model", sort=False):
        row: dict[str, Any] = {"model": model_name, "n_scenes": int(len(group))}
        for metric in METRIC_NAMES:
            values = group[metric].dropna()
            row[f"{metric}_mean"] = float(values.mean()) if len(values) else np.nan
            row[f"{metric}_std"] = float(values.std(ddof=1)) if len(values) > 1 else np.nan
            row[f"{metric}_min"] = float(values.min()) if len(values) else np.nan
            row[f"{metric}_max"] = float(values.max()) if len(values) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    config_path = Path(sys.argv[1] if len(sys.argv) > 1 else "configs/evaluation/indonesia_model_benchmark_semarang.yaml")
    cfg = load_config(config_path)
    evaluation = cfg["evaluation"]

    reference_dir = evaluation["reference_dir"]
    reference_glob = evaluation.get("reference_glob", "*.tif")
    s1_id_regex = evaluation.get("s1_id_regex", DEFAULT_S1_ID_REGEX)
    prediction_filename_template = evaluation.get("prediction_filename_template", "{s1_id}_mask.tif")
    model_dirs = evaluation["model_dirs"]
    reference_water_values = list(evaluation.get("reference_water_values", [1]))
    prediction_water_values = list(evaluation.get("prediction_water_values", [1]))
    reference_nodata_values = evaluation.get("reference_nodata_values", [255])
    prediction_nodata_values = evaluation.get("prediction_nodata_values", None)
    valid_mask_path = evaluation.get("valid_mask_path", None)
    valid_mask_value = evaluation.get("valid_mask_value", 1)
    transform_atol = float(evaluation.get("transform_atol", 1.0e-9))

    alignment_policy = evaluation.get("alignment_policy", "fail")
    if alignment_policy != "fail":
        raise ValueError("Only alignment_policy='fail' is supported in the official benchmark pipeline.")
    check_alignment = True

    missing_prediction_policy = evaluation.get("missing_prediction_policy", "record_and_continue")
    if missing_prediction_policy not in {"record_and_continue", "fail"}:
        raise ValueError("missing_prediction_policy must be 'record_and_continue' or 'fail'.")

    reference_files = resolve_reference_files(reference_dir, reference_glob)
    output_dir = resolve_output_dir(cfg, config_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(config_path, output_dir / "config.yaml")

    print(f"Config: {config_path}")
    print(f"Output directory: {output_dir}")
    print(f"Reference files: {len(reference_files)}")
    print(f"Models: {len(model_dirs)}")
    print("Alignment policy: fail")
    print(f"Transform tolerance: {transform_atol:.12g}")
    print(f"External valid mask: {valid_mask_path if valid_mask_path else 'None'}")

    started_at = utc_now_iso()
    metrics_rows: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    alignment_rows: list[dict[str, Any]] = []

    if not reference_files:
        raise ValueError(f"No reference files found in {reference_dir} with glob {reference_glob}")

    for model_name, model_path in model_dirs.items():
        print(f"Processing model: {model_name}")
        model_dir = Path(model_path)
        for reference_path in reference_files:
            s1_id = extract_s1_id(reference_path.name, regex=s1_id_regex)
            base_manifest = {
                "model": model_name,
                "s1_id": s1_id or "",
                "reference_file": reference_path.name,
                "reference_path": str(reference_path),
                "prediction_path": "",
                "status": "",
                "error_message": "",
            }

            if not s1_id:
                manifest_rows.append({**base_manifest, "status": "invalid_reference_filename"})
                continue

            prediction_path = model_dir / prediction_filename_template.format(s1_id=s1_id)
            base_manifest["prediction_path"] = str(prediction_path)

            if not prediction_path.exists():
                manifest_rows.append({**base_manifest, "status": "missing_prediction"})
                if missing_prediction_policy == "fail":
                    raise FileNotFoundError(f"Missing prediction: {prediction_path}")
                continue

            try:
                metrics, diagnostics = evaluate_pair(
                    str(reference_path),
                    str(prediction_path),
                    reference_water_values,
                    prediction_water_values,
                    reference_nodata_values,
                    prediction_nodata_values,
                    check_alignment,
                    transform_atol,
                    valid_mask_path=valid_mask_path,
                    valid_mask_value=valid_mask_value,
                )
                metrics_rows.append({
                    "model": model_name,
                    "s1_id": s1_id,
                    "reference_file": reference_path.name,
                    "reference_path": str(reference_path),
                    "prediction_path": str(prediction_path),
                    **metrics,
                })
                manifest_rows.append({**base_manifest, "status": "success"})
                for diagnostic in diagnostics:
                    alignment_rows.append({"model": model_name, "s1_id": s1_id, "reference_file": reference_path.name, **diagnostic})
            except Exception as exc:
                manifest_rows.append({**base_manifest, "status": "error", "error_message": str(exc)})
                raise

    df_metrics = pd.DataFrame(metrics_rows)
    df_manifest = pd.DataFrame(manifest_rows)
    df_summary = summarize_by_model(df_metrics)
    df_alignment = pd.DataFrame(alignment_rows)

    per_scene_csv = output_dir / "per_scene_metrics.csv"
    model_summary_csv = output_dir / "model_summary.csv"
    manifest_csv = output_dir / "evaluation_manifest.csv"
    alignment_csv = output_dir / "alignment_diagnostics.csv"

    df_metrics.to_csv(per_scene_csv, index=False)
    df_summary.to_csv(model_summary_csv, index=False)
    df_manifest.to_csv(manifest_csv, index=False)
    df_alignment.to_csv(alignment_csv, index=False)

    status_counts = df_manifest["status"].value_counts(dropna=False).to_dict() if not df_manifest.empty else {}
    metadata = {
        "script": "scripts/evaluate_indonesia_model_benchmark.py",
        "config_path": str(config_path),
        "started_at": started_at,
        "finished_at": utc_now_iso(),
        "output_dir": str(output_dir),
        "reference_dir": reference_dir,
        "reference_glob": reference_glob,
        "valid_mask_path": valid_mask_path,
        "valid_mask_value": valid_mask_value,
        "num_reference_files": len(reference_files),
        "num_models": len(model_dirs),
        "alignment_policy": alignment_policy,
        "transform_atol": transform_atol,
        "missing_prediction_policy": missing_prediction_policy,
        "metrics": METRIC_NAMES,
        "status_counts": status_counts,
        "outputs": {
            "per_scene_metrics_csv": str(per_scene_csv),
            "model_summary_csv": str(model_summary_csv),
            "evaluation_manifest_csv": str(manifest_csv),
            "alignment_diagnostics_csv": str(alignment_csv),
        },
    }
    write_json(output_dir / "run_metadata.json", metadata)

    print("Evaluation complete")
    print(f"Per-scene metrics: {per_scene_csv}")
    print(f"Model summary: {model_summary_csv}")
    print(f"Manifest: {manifest_csv}")
    print(f"Alignment diagnostics CSV: {alignment_csv}")
    print(f"Status counts: {status_counts}")


if __name__ == "__main__":
    main()
