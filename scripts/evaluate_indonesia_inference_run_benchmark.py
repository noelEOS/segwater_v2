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

from inference_overlap_utils import calculate_iou_precision_recall, load_overlap_reference_and_prediction


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


def extract_s1_id(reference_filename: str, regex: str = DEFAULT_S1_ID_REGEX) -> str | None:
    match = re.search(regex, reference_filename)
    return match.group(1) if match else None


def resolve_reference_files(reference_dir: str, reference_glob: str) -> list[Path]:
    return sorted(Path(reference_dir).glob(reference_glob))


def resolve_output_dir(cfg: dict[str, Any], config_path: Path) -> Path:
    output_cfg = cfg.get("output", {})
    output_root = Path(output_cfg.get("root", "outputs/evaluation/indonesia_inference_run_benchmark"))
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
    config_path = Path(sys.argv[1] if len(sys.argv) > 1 else "configs/evaluation/indonesia_inference_run_benchmark_semarang.yaml")
    cfg = load_config(config_path)
    evaluation = cfg["evaluation"]

    reference_dir = evaluation["reference_dir"]
    reference_glob = evaluation.get("reference_glob", "*.tif")
    s1_id_regex = evaluation.get("s1_id_regex", DEFAULT_S1_ID_REGEX)
    reference_water_values = list(evaluation.get("reference_water_values", [1]))
    reference_nodata_values = evaluation.get("reference_nodata_values", [255])

    alignment_policy = evaluation.get("alignment_policy", "evaluate_overlap")
    if alignment_policy != "evaluate_overlap":
        raise ValueError("This inference-run evaluator only supports alignment_policy='evaluate_overlap'.")
    resolution_atol = float(evaluation.get("resolution_atol", 1.0e-12))

    prediction_cfg = evaluation["prediction"]
    probability_threshold = float(prediction_cfg.get("threshold", 0.5))
    probability_comparison = prediction_cfg.get("comparison", "greater_than")
    probability_path_template = prediction_cfg.get("path_template", "{run_dir}/{s1_id}/{s1_id}_probability_water.tif")

    model_runs = evaluation["model_runs"]
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
    print(f"Model runs: {len(model_runs)}")
    print("Alignment policy: evaluate_overlap")
    print(f"Probability threshold: {probability_comparison} {probability_threshold}")

    started_at = utc_now_iso()
    metrics_rows: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    overlap_rows: list[dict[str, Any]] = []

    for model_name, model_info in model_runs.items():
        run_dir = model_info["run_dir"] if isinstance(model_info, dict) else str(model_info)
        print(f"Processing model run: {model_name}")

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

            prediction_path = Path(probability_path_template.format(run_dir=run_dir, s1_id=s1_id))
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
                )
                metrics = calculate_iou_precision_recall(y_pred=y_pred, y_true=y_true)

                metrics_rows.append({
                    "model": model_name,
                    "s1_id": s1_id,
                    "reference_file": reference_path.name,
                    "reference_path": str(reference_path),
                    "prediction_path": str(prediction_path),
                    "valid_pixels": int(len(y_true)),
                    "reference_water_pixels": int(y_true.sum()),
                    "prediction_water_pixels": int(y_pred.sum()),
                    **metrics,
                })
                overlap_rows.append({
                    "model": model_name,
                    "s1_id": s1_id,
                    "reference_file": reference_path.name,
                    "prediction_path": str(prediction_path),
                    **diagnostics,
                })
                manifest_rows.append({**base_manifest, "status": "success"})
            except Exception as exc:
                manifest_rows.append({**base_manifest, "status": "error", "error_message": str(exc)})
                raise

    df_metrics = pd.DataFrame(metrics_rows)
    df_summary = summarize_by_model(df_metrics)
    df_manifest = pd.DataFrame(manifest_rows)
    df_overlap = pd.DataFrame(overlap_rows)

    per_scene_csv = output_dir / "per_scene_metrics.csv"
    model_summary_csv = output_dir / "model_summary.csv"
    manifest_csv = output_dir / "evaluation_manifest.csv"
    overlap_csv = output_dir / "overlap_diagnostics.csv"

    df_metrics.to_csv(per_scene_csv, index=False)
    df_summary.to_csv(model_summary_csv, index=False)
    df_manifest.to_csv(manifest_csv, index=False)
    df_overlap.to_csv(overlap_csv, index=False)

    status_counts = df_manifest["status"].value_counts(dropna=False).to_dict() if not df_manifest.empty else {}
    metadata = {
        "script": "scripts/evaluate_indonesia_inference_run_benchmark.py",
        "config_path": str(config_path),
        "started_at": started_at,
        "finished_at": utc_now_iso(),
        "output_dir": str(output_dir),
        "reference_dir": reference_dir,
        "reference_glob": reference_glob,
        "num_reference_files": len(reference_files),
        "num_model_runs": len(model_runs),
        "alignment_policy": alignment_policy,
        "resolution_atol": resolution_atol,
        "prediction_type": "probability",
        "probability_threshold": probability_threshold,
        "probability_comparison": probability_comparison,
        "metrics": METRIC_NAMES,
        "status_counts": status_counts,
        "outputs": {
            "per_scene_metrics_csv": str(per_scene_csv),
            "model_summary_csv": str(model_summary_csv),
            "evaluation_manifest_csv": str(manifest_csv),
            "overlap_diagnostics_csv": str(overlap_csv),
        },
    }
    write_json(output_dir / "run_metadata.json", metadata)

    print("Evaluation complete")
    print(f"Per-scene metrics: {per_scene_csv}")
    print(f"Model summary: {model_summary_csv}")
    print(f"Manifest: {manifest_csv}")
    print(f"Overlap diagnostics: {overlap_csv}")
    print(f"Status counts: {status_counts}")


if __name__ == "__main__":
    main()
