import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from sklearn.metrics import accuracy_score, f1_score, jaccard_score, matthews_corrcoef, precision_score, recall_score

from inference_overlap_utils import load_overlap_reference_and_prediction


METRIC_NAMES = ["oa", "f1", "precision", "recall", "iou", "mcc"]


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


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "oa": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "iou": jaccard_score(y_true, y_pred, zero_division=0),
        "mcc": matthews_corrcoef(y_true, y_pred),
    }


def bootstrap_pair(pair_name: str, y_true_all: np.ndarray, y_pred_all: np.ndarray, sample_size: int, n_bootstraps: int, rng: np.random.Generator) -> list[dict[str, Any]]:
    n_pixels = len(y_true_all)
    if n_pixels != len(y_pred_all):
        raise ValueError(f"{pair_name}: y_true/y_pred length mismatch")
    if sample_size > n_pixels:
        raise ValueError(f"{pair_name}: sample_size ({sample_size}) > available pixels ({n_pixels})")
    if n_pixels == 0:
        raise ValueError(f"{pair_name}: no valid pixels available after masking")

    rows = []
    for bootstrap_idx in range(n_bootstraps):
        selected = rng.choice(n_pixels, sample_size, replace=True)
        rows.append({
            "pair": pair_name,
            "bootstrap_idx": bootstrap_idx,
            "sample_size": sample_size,
            "available_valid_pixels": n_pixels,
            **calculate_metrics(y_true_all[selected], y_pred_all[selected]),
        })
    return rows


def summarize_bootstrap(df_bootstrap: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    summary = []
    for pair_name, group in df_bootstrap.groupby("pair", sort=False):
        row: dict[str, Any] = {
            "pair": pair_name,
            "n_bootstraps": int(len(group)),
            "sample_size": int(group["sample_size"].iloc[0]),
            "available_valid_pixels": int(group["available_valid_pixels"].iloc[0]),
        }
        for metric in metrics:
            values = group[metric].to_numpy()
            row[f"{metric}_mean"] = float(np.mean(values))
            row[f"{metric}_std"] = float(np.std(values, ddof=1))
            row[f"{metric}_ci_lower"] = float(np.percentile(values, 2.5))
            row[f"{metric}_ci_upper"] = float(np.percentile(values, 97.5))
        summary.append(row)
    return pd.DataFrame(summary)


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
    reference_water_values = list(evaluation.get("reference_water_values", [1]))
    reference_nodata_values = evaluation.get("reference_nodata_values", [255])
    valid_mask_path = evaluation.get("valid_mask_path", None)
    valid_mask_value = evaluation.get("valid_mask_value", 1)
    resolution_atol = float(evaluation.get("resolution_atol", 1.0e-12))

    prediction_cfg = evaluation["prediction"]
    probability_threshold = float(prediction_cfg.get("threshold", 0.5))
    probability_comparison = prediction_cfg.get("comparison", "greater_than")
    probability_path_template = prediction_cfg.get("path_template", "{run_dir}/{s1_id}/{s1_id}_probability_water.tif")
    run_dir = evaluation["run_dir"]
    file_pairs = evaluation["file_pairs"]

    output_dir = resolve_output_dir(cfg, config_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_yaml(output_dir / "config.yaml", cfg)

    print(f"Config: {config_path}")
    print(f"Output directory: {output_dir}")
    print(f"Run directory: {run_dir}")
    print(f"Pairs: {len(file_pairs)}")
    print(f"Bootstraps per pair: {n_bootstraps}")
    print(f"Sample size: {sample_size}")
    print(f"Seed: {seed}")
    print("Alignment policy: evaluate_overlap")
    print(f"Probability threshold: {probability_comparison} {probability_threshold}")
    print(f"External valid mask: {valid_mask_path if valid_mask_path else 'None'}")

    started_at = utc_now_iso()
    rng = np.random.default_rng(seed)
    all_rows: list[dict[str, Any]] = []
    pair_metadata: list[dict[str, Any]] = []
    overlap_rows: list[dict[str, Any]] = []

    for pair_idx, pair in enumerate(file_pairs, start=1):
        pair_name = pair.get("name", f"PAIR_{pair_idx}")
        s1_id = pair["s1_id"]
        reference_path = pair["s2_reference_path"]
        prediction_path = Path(probability_path_template.format(run_dir=run_dir, s1_id=s1_id))
        print(f"[{pair_idx}/{len(file_pairs)}] {pair_name}")

        if not prediction_path.exists():
            raise FileNotFoundError(f"Missing probability prediction: {prediction_path}")

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

        pair_metadata.append({
            "pair": pair_name,
            "s1_id": s1_id,
            "s2_reference_path": reference_path,
            "prediction_path": str(prediction_path),
            "available_valid_pixels": int(len(y_true_all)),
            "reference_water_pixels": int(y_true_all.sum()),
            "prediction_water_pixels": int(y_pred_all.sum()),
        })
        overlap_rows.append({"pair": pair_name, "s1_id": s1_id, "prediction_path": str(prediction_path), **diagnostics})
        all_rows.extend(bootstrap_pair(pair_name, y_true_all, y_pred_all, sample_size, n_bootstraps, rng))

    df_bootstrap = pd.DataFrame(all_rows)
    df_summary = summarize_bootstrap(df_bootstrap, METRIC_NAMES)
    df_pair_metadata = pd.DataFrame(pair_metadata)
    df_overlap = pd.DataFrame(overlap_rows)

    samples_parquet = output_dir / "bootstrap_samples.parquet"
    samples_csv = output_dir / "bootstrap_samples.csv"
    summary_csv = output_dir / "bootstrap_summary.csv"
    pair_metadata_csv = output_dir / "pair_metadata.csv"
    overlap_csv = output_dir / "overlap_diagnostics.csv"

    df_bootstrap.to_parquet(samples_parquet, index=False)
    df_bootstrap.to_csv(samples_csv, index=False)
    df_summary.to_csv(summary_csv, index=False)
    df_pair_metadata.to_csv(pair_metadata_csv, index=False)
    df_overlap.to_csv(overlap_csv, index=False)

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
        "alignment_policy": "evaluate_overlap",
        "resolution_atol": resolution_atol,
        "prediction_type": "probability",
        "probability_threshold": probability_threshold,
        "probability_comparison": probability_comparison,
        "metrics": METRIC_NAMES,
        "bootstrap_type": "pixel_level_bootstrap",
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
            "bootstrap_summary_csv": str(summary_csv),
            "pair_metadata_csv": str(pair_metadata_csv),
            "overlap_diagnostics_csv": str(overlap_csv),
        },
    }
    write_json(output_dir / "run_metadata.json", metadata)

    print("Evaluation complete")
    print(f"Bootstrap samples Parquet: {samples_parquet}")
    print(f"Bootstrap samples CSV: {samples_csv}")
    print(f"Summary CSV: {summary_csv}")
    print(f"Pair metadata CSV: {pair_metadata_csv}")
    print(f"Overlap diagnostics: {overlap_csv}")


if __name__ == "__main__":
    main()
