import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import rasterio
from omegaconf import OmegaConf
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    jaccard_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
)


METRIC_NAMES = ["oa", "f1", "precision", "recall", "iou", "mcc"]


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


def read_raster_profile(path: str) -> dict[str, Any]:
    with rasterio.open(path) as src:
        return {
            "path": path,
            "width": src.width,
            "height": src.height,
            "shape": (src.height, src.width),
            "crs": str(src.crs),
            "transform": tuple(src.transform),
            "nodata": src.nodata,
        }


def validate_raster_alignment(reference_path: str, prediction_path: str, valid_mask_path: str) -> None:
    reference = read_raster_profile(reference_path)
    prediction = read_raster_profile(prediction_path)
    valid_mask = read_raster_profile(valid_mask_path)

    for key in ["shape", "crs", "transform"]:
        values = {
            "reference": reference[key],
            "prediction": prediction[key],
            "valid_mask": valid_mask[key],
        }
        if len(set(values.values())) != 1:
            raise ValueError(
                f"Raster alignment mismatch for {key}: "
                f"reference={values['reference']}, "
                f"prediction={values['prediction']}, "
                f"valid_mask={values['valid_mask']}"
            )


def get_valid_indices(valid_mask_path: str, valid_value: int | float = 1) -> tuple[np.ndarray, np.ndarray]:
    valid_mask = load_raster_array(valid_mask_path)
    return np.where(valid_mask == valid_value)


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
        raise ValueError(
            f"Reference/prediction valid-pixel shape mismatch: "
            f"reference={reference.shape}, prediction={prediction.shape}"
        )

    reference, prediction = apply_optional_nodata_filter(
        reference,
        prediction,
        reference_nodata_values=reference_nodata_values,
        prediction_nodata_values=prediction_nodata_values,
    )

    y_true = np.isin(reference, reference_water_values).astype(np.uint8)
    y_pred = np.isin(prediction, prediction_water_values).astype(np.uint8)
    return y_true, y_pred


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "oa": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "iou": jaccard_score(y_true, y_pred, zero_division=0),
        "mcc": matthews_corrcoef(y_true, y_pred),
    }


def bootstrap_pair(
    pair_name: str,
    y_true_all: np.ndarray,
    y_pred_all: np.ndarray,
    sample_size: int,
    n_bootstraps: int,
    rng: np.random.Generator,
) -> list[dict[str, Any]]:
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
        metrics = calculate_metrics(y_true_all[selected], y_pred_all[selected])
        rows.append(
            {
                "pair": pair_name,
                "bootstrap_idx": bootstrap_idx,
                "sample_size": sample_size,
                "available_valid_pixels": n_pixels,
                **metrics,
            }
        )
    return rows


def summarize_bootstrap(df_bootstrap: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    summary_rows = []
    for pair_name, group in df_bootstrap.groupby("pair", sort=False):
        row: dict[str, Any] = {"pair": pair_name}
        row["n_bootstraps"] = int(len(group))
        row["sample_size"] = int(group["sample_size"].iloc[0])
        row["available_valid_pixels"] = int(group["available_valid_pixels"].iloc[0])
        for metric in metrics:
            values = group[metric].to_numpy()
            row[f"{metric}_mean"] = float(np.mean(values))
            row[f"{metric}_std"] = float(np.std(values, ddof=1))
            row[f"{metric}_ci_lower"] = float(np.percentile(values, 2.5))
            row[f"{metric}_ci_upper"] = float(np.percentile(values, 97.5))
        summary_rows.append(row)
    return pd.DataFrame(summary_rows)


def resolve_output_dir(cfg: dict[str, Any], config_path: Path) -> Path:
    output_cfg = cfg.get("output", {})
    output_root = Path(output_cfg.get("root", "outputs/evaluation/indonesia_s1_s2_bootstrap"))
    run_name = output_cfg.get("run_name") or Path(config_path).stem
    add_timestamp = bool(output_cfg.get("add_timestamp", True))
    run_id = f"{run_name}__{timestamp_id()}" if add_timestamp else run_name
    return output_root / run_id


def main() -> None:
    config_path = Path(sys.argv[1] if len(sys.argv) > 1 else "configs/evaluation/indonesia_s1_s2_bootstrap_semarang.yaml")
    cfg = load_config(config_path)

    evaluation = cfg["evaluation"]
    valid_mask_path = evaluation["valid_mask_path"]
    valid_mask_value = evaluation.get("valid_mask_value", 1)
    sample_size = int(evaluation.get("sample_size", 400))
    n_bootstraps = int(evaluation.get("n_bootstraps", 1000))
    seed = int(evaluation.get("seed", 42))
    check_alignment = bool(evaluation.get("check_alignment", True))

    reference_water_values = list(evaluation.get("reference_water_values", [1]))
    prediction_water_values = list(evaluation.get("prediction_water_values", [1]))
    reference_nodata_values = evaluation.get("reference_nodata_values", [255])
    prediction_nodata_values = evaluation.get("prediction_nodata_values", None)

    file_pairs = evaluation["file_pairs"]
    output_dir = resolve_output_dir(cfg, config_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(config_path, output_dir / "config.yaml")

    print(f"Config: {config_path}")
    print(f"Output directory: {output_dir}")
    print(f"Pairs: {len(file_pairs)}")
    print(f"Bootstraps per pair: {n_bootstraps}")
    print(f"Sample size: {sample_size}")
    print(f"Seed: {seed}")

    started_at = utc_now_iso()
    rng = np.random.default_rng(seed)
    valid_indices = get_valid_indices(valid_mask_path, valid_mask_value)

    all_rows: list[dict[str, Any]] = []
    pair_metadata: list[dict[str, Any]] = []

    for pair_idx, pair in enumerate(file_pairs, start=1):
        pair_name = pair.get("name", f"PAIR_{pair_idx}")
        reference_path = pair["s2_reference_path"]
        prediction_path = pair["s1_prediction_path"]

        print(f"[{pair_idx}/{len(file_pairs)}] {pair_name}")

        if check_alignment:
            validate_raster_alignment(reference_path, prediction_path, valid_mask_path)

        y_true_all, y_pred_all = extract_valid_binary_pixels(
            reference_path=reference_path,
            prediction_path=prediction_path,
            valid_indices=valid_indices,
            reference_water_values=reference_water_values,
            prediction_water_values=prediction_water_values,
            reference_nodata_values=reference_nodata_values,
            prediction_nodata_values=prediction_nodata_values,
        )

        pair_metadata.append(
            {
                "pair": pair_name,
                "s2_reference_path": reference_path,
                "s1_prediction_path": prediction_path,
                "available_valid_pixels": int(len(y_true_all)),
                "reference_water_pixels": int(y_true_all.sum()),
                "prediction_water_pixels": int(y_pred_all.sum()),
            }
        )

        all_rows.extend(
            bootstrap_pair(
                pair_name=pair_name,
                y_true_all=y_true_all,
                y_pred_all=y_pred_all,
                sample_size=sample_size,
                n_bootstraps=n_bootstraps,
                rng=rng,
            )
        )

    df_bootstrap = pd.DataFrame(all_rows)
    df_summary = summarize_bootstrap(df_bootstrap, METRIC_NAMES)
    df_pair_metadata = pd.DataFrame(pair_metadata)

    samples_parquet = output_dir / "bootstrap_samples.parquet"
    samples_csv = output_dir / "bootstrap_samples.csv"
    summary_csv = output_dir / "bootstrap_summary.csv"
    pair_metadata_csv = output_dir / "pair_metadata.csv"

    df_bootstrap.to_parquet(samples_parquet, index=False)
    df_bootstrap.to_csv(samples_csv, index=False)
    df_summary.to_csv(summary_csv, index=False)
    df_pair_metadata.to_csv(pair_metadata_csv, index=False)

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
        "metrics": METRIC_NAMES,
        "bootstrap_type": "pixel_level_bootstrap",
        "outputs": {
            "bootstrap_samples_parquet": str(samples_parquet),
            "bootstrap_samples_csv": str(samples_csv),
            "bootstrap_summary_csv": str(summary_csv),
            "pair_metadata_csv": str(pair_metadata_csv),
        },
    }
    write_json(output_dir / "run_metadata.json", metadata)

    print("Evaluation complete")
    print(f"Bootstrap samples Parquet: {samples_parquet}")
    print(f"Bootstrap samples CSV: {samples_csv}")
    print(f"Summary CSV: {summary_csv}")
    print(f"Pair metadata CSV: {pair_metadata_csv}")


if __name__ == "__main__":
    main()
