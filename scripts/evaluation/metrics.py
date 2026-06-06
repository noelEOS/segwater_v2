from __future__ import annotations

from typing import Any, Iterable

import numpy as np
import pandas as pd

METRIC_NAMES = ["oa", "f1", "precision", "recall", "iou", "mcc"]
CONFUSION_COUNT_NAMES = ["tn", "fp", "fn", "tp"]
IDENTITY_COLUMNS = ["evaluation_id", "s1_id", "s2_id", "model_name", "prediction_type", "inference_mode", "reference_path", "prediction_path"]
PIXEL_COUNT_COLUMNS = ["valid_pixels", "reference_water_pixels", "prediction_water_pixels"]


def _flat_binary(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    true = np.asarray(y_true).astype(np.uint8).ravel()
    pred = np.asarray(y_pred).astype(np.uint8).ravel()
    if true.shape != pred.shape:
        raise ValueError(f"y_true/y_pred shape mismatch: y_true={true.shape}, y_pred={pred.shape}")
    return true, pred


def confusion_counts(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, int]:
    true, pred = _flat_binary(y_true, y_pred)
    true = true.astype(bool)
    pred = pred.astype(bool)
    return {
        "tn": int((~true & ~pred).sum()),
        "fp": int((~true & pred).sum()),
        "fn": int((true & ~pred).sum()),
        "tp": int((true & pred).sum()),
    }


def _safe_div(numerator: float, denominator: float, zero_value: float = 0.0) -> float:
    return float(numerator / denominator) if denominator else float(zero_value)


def compute_binary_metrics(y_true: np.ndarray, y_pred: np.ndarray, include_counts: bool = False) -> dict[str, float | int]:
    counts = confusion_counts(y_true, y_pred)
    tn, fp, fn, tp = counts["tn"], counts["fp"], counts["fn"], counts["tp"]
    total = tn + fp + fn + tp
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2.0 * precision * recall, precision + recall)
    iou = _safe_div(tp, tp + fp + fn)
    oa = _safe_div(tp + tn, total, zero_value=np.nan)
    mcc_den = np.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    mcc = _safe_div((tp * tn) - (fp * fn), mcc_den)
    metrics: dict[str, float | int] = {"oa": oa, "f1": f1, "precision": precision, "recall": recall, "iou": iou, "mcc": mcc}
    return {**counts, **metrics} if include_counts else metrics


def summarize_metric_groups(df_metrics: pd.DataFrame, group_cols: str | Iterable[str], metrics: list[str] | None = None) -> pd.DataFrame:
    if df_metrics.empty:
        return pd.DataFrame()
    metrics = metrics or METRIC_NAMES
    group_cols = [group_cols] if isinstance(group_cols, str) else list(group_cols)
    rows: list[dict[str, Any]] = []
    for group_key, group in df_metrics.groupby(group_cols, sort=False, dropna=False):
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        row: dict[str, Any] = {col: value for col, value in zip(group_cols, group_key)}
        row["n_scenes"] = int(len(group))
        for metric in metrics:
            values = group[metric].dropna()
            row[f"{metric}_mean"] = float(values.mean()) if len(values) else np.nan
            row[f"{metric}_std"] = float(values.std(ddof=1)) if len(values) > 1 else np.nan
            row[f"{metric}_min"] = float(values.min()) if len(values) else np.nan
            row[f"{metric}_max"] = float(values.max()) if len(values) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def deterministic_model_pair_summary(df_metrics: pd.DataFrame, metrics: list[str] | None = None) -> pd.DataFrame:
    if df_metrics.empty:
        return pd.DataFrame()
    metrics = metrics or METRIC_NAMES
    base_cols = [col for col in IDENTITY_COLUMNS + PIXEL_COUNT_COLUMNS + CONFUSION_COUNT_NAMES if col in df_metrics.columns]
    rows: list[dict[str, Any]] = []
    for _, source_row in df_metrics.iterrows():
        row = {col: source_row[col] for col in base_cols}
        row.update({"estimate_type": "deterministic_full_pixel", "n_bootstraps": np.nan, "sample_size": np.nan, "available_valid_pixels": source_row.get("valid_pixels", np.nan)})
        for metric in metrics:
            row[f"{metric}_mean"] = source_row.get(metric, np.nan)
            row[f"{metric}_std"] = np.nan
            row[f"{metric}_ci_lower"] = np.nan
            row[f"{metric}_ci_upper"] = np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_bootstrap_samples(df_bootstrap: pd.DataFrame, metrics: list[str] | None = None) -> pd.DataFrame:
    if df_bootstrap.empty:
        return pd.DataFrame()
    metrics = metrics or METRIC_NAMES
    group_cols = [col for col in IDENTITY_COLUMNS if col in df_bootstrap.columns]
    if "evaluation_id" not in group_cols:
        raise ValueError("df_bootstrap must contain an evaluation_id column")
    summary: list[dict[str, Any]] = []
    for group_key, group in df_bootstrap.groupby(group_cols, sort=False, dropna=False):
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        row = {col: value for col, value in zip(group_cols, group_key)}
        row.update({"estimate_type": "bootstrap_pixel_level", "n_bootstraps": int(len(group)), "sample_size": int(group["sample_size"].iloc[0]), "available_valid_pixels": int(group["available_valid_pixels"].iloc[0])})
        for metric in metrics:
            values = group[metric].to_numpy()
            row[f"{metric}_mean"] = float(np.mean(values))
            row[f"{metric}_std"] = float(np.std(values, ddof=1))
            row[f"{metric}_ci_lower"] = float(np.percentile(values, 2.5))
            row[f"{metric}_ci_upper"] = float(np.percentile(values, 97.5))
        summary.append(row)
    return pd.DataFrame(summary)
