import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import rasterio
from omegaconf import OmegaConf

from inference_overlap_utils import (
    assert_same_crs,
    assert_same_resolution,
    intersection_bounds,
    read_profile,
    rounded_window_from_bounds,
)


def load_config(path: Path) -> dict[str, Any]:
    return OmegaConf.to_container(OmegaConf.load(path), resolve=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare the evaluation pixel pools used by the binary-mask bootstrap "
            "and the inference-run probability bootstrap."
        )
    )
    parser.add_argument(
        "--binary-config",
        default="configs/evaluation/indonesia_s1_s2_bootstrap_semarang.yaml",
        help="YAML config for scripts/evaluate_indonesia_s1_s2_bootstrap.py",
    )
    parser.add_argument(
        "--inference-config",
        default="configs/evaluation/indonesia_inference_run_bootstrap_semarang.yaml",
        help="YAML config for scripts/evaluate_indonesia_inference_run_bootstrap.py",
    )
    parser.add_argument(
        "--run-dir",
        default=None,
        help="Optional override for evaluation.run_dir in the inference bootstrap config.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/evaluation/pixel_pool_verification",
        help="Directory where verification CSV/JSON outputs are written.",
    )
    parser.add_argument(
        "--max-mismatch-examples",
        type=int,
        default=100,
        help="Maximum number of mismatch coordinate examples to save per pair.",
    )
    return parser.parse_args()


def load_raster(path: str) -> np.ndarray:
    with rasterio.open(path) as src:
        return src.read(1)


def values_valid(values: np.ndarray, nodata_values: list[int | float] | None) -> np.ndarray:
    if nodata_values:
        return ~np.isin(values, nodata_values)
    return np.ones(values.shape, dtype=bool)


def binary_bootstrap_pixel_coords(pair: dict[str, Any], cfg: dict[str, Any]) -> np.ndarray:
    evaluation = cfg["evaluation"]
    valid_mask_path = evaluation["valid_mask_path"]
    valid_mask_value = evaluation.get("valid_mask_value", 1)
    reference_nodata_values = evaluation.get("reference_nodata_values", [255])
    prediction_nodata_values = evaluation.get("prediction_nodata_values", None)

    reference = load_raster(pair["s2_reference_path"])
    valid_mask = load_raster(valid_mask_path)
    rows, cols = np.where(valid_mask == valid_mask_value)

    ref_values = reference[rows, cols]
    keep = values_valid(ref_values, reference_nodata_values)

    if prediction_nodata_values:
        prediction = load_raster(pair["s1_prediction_path"])
        pred_values = prediction[rows, cols]
        keep &= values_valid(pred_values, prediction_nodata_values)

    return np.column_stack([rows[keep], cols[keep]]).astype(np.int64)


def inference_bootstrap_pixel_coords(pair: dict[str, Any], cfg: dict[str, Any], run_dir_override: str | None) -> tuple[np.ndarray, dict[str, Any]]:
    evaluation = cfg["evaluation"]
    reference_path = pair["s2_reference_path"]
    run_dir = run_dir_override or evaluation["run_dir"]
    prediction_cfg = evaluation["prediction"]
    path_template = prediction_cfg.get("path_template", "{run_dir}/{s1_id}/{s1_id}_probability_water.tif")
    prediction_path = Path(path_template.format(run_dir=run_dir, s1_id=pair["s1_id"]))

    if not prediction_path.exists():
        raise FileNotFoundError(f"Missing probability prediction: {prediction_path}")

    valid_mask_path = evaluation.get("valid_mask_path", None)
    valid_mask_value = evaluation.get("valid_mask_value", 1)
    reference_nodata_values = evaluation.get("reference_nodata_values", [255])
    resolution_atol = float(evaluation.get("resolution_atol", 1.0e-12))

    reference_profile = read_profile(reference_path)
    prediction_profile = read_profile(str(prediction_path))
    assert_same_crs(reference_profile, prediction_profile, "prediction")
    assert_same_resolution(reference_profile, prediction_profile, resolution_atol, "prediction")

    bounds_inputs = [reference_profile["bounds"], prediction_profile["bounds"]]
    mask_profile = None
    if valid_mask_path:
        mask_profile = read_profile(valid_mask_path)
        assert_same_crs(reference_profile, mask_profile, "valid_mask")
        assert_same_resolution(reference_profile, mask_profile, resolution_atol, "valid_mask")
        bounds_inputs.append(mask_profile["bounds"])

    overlap_bounds = intersection_bounds(*bounds_inputs)

    with rasterio.open(reference_path) as src_ref:
        ref_window = rounded_window_from_bounds(overlap_bounds, src_ref.transform)
        reference = src_ref.read(1, window=ref_window)
        row_off = int(ref_window.row_off)
        col_off = int(ref_window.col_off)

    valid = values_valid(reference, reference_nodata_values)

    if valid_mask_path:
        with rasterio.open(valid_mask_path) as src_mask:
            mask_window = rounded_window_from_bounds(overlap_bounds, src_mask.transform)
            external_mask = src_mask.read(1, window=mask_window)
        if external_mask.shape != reference.shape:
            raise ValueError(
                f"Valid mask overlap shape mismatch for {pair['name']}: "
                f"reference={reference.shape}, valid_mask={external_mask.shape}"
            )
        valid &= external_mask == valid_mask_value

    local_rows, local_cols = np.where(valid)
    full_rows = local_rows + row_off
    full_cols = local_cols + col_off
    coords = np.column_stack([full_rows, full_cols]).astype(np.int64)

    diagnostics = {
        "prediction_path": str(prediction_path),
        "overlap_shape": str(reference.shape),
        "overlap_row_off": row_off,
        "overlap_col_off": col_off,
        "valid_pixels_after_all_masks": int(len(coords)),
    }
    return coords, diagnostics


def coords_to_set(coords: np.ndarray) -> set[tuple[int, int]]:
    return set(map(tuple, coords.tolist()))


def compare_coords(binary_coords: np.ndarray, inference_coords: np.ndarray) -> dict[str, Any]:
    same_length = len(binary_coords) == len(inference_coords)
    same_order = same_length and np.array_equal(binary_coords, inference_coords)

    binary_set = coords_to_set(binary_coords)
    inference_set = coords_to_set(inference_coords)
    intersection = binary_set & inference_set
    binary_only = binary_set - inference_set
    inference_only = inference_set - binary_set

    return {
        "binary_valid_pixels": int(len(binary_coords)),
        "inference_valid_pixels": int(len(inference_coords)),
        "same_count": bool(same_length),
        "same_ordered_pixel_pool": bool(same_order),
        "same_pixel_set": bool(binary_set == inference_set),
        "intersection_pixels": int(len(intersection)),
        "binary_only_pixels": int(len(binary_only)),
        "inference_only_pixels": int(len(inference_only)),
        "binary_coverage_of_inference": float(len(intersection) / len(inference_set)) if inference_set else np.nan,
        "inference_coverage_of_binary": float(len(intersection) / len(binary_set)) if binary_set else np.nan,
    }


def first_mismatch_examples(binary_coords: np.ndarray, inference_coords: np.ndarray, max_examples: int) -> pd.DataFrame:
    binary_set = coords_to_set(binary_coords)
    inference_set = coords_to_set(inference_coords)

    rows = []
    for row, col in sorted(binary_set - inference_set)[:max_examples]:
        rows.append({"source": "binary_only", "row": row, "col": col})
    for row, col in sorted(inference_set - binary_set)[:max_examples]:
        rows.append({"source": "inference_only", "row": row, "col": col})
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    binary_cfg_path = Path(args.binary_config)
    inference_cfg_path = Path(args.inference_config)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    binary_cfg = load_config(binary_cfg_path)
    inference_cfg = load_config(inference_cfg_path)

    binary_pairs = {pair.get("name", f"PAIR_{idx + 1}"): pair for idx, pair in enumerate(binary_cfg["evaluation"]["file_pairs"])}
    inference_pairs = {pair.get("name", f"PAIR_{idx + 1}"): pair for idx, pair in enumerate(inference_cfg["evaluation"]["file_pairs"])}
    common_pair_names = [name for name in binary_pairs if name in inference_pairs]

    if not common_pair_names:
        raise ValueError("No matching pair names found between the two configs.")

    summary_rows = []
    all_mismatch_examples = []

    for pair_name in common_pair_names:
        print(f"Checking {pair_name}...")
        binary_coords = binary_bootstrap_pixel_coords(binary_pairs[pair_name], binary_cfg)
        inference_coords, diagnostics = inference_bootstrap_pixel_coords(
            inference_pairs[pair_name],
            inference_cfg,
            run_dir_override=args.run_dir,
        )
        comparison = compare_coords(binary_coords, inference_coords)
        summary_rows.append({"pair": pair_name, **comparison, **diagnostics})

        mismatch_df = first_mismatch_examples(binary_coords, inference_coords, args.max_mismatch_examples)
        if not mismatch_df.empty:
            mismatch_df.insert(0, "pair", pair_name)
            all_mismatch_examples.append(mismatch_df)

    summary = pd.DataFrame(summary_rows)
    summary_csv = output_dir / "bootstrap_pixel_pool_comparison.csv"
    summary.to_csv(summary_csv, index=False)

    if all_mismatch_examples:
        pd.concat(all_mismatch_examples, ignore_index=True).to_csv(output_dir / "bootstrap_pixel_pool_mismatch_examples.csv", index=False)

    metadata = {
        "binary_config": str(binary_cfg_path),
        "inference_config": str(inference_cfg_path),
        "run_dir_override": args.run_dir,
        "output_dir": str(output_dir),
        "num_pairs_checked": len(common_pair_names),
        "all_pairs_same_ordered_pixel_pool": bool(summary["same_ordered_pixel_pool"].all()),
        "all_pairs_same_pixel_set": bool(summary["same_pixel_set"].all()),
    }
    (output_dir / "bootstrap_pixel_pool_verification_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print(f"Wrote: {summary_csv}")
    print(summary[["pair", "binary_valid_pixels", "inference_valid_pixels", "same_ordered_pixel_pool", "same_pixel_set", "binary_only_pixels", "inference_only_pixels"]].to_string(index=False))

    if summary["same_ordered_pixel_pool"].all():
        print("Result: pixel pools are identical and in the same order. With the same seed, both scripts sample the same pixels.")
    elif summary["same_pixel_set"].all():
        print("Result: pixel sets match, but order differs. Bootstrap index samples may map to different pixel orderings.")
    else:
        print("Result: pixel pools differ. Compare the mismatch examples and overlap diagnostics before comparing metrics.")


if __name__ == "__main__":
    main()
