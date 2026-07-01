import csv
import json
import re
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
import torch
from omegaconf import DictConfig, OmegaConf


@dataclass(frozen=True)
class InferenceOutputPaths:
    """Filesystem contract for one SegWater2 inference scene."""

    scene_id: str
    run_id: str
    root_dir: Path
    run_dir: Path
    scene_dir: Path
    probability_memmap: Path
    probability_geotiff: Path
    binary_mask_geotiff: Path
    shoreline_geojson: Path
    scene_metadata: Path
    run_config: Path
    run_manifest: Path
    run_summary: Path


def sanitize_for_path(value: str) -> str:
    """Convert arbitrary identifiers into stable, filesystem-safe path components."""
    value = str(value).strip().replace(".", "p")
    value = re.sub(r"[^A-Za-z0-9_-]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("_")
    return value or "unknown"


def utc_now_iso() -> str:
    """Return a timezone-aware UTC timestamp for metadata, not output folder names."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def get_git_commit() -> str | None:
    """Best-effort capture of the active git commit for scientific provenance."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except Exception:
        return None


def to_json_safe(value: Any) -> Any:
    """Convert common ML/Python objects into JSON-serializable values."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, DictConfig):
        return OmegaConf.to_container(value, resolve=True)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.detach().cpu().item()
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(k): to_json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_json_safe(v) for v in value]
    return value


def build_checkpoint_id(cfg: DictConfig) -> str:
    """Derive a deterministic run identifier from the model/checkpoint identity."""
    checkpoint_stem = sanitize_for_path(Path(str(cfg.inference.checkpoint_path)).stem)
    arch = sanitize_for_path(str(cfg.model.arch))
    encoder = sanitize_for_path(str(cfg.model.encoder_name))

    if checkpoint_stem in {"best_model", "model", "checkpoint", "unknown"}:
        return f"{arch}-{encoder}_{checkpoint_stem}"

    return checkpoint_stem


def prepare_output_paths(cfg: DictConfig) -> InferenceOutputPaths:
    """
    Create deterministic, provenance-aware output paths.

    Output layout:
        {root_dir}/{run_id}/{scene_id}/{scene_id}_{product}.{ext}
    """
    input_image = Path(str(cfg.inference.data.input_image))
    scene_id = sanitize_for_path(input_image.stem)

    configured_run_name = cfg.inference.output.run_name
    run_id = sanitize_for_path(configured_run_name) if configured_run_name else build_checkpoint_id(cfg)

    # Shoreline vector extension follows the configured output format (gpkg default).
    shoreline_format = str(
        OmegaConf.select(cfg, "inference.post_processing.shoreline.output_format")
        or "gpkg"
    ).lower()
    shoreline_ext = {"gpkg": "gpkg", "geojson": "geojson"}.get(shoreline_format, "gpkg")

    root_dir = Path(str(cfg.inference.output.root_dir))
    run_dir = root_dir / run_id
    scene_dir = run_dir / scene_id

    overwrite = bool(cfg.inference.output.overwrite)
    if scene_dir.exists() and not overwrite:
        raise FileExistsError(
            f"Output directory already exists and overwrite=False: {scene_dir}\n"
            "Use a different inference.output.run_name, remove the existing scene directory, "
            "or set inference.output.overwrite=True for intentional replacement."
        )

    scene_dir.mkdir(parents=True, exist_ok=True)

    return InferenceOutputPaths(
        scene_id=scene_id,
        run_id=run_id,
        root_dir=root_dir,
        run_dir=run_dir,
        scene_dir=scene_dir,
        probability_memmap=scene_dir / f"{scene_id}_probability_water.memmap",
        probability_geotiff=scene_dir / f"{scene_id}_probability_water.tif",
        binary_mask_geotiff=scene_dir / f"{scene_id}_mask_water.tif",
        shoreline_geojson=scene_dir / f"{scene_id}_shoreline.{shoreline_ext}",
        scene_metadata=scene_dir / f"{scene_id}_metadata.json",
        run_config=run_dir / "run_config.yaml",
        run_manifest=run_dir / "run_manifest.csv",
        run_summary=run_dir / "run_summary.json",
    )


def read_raster_metadata(reference_tif_path: str | Path) -> dict[str, Any]:
    """Capture source geospatial metadata for JSON provenance."""
    with rasterio.open(reference_tif_path) as src:
        return {
            "width": src.width,
            "height": src.height,
            "count": src.count,
            "crs": src.crs.to_string() if src.crs else None,
            "transform": list(src.transform),
            "bounds": list(src.bounds),
            "dtypes": list(src.dtypes),
            "nodata": src.nodata,
        }


def write_run_config_once(cfg: DictConfig, run_config_path: Path) -> None:
    """Persist the active Hydra config at run level without rewriting it every scene."""
    if not run_config_path.exists():
        run_config_path.parent.mkdir(parents=True, exist_ok=True)
        run_config_path.write_text(OmegaConf.to_yaml(cfg, resolve=True), encoding="utf-8")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_json_safe(payload), indent=2, sort_keys=False), encoding="utf-8")


def append_manifest_row(manifest_path: Path, row: dict[str, Any]) -> None:
    """Append one scene record to the run manifest CSV."""
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "run_id",
        "scene_id",
        "status",
        "input_path",
        "probability_memmap",
        "probability_geotiff",
        "binary_mask_geotiff",
        "shoreline_geojson",
        "metadata_json",
        "width",
        "height",
        "crs",
        "threshold",
        "checkpoint_path",
        "created_utc",
        "elapsed_minutes",
    ]

    safe_row = to_json_safe(row)
    write_header = not manifest_path.exists()
    with manifest_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow({field: safe_row.get(field, "") for field in fieldnames})


def build_geotiff_tags(
    cfg: DictConfig,
    scene_id: str,
    run_id: str,
    product: str,
    created_utc: str,
    git_commit: str | None,
) -> dict[str, str]:
    """Small provenance payload embedded directly into GeoTIFF tags."""
    return {
        "product": product,
        "scene_id": scene_id,
        "run_id": run_id,
        "created_utc": created_utc,
        "model_arch": str(cfg.model.arch),
        "model_encoder": str(cfg.model.encoder_name),
        "checkpoint_name": Path(str(cfg.inference.checkpoint_path)).name,
        "threshold": str(cfg.inference.post_processing.threshold),
        "normalization_method": str(cfg.inference.data.normalization.method),
        "normalization_means": json.dumps(list(cfg.inference.data.normalization.means)),
        "normalization_stds": json.dumps(list(cfg.inference.data.normalization.stds)),
        "padding_mode": str(cfg.inference.data.padding.mode),
        "segwater_git_commit": git_commit or "unknown",
    }


def build_scene_metadata(
    *,
    cfg: DictConfig,
    paths: InferenceOutputPaths,
    input_image: str | Path,
    raster_metadata: dict[str, Any],
    saved_step: Any,
    saved_miou: Any,
    device: Any,
    gpu_name: str | None,
    vram_gb: float | None,
    probability_geotiff_path: Path | None,
    binary_mask_path: Path | None,
    shoreline_path: str | Path | None,
    created_utc: str,
    elapsed_minutes: float,
    git_commit: str | None,
) -> dict[str, Any]:
    """Build the scene-level provenance record."""
    input_image = Path(str(input_image))
    return {
        "scene_id": paths.scene_id,
        "input": {
            "path": str(input_image),
            "filename": input_image.name,
            **raster_metadata,
        },
        "model": {
            "architecture": str(cfg.model.arch),
            "encoder": str(cfg.model.encoder_name),
            "in_channels": int(cfg.model.in_channels),
            "num_classes": int(cfg.model.num_classes),
            "checkpoint_path": str(cfg.inference.checkpoint_path),
            "checkpoint_name": Path(str(cfg.inference.checkpoint_path)).name,
            "checkpoint_step": saved_step,
            "checkpoint_val_miou": saved_miou,
        },
        "preprocessing": {
            "input_units": "dB",
            "normalization_enabled": bool(cfg.inference.data.normalization.enabled),
            "normalization_method": str(cfg.inference.data.normalization.method),
            "channel_order": ["VV", "VH"],
            "means": list(cfg.inference.data.normalization.means),
            "stds": list(cfg.inference.data.normalization.stds),
            "padding_mode": str(cfg.inference.data.padding.mode),
            "channel_fill_values": list(cfg.inference.data.padding.channel_fill_values),
        },
        "inference": {
            "tile_size": int(cfg.inference.data.tile_size),
            "buffer_size": int(cfg.inference.data.buffer_size),
            "model_input_size": int(cfg.inference.data.tile_size + 2 * cfg.inference.data.buffer_size),
            "batch_size": int(cfg.inference.data.batch_size),
            "num_workers": int(cfg.inference.data.num_workers),
            "precision": str(cfg.inference.data.precision),
            "device": str(device),
            "gpu_name": gpu_name,
            "gpu_vram_gb": vram_gb,
        },
        "post_processing": {
            "threshold": float(cfg.inference.post_processing.threshold),
            "extract_shoreline": bool(cfg.inference.output.extract_shoreline),
            "apply_length_filter": bool(cfg.inference.post_processing.filtering.apply_length_filter),
            "min_length_meters": float(cfg.inference.post_processing.filtering.min_length_meters),
            "keep_top_k": cfg.inference.post_processing.filtering.keep_top_k,
            "apply_simplification": bool(cfg.inference.post_processing.smoothing.apply_simplification),
            "simplify_tolerance_meters": float(cfg.inference.post_processing.smoothing.simplify_tolerance_meters),
        },
        "outputs": {
            "scene_dir": str(paths.scene_dir),
            "probability_memmap": str(paths.probability_memmap),
            "probability_geotiff": str(probability_geotiff_path) if probability_geotiff_path else None,
            "binary_mask_geotiff": str(binary_mask_path) if binary_mask_path else None,
            "shoreline_geojson": str(shoreline_path) if shoreline_path else None,
            "metadata_json": str(paths.scene_metadata),
        },
        "run": {
            "run_id": paths.run_id,
            "created_utc": created_utc,
            "elapsed_minutes": elapsed_minutes,
            "segwater_git_commit": git_commit,
            "run_config": str(paths.run_config),
            "run_manifest": str(paths.run_manifest),
        },
    }


def build_manifest_row(
    *,
    cfg: DictConfig,
    paths: InferenceOutputPaths,
    input_image: str | Path,
    raster_metadata: dict[str, Any],
    probability_geotiff_path: Path | None,
    binary_mask_path: Path | None,
    shoreline_path: str | Path | None,
    created_utc: str,
    elapsed_minutes: float,
) -> dict[str, Any]:
    """Build one row for the run-level manifest."""
    return {
        "run_id": paths.run_id,
        "scene_id": paths.scene_id,
        "status": "success",
        "input_path": str(input_image),
        "probability_memmap": str(paths.probability_memmap),
        "probability_geotiff": str(probability_geotiff_path) if probability_geotiff_path else "",
        "binary_mask_geotiff": str(binary_mask_path) if binary_mask_path else "",
        "shoreline_geojson": str(shoreline_path) if shoreline_path else "",
        "metadata_json": str(paths.scene_metadata),
        "width": raster_metadata["width"],
        "height": raster_metadata["height"],
        "crs": raster_metadata["crs"],
        "threshold": cfg.inference.post_processing.threshold,
        "checkpoint_path": str(cfg.inference.checkpoint_path),
        "created_utc": created_utc,
        "elapsed_minutes": f"{elapsed_minutes:.4f}",
    }


def build_run_summary(
    *,
    cfg: DictConfig,
    paths: InferenceOutputPaths,
    git_commit: str | None,
) -> dict[str, Any]:
    """Build the run-level summary JSON."""
    return {
        "run_id": paths.run_id,
        "checkpoint_path": str(cfg.inference.checkpoint_path),
        "model_architecture": str(cfg.model.arch),
        "model_encoder": str(cfg.model.encoder_name),
        "last_updated_utc": utc_now_iso(),
        "segwater_git_commit": git_commit,
        "run_config": str(paths.run_config),
        "run_manifest": str(paths.run_manifest),
        "root_dir": str(paths.root_dir),
    }
