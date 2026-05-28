import csv
import json
import os
import re
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
import logging

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf
import rasterio
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from src.utils.vectorizer import ShorelineVectorizer

# Import our custom pipeline modules
from src.data.inference_dataset import InferenceDataset
from src.utils.stitcher import ProbabilityStitcher
from src.models.factory import SegmentationModelFactory

# Configure basic logging level (Hydra handles the file routing automatically)
logger = logging.getLogger(__name__)


def sanitize_for_path(value: str) -> str:
    """Convert arbitrary identifiers into stable, filesystem-safe path components."""
    value = str(value).strip().replace(".", "p")
    value = re.sub(r"[^A-Za-z0-9_-]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("_")
    return value or "unknown"


def utc_now_iso() -> str:
    """Return a timezone-aware UTC timestamp for metadata, not for output folder names."""
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


def build_checkpoint_id(cfg: DictConfig) -> str:
    """Derive a deterministic run identifier from the model/checkpoint identity."""
    checkpoint_stem = sanitize_for_path(Path(str(cfg.inference.checkpoint_path)).stem)
    arch = sanitize_for_path(str(cfg.model.arch))
    encoder = sanitize_for_path(str(cfg.model.encoder_name))

    if checkpoint_stem in {"best_model", "model", "checkpoint", "unknown"}:
        return f"{arch}-{encoder}_{checkpoint_stem}"

    return checkpoint_stem


def prepare_output_paths(cfg: DictConfig) -> dict[str, Path | str]:
    """
    Create deterministic, provenance-aware output paths.

    Output layout:
        {root_dir}/{run_id}/{scene_id}/{scene_id}_{product}.{ext}
    """
    input_image = Path(str(cfg.inference.data.input_image))
    scene_id = sanitize_for_path(input_image.stem)

    configured_run_name = cfg.inference.output.run_name
    run_id = sanitize_for_path(configured_run_name) if configured_run_name else build_checkpoint_id(cfg)

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

    paths = {
        "scene_id": scene_id,
        "run_id": run_id,
        "root_dir": root_dir,
        "run_dir": run_dir,
        "scene_dir": scene_dir,
        "probability_memmap": scene_dir / f"{scene_id}_probability_water.memmap",
        "probability_geotiff": scene_dir / f"{scene_id}_probability_water.tif",
        "binary_mask_geotiff": scene_dir / f"{scene_id}_mask_water.tif",
        "shoreline_geojson": scene_dir / f"{scene_id}_shoreline.geojson",
        "scene_metadata": scene_dir / f"{scene_id}_metadata.json",
        "run_config": run_dir / "run_config.yaml",
        "run_manifest": run_dir / "run_manifest.csv",
        "run_summary": run_dir / "run_summary.json",
    }

    return paths


def build_inference_transform(cfg):
    if not cfg.inference.data.normalization.enabled:
        return None

    means = np.array(cfg.inference.data.normalization.means, dtype=np.float32)
    stds = np.array(cfg.inference.data.normalization.stds, dtype=np.float32)

    def transform(data: np.ndarray) -> np.ndarray:
        data = data.astype(np.float32, copy=False)

        if data.shape[0] != len(means):
            raise ValueError(
                f"Expected {len(means)} channels for normalization, "
                f"but got {data.shape[0]} channels."
            )

        return (data - means[:, None, None]) / stds[:, None, None]

    return transform


def read_raster_metadata(reference_tif_path: str) -> dict:
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


def memmap_to_geotiff(
    memmap_path: str | Path,
    reference_tif_path: str | Path,
    output_tif_path: str | Path,
    shape: tuple[int, int],
    precision: str,
    band_description: str,
    tags: dict[str, str] | None = None,
) -> Path:
    """Export a probability memmap as a georeferenced single-band GeoTIFF."""
    memmap_path = Path(memmap_path)
    output_tif_path = Path(output_tif_path)

    memmap_dtype = np.float32 if precision == "float32" else np.float16
    output_dtype = "float32"  # Broad GeoTIFF/GIS compatibility, even when the memmap is float16.

    array = np.memmap(memmap_path, dtype=memmap_dtype, mode="r", shape=shape)

    with rasterio.open(reference_tif_path) as src:
        profile = src.profile.copy()

    profile.update(
        driver="GTiff",
        height=shape[0],
        width=shape[1],
        count=1,
        dtype=output_dtype,
        nodata=None,
        compress="deflate",
        predictor=2,
        tiled=True,
        blockxsize=512,
        blockysize=512,
    )

    output_tif_path.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(output_tif_path, "w", **profile) as dst:
        dst.write(np.asarray(array, dtype=np.float32), 1)
        dst.set_band_description(1, band_description)
        if tags:
            dst.update_tags(**tags)

    return output_tif_path


def write_binary_mask_geotiff(
    probability_memmap_path: str | Path,
    reference_tif_path: str | Path,
    output_tif_path: str | Path,
    shape: tuple[int, int],
    precision: str,
    threshold: float,
    tags: dict[str, str] | None = None,
) -> Path:
    """Export a thresholded water mask GeoTIFF from the probability memmap."""
    probability_memmap_path = Path(probability_memmap_path)
    output_tif_path = Path(output_tif_path)

    memmap_dtype = np.float32 if precision == "float32" else np.float16
    prob_map = np.memmap(probability_memmap_path, dtype=memmap_dtype, mode="r", shape=shape)
    mask = (prob_map >= threshold).astype(np.uint8)

    with rasterio.open(reference_tif_path) as src:
        profile = src.profile.copy()

    profile.update(
        driver="GTiff",
        height=shape[0],
        width=shape[1],
        count=1,
        dtype="uint8",
        nodata=255,
        compress="deflate",
        tiled=True,
        blockxsize=512,
        blockysize=512,
    )

    output_tif_path.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(output_tif_path, "w", **profile) as dst:
        dst.write(mask, 1)
        dst.set_band_description(1, "water_mask")
        if tags:
            dst.update_tags(**tags)

    return output_tif_path


def write_run_config_once(cfg: DictConfig, run_config_path: Path):
    """Persist the active Hydra config at run level without rewriting it every scene."""
    if not run_config_path.exists():
        run_config_path.parent.mkdir(parents=True, exist_ok=True)
        run_config_path.write_text(OmegaConf.to_yaml(cfg, resolve=True), encoding="utf-8")


def write_json(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=False), encoding="utf-8")


def append_manifest_row(manifest_path: Path, row: dict):
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

    write_header = not manifest_path.exists()
    with manifest_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in fieldnames})


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


@hydra.main(version_base="1.3", config_path="../configs", config_name="inference")
def main(cfg: DictConfig):
    start_time = time.time()
    created_utc = utc_now_iso()
    git_commit = get_git_commit()

    logger.info("=" * 60)
    logger.info("🌊 INITIALIZING SEGWATER V2 INFERENCE PIPELINE")
    logger.info("=" * 60)
    
    # Dump the config to the log so we know exactly what parameters were injected
    logger.info(f"[CONFIG] Active Runtime Configuration:\n{OmegaConf.to_yaml(cfg)}")

    output_paths = prepare_output_paths(cfg)
    write_run_config_once(cfg, output_paths["run_config"])

    scene_id = output_paths["scene_id"]
    run_id = output_paths["run_id"]
    scene_dir = output_paths["scene_dir"]

    logger.info(f"[OUTPUT] Run ID: {run_id}")
    logger.info(f"[OUTPUT] Scene ID: {scene_id}")
    logger.info(f"[OUTPUT] Scene output directory: {scene_dir}")

    # ---------------------------------------------------------
    # 1. Environment & Hardware Setup
    # ---------------------------------------------------------
    logger.info("--- STAGE 1: HARDWARE INITIALIZATION ---")
    device = torch.device(cfg.inference.device if torch.cuda.is_available() else "cpu")
    logger.info(f"[ENV] Target Device: {device}")
    
    gpu_name = None
    vram_gb = None

    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info(f"[ENV] Detected GPU: {gpu_name} ({vram_gb:.2f} GB VRAM)")
        
        # Performance tuning for RTX 6000 Tensor Cores
        torch.backends.cudnn.benchmark = True
        logger.info("[ENV] cuDNN benchmarking enabled for static graph optimization.")

    # ---------------------------------------------------------
    # 2. Model & Checkpoint Loading
    # ---------------------------------------------------------
    logger.info("--- STAGE 2: MODEL INSTANTIATION ---")
    logger.info(f"[MODEL] Building {cfg.model.arch} with {cfg.model.encoder_name} encoder...")
    
    # We  pull directly from  configs/model/smp.yaml safely
    arch = cfg.model.arch
    encoder = cfg.model.encoder_name
    in_channels = cfg.model.in_channels
    classes = cfg.model.num_classes

    logger.info(f"[MODEL] Building {arch} with {encoder} encoder ({classes} classes)...")
    
    model = SegmentationModelFactory.build(
        arch=arch,
        encoder_name=encoder,
        encoder_weights=None, # Prevent downloading ImageNet weights again
        in_channels=in_channels,
        classes=classes
    )
    
    ckpt_path = cfg.inference.checkpoint_path

    logger.info("[MODEL] Loading state dictionary into VRAM...")
    
    # 1. Load the comprehensive checkpoint object
    checkpoint = torch.load(str(ckpt_path), map_location="cpu")
    
    # 2. Safely unpack the actual weights (Handling our custom MLOps format)
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        saved_step = checkpoint.get('step', 'Unknown')
        saved_miou = checkpoint.get('val_miou', 'Unknown')
        logger.info(f"[MODEL] Successfully unpacked weights from Step: {saved_step} | mIoU: {saved_miou}")
    else:
        # Fallback just in case you ever load a raw PyTorch model
        state_dict = checkpoint
        saved_step = "Unknown"
        saved_miou = "Unknown"
        logger.info("[MODEL] Loaded raw state dictionary directly.")

    model.load_state_dict(state_dict, strict=True)
    
    model.to(device)
    model.eval()  # CRITICAL: Freeze batch norm and dropout
    logger.info("[MODEL] Checkpoint successfully loaded and model set to eval() mode.")

    # ---------------------------------------------------------
    # 3. Data & DataLoader Setup
    # ---------------------------------------------------------
    logger.info("--- STAGE 3: DATA PIPELINE SETUP ---")
    input_image = cfg.inference.data.input_image
    logger.info(f"[DATA] Mounting SAR Swath: {input_image}")

    inference_transform = build_inference_transform(cfg)

    channel_fill_values = None

    if cfg.inference.data.padding.mode == "training_channel_mean":
        channel_fill_values = list(cfg.inference.data.padding.channel_fill_values)
        logger.info(f"[DATA] Using training channel means for padding: {channel_fill_values}")
    
    dataset = InferenceDataset(
        image_path=input_image,
        tile_size=cfg.inference.data.tile_size,
        buffer_size=cfg.inference.data.buffer_size,
        fill_value=cfg.inference.data.fill_value,
        precision=cfg.inference.data.precision,
        transform=inference_transform,
        channel_fill_values=channel_fill_values,
    )
    
    logger.info(f"[DATA] Swath fully gridded. Total tiles to process: {len(dataset)}")
    
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.inference.data.batch_size,
        shuffle=False, # Must be False to preserve spatial order metadata
        num_workers=cfg.inference.data.num_workers,
        pin_memory=True
    )
    logger.info(f"[DATA] DataLoader active: BS={cfg.inference.data.batch_size}, Workers={cfg.inference.data.num_workers}")

    # ---------------------------------------------------------
    # 4. Canvas / Stitcher Setup
    # ---------------------------------------------------------
    logger.info("--- STAGE 4: PROBABILITY STITCHER SETUP ---")
    output_map_path = output_paths["probability_memmap"]
    global_shape = (dataset.height, dataset.width)
    
    stitcher = ProbabilityStitcher(
        output_path=str(output_map_path),
        shape=global_shape,
        precision=cfg.inference.data.precision
    )
    logger.info(f"[STITCHER] Canvas allocated at {output_map_path}")

    # ---------------------------------------------------------
    # 5. The Inference Engine Loop
    # ---------------------------------------------------------
    logger.info("--- STAGE 5: COMMENCING NEURAL INFERENCE ---")
    
    # Dynamic casting based on config
    amp_dtype = torch.float16 if cfg.inference.data.precision == "float16" else torch.float32
    logger.info(f"[INFERENCE] Executing with Automatic Mixed Precision (AMP) dtype: {amp_dtype}")

    total_batches = len(dataloader)
    
    with torch.no_grad():
        for batch_idx, (images, metadata) in enumerate(tqdm(dataloader, desc="Inference Progress")):
            images = images.to(device)

            # Cast inputs to fp16/bf16 for massive speedups on the RTX 6000
            with torch.autocast(device_type=device.type, dtype=amp_dtype):
                logits = model(images)
                
                if cfg.model.num_classes == 1:
                    # Single class logic (Sigmoid)
                    probs = torch.sigmoid(logits).squeeze(1)
                else:
                    # Multi-class logic (Softmax)
                    # Shape goes from [B, 2, H, W] -> Softmax -> slice Channel 1 (Water) -> [B, H, W]
                    probs = torch.softmax(logits, dim=1)[:, 1, :, :]
            
            # Send to disk
            stitcher.add_batch(probs, metadata)

            # Granular logging for development monitoring
            if (batch_idx + 1) % 25 == 0 or (batch_idx + 1) == total_batches:
                logger.info(f"[INFERENCE] Successfully processed {batch_idx + 1}/{total_batches} batches.")

    logger.info("[INFERENCE] GPU computation complete.")

    # ---------------------------------------------------------
    # 6. Finalization, Product Export & Vectorization
    # ---------------------------------------------------------
    logger.info("--- STAGE 6: DISK FLUSH & PRODUCT EXPORT ---")
    logger.info("[STITCHER] Initiating forced flush to write remaining memory buffer to disk...")
    stitcher.close()

    geotiff_tags = build_geotiff_tags(
        cfg=cfg,
        scene_id=scene_id,
        run_id=run_id,
        product="water_probability",
        created_utc=created_utc,
        git_commit=git_commit,
    )

    probability_geotiff_path = None
    if cfg.inference.output.save_probability_geotiff:
        probability_geotiff_path = memmap_to_geotiff(
            memmap_path=output_map_path,
            reference_tif_path=input_image,
            output_tif_path=output_paths["probability_geotiff"],
            shape=global_shape,
            precision=cfg.inference.data.precision,
            band_description="water_probability",
            tags=geotiff_tags,
        )
        logger.info(f"[OUTPUT] Probability GeoTIFF saved to: {probability_geotiff_path}")

    binary_mask_path = None
    if cfg.inference.output.save_binary_mask_geotiff:
        mask_tags = build_geotiff_tags(
            cfg=cfg,
            scene_id=scene_id,
            run_id=run_id,
            product="water_mask",
            created_utc=created_utc,
            git_commit=git_commit,
        )
        binary_mask_path = write_binary_mask_geotiff(
            probability_memmap_path=output_map_path,
            reference_tif_path=input_image,
            output_tif_path=output_paths["binary_mask_geotiff"],
            shape=global_shape,
            precision=cfg.inference.data.precision,
            threshold=cfg.inference.post_processing.threshold,
            tags=mask_tags,
        )
        logger.info(f"[OUTPUT] Binary mask GeoTIFF saved to: {binary_mask_path}")
    
    shoreline_path = None
    if cfg.inference.output.extract_shoreline:
        logger.info("[POST-PROC] Shoreline extraction requested! Handing off to Vectorizer...")

        min_length_meters = (
            cfg.inference.post_processing.filtering.min_length_meters
            if cfg.inference.post_processing.filtering.apply_length_filter
            else 0.0
        )

        simplify_tolerance_meters = (
            cfg.inference.post_processing.smoothing.simplify_tolerance_meters
            if cfg.inference.post_processing.smoothing.apply_simplification
            else 0.0
        )

        vectorizer = ShorelineVectorizer(
            prob_map_path=str(output_map_path),
            reference_tif_path=input_image,
            shape=global_shape,
            precision=cfg.inference.data.precision,
            threshold=cfg.inference.post_processing.threshold,
            min_length_meters=min_length_meters,
            simplify_tolerance_meters=simplify_tolerance_meters,
            keep_top_k=cfg.inference.post_processing.filtering.keep_top_k
        )

        shoreline_path = vectorizer.extract_and_save(
            output_geojson_path=str(output_paths["shoreline_geojson"])
        )

        logger.info(f"[POST-PROC] Shoreline vector output available at: {shoreline_path}")

    else:
        logger.info("[POST-PROC] Shoreline extraction bypassed via configuration.")

    elapsed = (time.time() - start_time) / 60

    raster_metadata = read_raster_metadata(input_image)
    scene_metadata = {
        "scene_id": scene_id,
        "input": {
            "path": str(input_image),
            "filename": Path(str(input_image)).name,
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
            "scene_dir": str(scene_dir),
            "probability_memmap": str(output_map_path),
            "probability_geotiff": str(probability_geotiff_path) if probability_geotiff_path else None,
            "binary_mask_geotiff": str(binary_mask_path) if binary_mask_path else None,
            "shoreline_geojson": str(shoreline_path) if shoreline_path else None,
            "metadata_json": str(output_paths["scene_metadata"]),
        },
        "run": {
            "run_id": run_id,
            "created_utc": created_utc,
            "elapsed_minutes": elapsed,
            "segwater_git_commit": git_commit,
            "run_config": str(output_paths["run_config"]),
            "run_manifest": str(output_paths["run_manifest"]),
        },
    }

    write_json(output_paths["scene_metadata"], scene_metadata)
    logger.info(f"[OUTPUT] Scene metadata saved to: {output_paths['scene_metadata']}")

    manifest_row = {
        "run_id": run_id,
        "scene_id": scene_id,
        "status": "success",
        "input_path": str(input_image),
        "probability_memmap": str(output_map_path),
        "probability_geotiff": str(probability_geotiff_path) if probability_geotiff_path else "",
        "binary_mask_geotiff": str(binary_mask_path) if binary_mask_path else "",
        "shoreline_geojson": str(shoreline_path) if shoreline_path else "",
        "metadata_json": str(output_paths["scene_metadata"]),
        "width": dataset.width,
        "height": dataset.height,
        "crs": raster_metadata["crs"],
        "threshold": cfg.inference.post_processing.threshold,
        "checkpoint_path": str(cfg.inference.checkpoint_path),
        "created_utc": created_utc,
        "elapsed_minutes": f"{elapsed:.4f}",
    }
    append_manifest_row(output_paths["run_manifest"], manifest_row)
    logger.info(f"[OUTPUT] Run manifest updated at: {output_paths['run_manifest']}")

    run_summary = {
        "run_id": run_id,
        "checkpoint_path": str(cfg.inference.checkpoint_path),
        "model_architecture": str(cfg.model.arch),
        "model_encoder": str(cfg.model.encoder_name),
        "last_updated_utc": utc_now_iso(),
        "segwater_git_commit": git_commit,
        "run_config": str(output_paths["run_config"]),
        "run_manifest": str(output_paths["run_manifest"]),
        "root_dir": str(output_paths["root_dir"]),
    }
    write_json(output_paths["run_summary"], run_summary)

    logger.info("=" * 60)
    logger.info(f"✅ SEGWATER V2 INFERENCE SUCCESSFUL in {elapsed:.2f} minutes.")
    logger.info(f"📁 Scene output directory: {scene_dir}")
    logger.info(f"🧠 Probability memmap available at: {output_map_path}")
    if probability_geotiff_path:
        logger.info(f"🗺️ Probability GeoTIFF available at: {probability_geotiff_path}")
    if binary_mask_path:
        logger.info(f"🎭 Binary mask GeoTIFF available at: {binary_mask_path}")
    if shoreline_path:
        logger.info(f"🌊 Shoreline vector available at: {shoreline_path}")
    logger.info(f"🧾 Metadata available at: {output_paths['scene_metadata']}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
