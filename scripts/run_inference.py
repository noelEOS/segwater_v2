import logging
import time
from pathlib import Path

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.inference_dataset import InferenceDataset
from src.models.factory import SegmentationModelFactory
from src.utils.inference_outputs import (
    append_manifest_row,
    build_geotiff_tags,
    build_manifest_row,
    build_run_summary,
    build_scene_metadata,
    get_git_commit,
    prepare_output_paths,
    read_raster_metadata,
    sanitize_for_path,
    utc_now_iso,
    write_json,
    write_run_config_once,
)
from src.utils.raster_export import memmap_to_geotiff, write_binary_mask_geotiff
from src.utils.stitcher import ProbabilityStitcher
from src.utils.tta import predict_once, predict_with_tta, validate_tta_transforms
from src.utils.vectorizer import ShorelineVectorizer

logger = logging.getLogger(__name__)


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


def build_channel_fill_values(cfg):
    """Return per-channel dB padding values when configured, otherwise scalar fallback is used."""
    if cfg.inference.data.padding.mode == "training_channel_mean":
        return list(cfg.inference.data.padding.channel_fill_values)
    return None


def _get_optional_cfg(cfg_node, key, default):
    """Safely read optional Hydra/OmegaConf keys while preserving old configs."""
    return cfg_node[key] if key in cfg_node else default


def resolve_input_images(cfg) -> list[str]:
    """Resolve one or more input images for a single checkpoint-loaded process."""
    data_cfg = cfg.inference.data

    input_list_file = _get_optional_cfg(data_cfg, "input_list_file", None)
    if input_list_file:
        path = Path(str(input_list_file))
        images = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
        return [image for image in images if image and not image.startswith("#")]

    input_images = _get_optional_cfg(data_cfg, "input_images", None)
    if input_images:
        return [str(image) for image in input_images]

    input_dir = _get_optional_cfg(data_cfg, "input_dir", None)
    if input_dir:
        input_glob = _get_optional_cfg(data_cfg, "input_glob", "*.tif")
        return sorted(str(path) for path in Path(str(input_dir)).glob(str(input_glob)))

    return [str(data_cfg.input_image)]


def build_model_and_load_checkpoint(cfg, device):
    arch = cfg.model.arch
    encoder = cfg.model.encoder_name
    in_channels = cfg.model.in_channels
    classes = cfg.model.num_classes

    logger.info(f"[MODEL] Building {arch} with {encoder} encoder ({classes} classes)...")

    model = SegmentationModelFactory.build(
        arch=arch,
        encoder_name=encoder,
        encoder_weights=None,
        in_channels=in_channels,
        classes=classes,
    )

    ckpt_path = cfg.inference.checkpoint_path
    logger.info(f"[MODEL] Loading state dictionary into VRAM: {ckpt_path}")
    checkpoint = torch.load(str(ckpt_path), map_location="cpu")

    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        saved_step = checkpoint.get("step", "Unknown")
        saved_miou = checkpoint.get("val_miou", "Unknown")
        logger.info(f"[MODEL] Successfully unpacked weights from Step: {saved_step} | mIoU: {saved_miou}")
    else:
        state_dict = checkpoint
        saved_step = "Unknown"
        saved_miou = "Unknown"
        logger.info("[MODEL] Loaded raw state dictionary directly.")

    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    logger.info("[MODEL] Checkpoint successfully loaded and model set to eval() mode.")

    return model, saved_step, saved_miou


def _build_stitcher(
    output_path,
    global_shape,
    cfg,
    stitching_mode,
    blend_window,
    min_weight,
    keep_accumulator_memmaps,
):
    return ProbabilityStitcher(
        output_path=str(output_path),
        shape=global_shape,
        precision=cfg.inference.data.precision,
        mode=stitching_mode,
        blend_window=blend_window,
        min_weight=min_weight,
        keep_accumulator_memmaps=keep_accumulator_memmaps,
    )


def _build_tta_debug_stitchers(
    paths,
    global_shape,
    cfg,
    transforms,
    stitching_mode,
    blend_window,
    min_weight,
    keep_accumulator_memmaps,
):
    """Create one stitcher per inverse-transformed TTA view for QGIS inspection."""
    stitchers = {}
    for transform in transforms:
        safe_transform = sanitize_for_path(transform)
        output_path = paths.scene_dir / f"{paths.scene_id}_probability_water_tta_{safe_transform}.memmap"
        stitchers[transform] = _build_stitcher(
            output_path=output_path,
            global_shape=global_shape,
            cfg=cfg,
            stitching_mode=stitching_mode,
            blend_window=blend_window,
            min_weight=min_weight,
            keep_accumulator_memmaps=keep_accumulator_memmaps,
        )
    return stitchers


def process_scene(
    cfg: DictConfig,
    input_image: str,
    model: torch.nn.Module,
    device: torch.device,
    gpu_name: str | None,
    vram_gb: float | None,
    saved_step,
    saved_miou,
    git_commit: str | None,
):
    scene_start_time = time.time()
    created_utc = utc_now_iso()

    cfg.inference.data.input_image = str(input_image)
    paths = prepare_output_paths(cfg)
    write_run_config_once(cfg, paths.run_config)

    logger.info("=" * 60)
    logger.info(f"[SCENE] Processing: {input_image}")
    logger.info(f"[OUTPUT] Run ID: {paths.run_id}")
    logger.info(f"[OUTPUT] Scene ID: {paths.scene_id}")
    logger.info(f"[OUTPUT] Scene output directory: {paths.scene_dir}")

    # ---------------------------------------------------------
    # 3. Data & DataLoader Setup
    # ---------------------------------------------------------
    logger.info("--- STAGE 3: DATA PIPELINE SETUP ---")
    inference_transform = build_inference_transform(cfg)
    channel_fill_values = build_channel_fill_values(cfg)

    if channel_fill_values is not None:
        logger.info(f"[DATA] Using training channel means for padding: {channel_fill_values}")

    stride = _get_optional_cfg(cfg.inference.data, "stride", cfg.inference.data.tile_size)
    edge_policy = _get_optional_cfg(cfg.inference.data, "edge_policy", "shift_inward")

    dataset = InferenceDataset(
        image_path=input_image,
        tile_size=cfg.inference.data.tile_size,
        buffer_size=cfg.inference.data.buffer_size,
        fill_value=cfg.inference.data.fill_value,
        precision=cfg.inference.data.precision,
        transform=inference_transform,
        channel_fill_values=channel_fill_values,
        stride=stride,
        edge_policy=edge_policy,
    )

    logger.info(f"[DATA] Swath fully gridded. Total tiles to process: {len(dataset)}")

    dataloader = DataLoader(
        dataset,
        batch_size=cfg.inference.data.batch_size,
        shuffle=False,
        num_workers=cfg.inference.data.num_workers,
        pin_memory=True,
    )
    logger.info(f"[DATA] DataLoader active: BS={cfg.inference.data.batch_size}, Workers={cfg.inference.data.num_workers}")

    # ---------------------------------------------------------
    # 4. Canvas / Stitcher Setup
    # ---------------------------------------------------------
    logger.info("--- STAGE 4: PROBABILITY STITCHER SETUP ---")
    global_shape = (dataset.height, dataset.width)

    stitching_cfg = _get_optional_cfg(cfg.inference, "stitching", {})
    stitching_mode = _get_optional_cfg(stitching_cfg, "mode", "crop_only")
    blend_window = _get_optional_cfg(stitching_cfg, "blend_window", "hann")
    min_weight = _get_optional_cfg(stitching_cfg, "min_weight", 1e-3)
    keep_accumulator_memmaps = bool(_get_optional_cfg(stitching_cfg, "keep_accumulator_memmaps", False))

    stitcher = _build_stitcher(
        output_path=paths.probability_memmap,
        global_shape=global_shape,
        cfg=cfg,
        stitching_mode=stitching_mode,
        blend_window=blend_window,
        min_weight=min_weight,
        keep_accumulator_memmaps=keep_accumulator_memmaps,
    )
    logger.info(f"[STITCHER] Canvas allocated at {paths.probability_memmap}")

    tta_cfg = _get_optional_cfg(cfg.inference, "tta", {})
    tta_enabled = bool(_get_optional_cfg(tta_cfg, "enabled", False))
    tta_transforms = list(_get_optional_cfg(tta_cfg, "transforms", ["identity"]))
    tta_save_individual = bool(_get_optional_cfg(tta_cfg, "save_individual_probability_geotiffs", False))
    tta_debug_stitchers = {}

    if tta_enabled:
        tta_transforms = validate_tta_transforms(tta_transforms)
        logger.info(f"[TTA] Enabled with transforms: {tta_transforms}")
        if tta_save_individual:
            tta_debug_stitchers = _build_tta_debug_stitchers(
                paths,
                global_shape,
                cfg,
                tta_transforms,
                stitching_mode,
                blend_window,
                min_weight,
                keep_accumulator_memmaps,
            )
            logger.info("[TTA] Individual inverse-transformed probability GeoTIFF export enabled.")
    else:
        logger.info("[TTA] Disabled.")

    # ---------------------------------------------------------
    # 5. Inference Engine Loop
    # ---------------------------------------------------------
    logger.info("--- STAGE 5: COMMENCING NEURAL INFERENCE ---")
    amp_dtype = torch.float16 if cfg.inference.data.precision == "float16" else torch.float32
    logger.info(f"[INFERENCE] Executing with Automatic Mixed Precision (AMP) dtype: {amp_dtype}")

    total_batches = len(dataloader)

    with torch.no_grad():
        for batch_idx, (images, metadata) in enumerate(tqdm(dataloader, desc=f"Inference {paths.scene_id}")):
            images = images.to(device)

            with torch.autocast(device_type=device.type, dtype=amp_dtype):
                if tta_enabled:
                    probs, individual_tta_probs = predict_with_tta(
                        model=model,
                        images=images,
                        num_classes=cfg.model.num_classes,
                        transforms=tta_transforms,
                        return_individual=tta_save_individual,
                    )
                else:
                    probs = predict_once(model, images, num_classes=cfg.model.num_classes)
                    individual_tta_probs = {}

            stitcher.add_batch(probs, metadata)

            for transform, transform_probs in individual_tta_probs.items():
                tta_debug_stitchers[transform].add_batch(transform_probs, metadata)

            if (batch_idx + 1) % 25 == 0 or (batch_idx + 1) == total_batches:
                logger.info(f"[INFERENCE] Successfully processed {batch_idx + 1}/{total_batches} batches.")

    logger.info("[INFERENCE] GPU computation complete.")

    # ---------------------------------------------------------
    # 6. Finalization, Product Export & Vectorization
    # ---------------------------------------------------------
    logger.info("--- STAGE 6: DISK FLUSH & PRODUCT EXPORT ---")
    stitcher.close()

    tta_probability_geotiff_paths = {}
    for transform, transform_stitcher in tta_debug_stitchers.items():
        transform_stitcher.close()

    geotiff_tags = build_geotiff_tags(
        cfg=cfg,
        scene_id=paths.scene_id,
        run_id=paths.run_id,
        product="water_probability",
        created_utc=created_utc,
        git_commit=git_commit,
    )

    probability_geotiff_path = None
    if cfg.inference.output.save_probability_geotiff:
        probability_geotiff_path = memmap_to_geotiff(
            memmap_path=paths.probability_memmap,
            reference_tif_path=input_image,
            output_tif_path=paths.probability_geotiff,
            shape=global_shape,
            precision=cfg.inference.data.precision,
            band_description="water_probability",
            tags=geotiff_tags,
        )
        logger.info(f"[OUTPUT] Probability GeoTIFF saved to: {probability_geotiff_path}")

    if tta_save_individual:
        for transform in tta_debug_stitchers:
            safe_transform = sanitize_for_path(transform)
            memmap_path = paths.scene_dir / f"{paths.scene_id}_probability_water_tta_{safe_transform}.memmap"
            geotiff_path = paths.scene_dir / f"{paths.scene_id}_probability_water_tta_{safe_transform}.tif"
            tta_tags = build_geotiff_tags(
                cfg=cfg,
                scene_id=paths.scene_id,
                run_id=paths.run_id,
                product=f"water_probability_tta_{safe_transform}",
                created_utc=created_utc,
                git_commit=git_commit,
            )
            tta_probability_geotiff_paths[transform] = memmap_to_geotiff(
                memmap_path=memmap_path,
                reference_tif_path=input_image,
                output_tif_path=geotiff_path,
                shape=global_shape,
                precision=cfg.inference.data.precision,
                band_description=f"water_probability_tta_{safe_transform}",
                tags=tta_tags,
            )
            logger.info(
                f"[TTA] Individual probability GeoTIFF for {transform} saved to: "
                f"{tta_probability_geotiff_paths[transform]}"
            )

    binary_mask_path = None
    if cfg.inference.output.save_binary_mask_geotiff:
        mask_tags = build_geotiff_tags(
            cfg=cfg,
            scene_id=paths.scene_id,
            run_id=paths.run_id,
            product="water_mask",
            created_utc=created_utc,
            git_commit=git_commit,
        )
        binary_mask_path = write_binary_mask_geotiff(
            probability_memmap_path=paths.probability_memmap,
            reference_tif_path=input_image,
            output_tif_path=paths.binary_mask_geotiff,
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
            prob_map_path=str(paths.probability_memmap),
            reference_tif_path=input_image,
            shape=global_shape,
            precision=cfg.inference.data.precision,
            threshold=cfg.inference.post_processing.threshold,
            min_length_meters=min_length_meters,
            simplify_tolerance_meters=simplify_tolerance_meters,
            keep_top_k=cfg.inference.post_processing.filtering.keep_top_k,
        )

        shoreline_path = vectorizer.extract_and_save(output_geojson_path=str(paths.shoreline_geojson))
        logger.info(f"[POST-PROC] Shoreline vector output available at: {shoreline_path}")
    else:
        logger.info("[POST-PROC] Shoreline extraction bypassed via configuration.")

    # ---------------------------------------------------------
    # 7. Provenance Records
    # ---------------------------------------------------------
    elapsed = (time.time() - scene_start_time) / 60
    raster_metadata = read_raster_metadata(input_image)

    scene_metadata = build_scene_metadata(
        cfg=cfg,
        paths=paths,
        input_image=input_image,
        raster_metadata=raster_metadata,
        saved_step=saved_step,
        saved_miou=saved_miou,
        device=device,
        gpu_name=gpu_name,
        vram_gb=vram_gb,
        probability_geotiff_path=probability_geotiff_path,
        binary_mask_path=binary_mask_path,
        shoreline_path=shoreline_path,
        created_utc=created_utc,
        elapsed_minutes=elapsed,
        git_commit=git_commit,
    )
    scene_metadata["tta_probability_geotiffs"] = tta_probability_geotiff_paths
    write_json(paths.scene_metadata, scene_metadata)
    logger.info(f"[OUTPUT] Scene metadata saved to: {paths.scene_metadata}")

    manifest_row = build_manifest_row(
        cfg=cfg,
        paths=paths,
        input_image=input_image,
        raster_metadata=raster_metadata,
        probability_geotiff_path=probability_geotiff_path,
        binary_mask_path=binary_mask_path,
        shoreline_path=shoreline_path,
        created_utc=created_utc,
        elapsed_minutes=elapsed,
    )
    append_manifest_row(paths.run_manifest, manifest_row)
    logger.info(f"[OUTPUT] Run manifest updated at: {paths.run_manifest}")

    run_summary = build_run_summary(cfg=cfg, paths=paths, git_commit=git_commit)
    write_json(paths.run_summary, run_summary)

    logger.info("=" * 60)
    logger.info(f"✅ SCENE INFERENCE SUCCESSFUL in {elapsed:.2f} minutes.")
    logger.info(f"📁 Scene output directory: {paths.scene_dir}")
    logger.info(f"🧠 Probability memmap available at: {paths.probability_memmap}")
    if probability_geotiff_path:
        logger.info(f"🗺️ Probability GeoTIFF available at: {probability_geotiff_path}")
    for transform, path in tta_probability_geotiff_paths.items():
        logger.info(f"🧪 TTA {transform} probability GeoTIFF available at: {path}")
    if binary_mask_path:
        logger.info(f"🎭 Binary mask GeoTIFF available at: {binary_mask_path}")
    if shoreline_path:
        logger.info(f"🌊 Shoreline vector available at: {shoreline_path}")
    logger.info(f"🧾 Metadata available at: {paths.scene_metadata}")
    logger.info("=" * 60)


@hydra.main(version_base="1.3", config_path="../configs", config_name="inference")
def main(cfg: DictConfig):
    pipeline_start_time = time.time()
    git_commit = get_git_commit()

    logger.info("=" * 60)
    logger.info("🌊 INITIALIZING SEGWATER V2 INFERENCE PIPELINE")
    logger.info("=" * 60)
    logger.info(f"[CONFIG] Active Runtime Configuration:\n{OmegaConf.to_yaml(cfg)}")

    input_images = resolve_input_images(cfg)
    if not input_images:
        raise ValueError("No input images were resolved for inference.")

    logger.info(f"[DATA] Resolved {len(input_images)} input scene(s).")

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
        torch.backends.cudnn.benchmark = True
        logger.info("[ENV] cuDNN benchmarking enabled for static graph optimization.")

    # ---------------------------------------------------------
    # 2. Model & Checkpoint Loading -- done once per process.
    # ---------------------------------------------------------
    logger.info("--- STAGE 2: MODEL INSTANTIATION ---")
    model, saved_step, saved_miou = build_model_and_load_checkpoint(cfg, device)

    successful = 0
    failed = 0

    for scene_index, input_image in enumerate(input_images, start=1):
        logger.info(f"[BATCH] Starting scene {scene_index}/{len(input_images)}")
        try:
            process_scene(
                cfg=cfg,
                input_image=input_image,
                model=model,
                device=device,
                gpu_name=gpu_name,
                vram_gb=vram_gb,
                saved_step=saved_step,
                saved_miou=saved_miou,
                git_commit=git_commit,
            )
            successful += 1
        except Exception as exc:
            failed += 1
            logger.exception(f"[BATCH] Scene failed: {input_image} | Error: {exc}")
            if not bool(_get_optional_cfg(cfg.inference, "continue_on_error", False)):
                raise

    elapsed = (time.time() - pipeline_start_time) / 60
    logger.info("=" * 60)
    logger.info(
        f"✅ SEGWATER V2 INFERENCE PROCESS COMPLETE in {elapsed:.2f} minutes. "
        f"Successful scenes: {successful} | Failed scenes: {failed}"
    )
    logger.info("=" * 60)

    if failed > 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
