import os
import time
import logging
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# Import our custom pipeline modules
from src.data.inference_dataset import InferenceDataset
from src.utils.stitcher import ProbabilityStitcher
from src.models.factory import SegmentationModelFactory

# Configure basic logging level (Hydra handles the file routing automatically)
logger = logging.getLogger(__name__)

@hydra.main(version_base="1.3", config_path="../configs", config_name="inference")
def main(cfg: DictConfig):
    start_time = time.time()
    logger.info("=" * 60)
    logger.info("🌊 INITIALIZING SEGWATER V2 INFERENCE PIPELINE")
    logger.info("=" * 60)
    
    # Dump the config to the log so we know exactly what parameters were injected
    logger.info(f"[CONFIG] Active Runtime Configuration:\n{OmegaConf.to_yaml(cfg)}")

    # ---------------------------------------------------------
    # 1. Environment & Hardware Setup
    # ---------------------------------------------------------
    logger.info("--- STAGE 1: HARDWARE INITIALIZATION ---")
    device = torch.device(cfg.inference.device if torch.cuda.is_available() else "cpu")
    logger.info(f"[ENV] Target Device: {device}")
    
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
    
    dataset = InferenceDataset(
        image_path=input_image,
        tile_size=cfg.inference.data.tile_size,
        buffer_size=cfg.inference.data.buffer_size,
        fill_value=cfg.inference.data.fill_value,
        precision=cfg.inference.data.precision,
        transform=None # Add torchvision/albumentations here if required
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
    output_map_path = cfg.inference.output.probability_map
    global_shape = (dataset.height, dataset.width)
    
    stitcher = ProbabilityStitcher(
        output_path=output_map_path,
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
    # 6. Finalization & Teardown
    # ---------------------------------------------------------
    logger.info("--- STAGE 6: DISK FLUSH & VECTORIZATION ---")
    logger.info("[STITCHER] Initiating forced flush to write remaining memory buffer to disk...")
    stitcher.close()
    
    if cfg.inference.output.extract_shoreline:
        logger.info("[POST-PROC] Shoreline extraction requested! Handing off to Vectorizer...")
        # NOTE: Marching Squares logic will go here
    else:
        logger.info("[POST-PROC] Shoreline extraction bypassed via configuration.")

    elapsed = (time.time() - start_time) / 60
    logger.info("=" * 60)
    logger.info(f"✅ SEGWATER V2 INFERENCE SUCCESSFUL in {elapsed:.2f} minutes.")
    logger.info(f"🗺️  Output available at: {output_map_path}")
    logger.info("=" * 60)

if __name__ == "__main__":
    main()