import os
import numpy as np
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_random_memmap_subset(
    src_path: str, 
    dst_path: str, 
    original_shape: tuple, 
    subset_ratio: float = 0.10, 
    chunk_size: int = 500,
    dtype: str = "float32", # Derived from the S1 VV/VH arrays
    seed: int = 42
):
    """
    Safely copies a random subset of a large memmap file to a new memmap file.
    Sorts indices to ensure sequential disk reads, preventing I/O bottlenecks.
    """
    if not os.path.exists(src_path):
        raise FileNotFoundError(f"Source file not found: {src_path}")

    total_samples = original_shape[0]
    subset_samples = int(total_samples * subset_ratio)
    
    logger.info(f"Targeting {subset_samples} samples ({subset_ratio*100}%) from {total_samples} total.")

    # 1. Generate and SORT random indices
    # Sorting is critical: it prevents random disk thrashing during extraction
    np.random.seed(seed)
    logger.info("Generating and sorting random indices for sequential reading...")
    indices = np.random.choice(total_samples, subset_samples, replace=False)
    indices.sort()

    # 2. Open the original memmap in read-only mode
    logger.info(f"Opening source memmap: {src_path}")
    src_mm = np.memmap(src_path, dtype=dtype, mode="r", shape=original_shape)
    
    # 3. Define the new shape and create the destination memmap
    subset_shape = (subset_samples, *original_shape[1:])
    logger.info(f"Creating destination memmap: {dst_path} with shape {subset_shape}")
    dst_mm = np.memmap(dst_path, dtype=dtype, mode="w+", shape=subset_shape)
    
    # 4. Copy data in chunks to prevent RAM overflow
    logger.info(f"Copying data in chunks of {chunk_size}...")
    for start_idx in tqdm(range(0, subset_samples, chunk_size), desc="Extracting Subset"):
        end_idx = min(start_idx + chunk_size, subset_samples)
        
        # Get the specific sorted indices for this chunk
        chunk_indices = indices[start_idx:end_idx]
        
        # Extract from source to RAM, then write from RAM to destination disk
        dst_mm[start_idx:end_idx] = src_mm[chunk_indices]
        
    # 5. Flush changes to disk
    dst_mm.flush()
    logger.info(f"Successfully saved random subset to {dst_path}\n")

if __name__ == "__main__":
    # Path provided
    #SRC_PATH = "/Volumes/NTU-Backup/aria/noelivan.ulloa/projects/Global_Sen12_Coast/data/memmap_dataset_v2/memmap_datasets_v2/train.memmap"
    SRC_PATH = "/Volumes/NTU_5TB/aria/noelivan.ulloa/projects/Global_Sen12_Coast/data/memmap_datasets_v2/val.memmap"

    # Automatically append _10pct to the new filename
    #DST_PATH = SRC_PATH.replace("train.memmap", "train_10pct.memmap")
    #DST_PATH = "/Users/noel/Documents/Research_Projects/segwater_v2/data/train_10pct.memmap"
    DST_PATH = "/Users/noel/Documents/Research_Projects/segwater_v2/data/val_10pct.memmap"

    # Current dataset shape (1,032,526 samples, 3 channels, 224x224)
    #ORIGINAL_SHAPE = (1032526, 3, 224, 224)
    ORIGINAL_SHAPE = (294643,3,224,224)
    
    # Execute extraction
    create_random_memmap_subset(
        src_path=SRC_PATH, 
        dst_path=DST_PATH, 
        original_shape=ORIGINAL_SHAPE,
        subset_ratio=0.10,
        chunk_size=500,    # Pull 500 samples into RAM at a time
        dtype="float32"    # Matches the original VV/VH/Mask tensor layout
    )