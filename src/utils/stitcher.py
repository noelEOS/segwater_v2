import os
import logging
from typing import Dict
import numpy as np
import torch

logger = logging.getLogger(__name__)

class ProbabilityStitcher:
    """
    Constructs a global probability map out-of-core using np.memmap.
    Receives buffered model outputs, crops the edges, and stitches the high-confidence
    centers back into their exact geographical positions.
    """
    def __init__(
        self,
        output_path: str,
        shape: tuple[int, int],
        precision: str
    ):
        """
        Args:
            output_path: Destination path for the .memmap file.
            shape: Tuple of (height, width) matching the original SAR Swath.
            precision: Data type for the probability map (e.g., 'float32', 'float16').
        """
        self.output_path = output_path
        self.shape = shape

        # 1. Strict Type Enforcement
        if precision == "float32":
            self.dtype = np.float32
        elif precision == "float16":
            self.dtype = np.float16
        else:
            raise ValueError(f"Unsupported precision: {precision}. Must be 'float32' or 'float16'.")

        # 2. File Initialization
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        
        # 'w+' creates a new file or overwrites an existing one, ensuring a clean canvas.
        self.memmap = np.memmap(
            self.output_path,
            dtype=self.dtype,
            mode='w+',
            shape=self.shape
        )
        logger.info(f"Initialized Global Probability Canvas: {self.shape} | Dtype: {self.dtype}")

    def add_batch(self, batch_probs: torch.Tensor, metadata: Dict[str, torch.Tensor]):
        """
        Slices the edge-buffers from the model predictions and writes the 
        valid centers into the global memory-mapped file.
        
        Args:
            batch_probs: Tensor of shape (B, H_padded, W_padded) containing sigmoid probabilities.
            metadata: Dictionary of batch spatial coordinates from InferenceDataset.
        """
        # Move probabilities to CPU and convert to numpy for disk writing
        batch_probs_np = batch_probs.detach().cpu().numpy()
        batch_size = batch_probs_np.shape[0]

        for i in range(batch_size):
            # PyTorch's default_collate converts our dataset integers into 1D tensors.
            # We use .item() to pull them back into native Python ints.
            y0 = metadata["valid_y0"][i].item()
            x0 = metadata["valid_x0"][i].item()
            h = metadata["valid_h"][i].item()
            w = metadata["valid_w"][i].item()
            buffer = metadata["buffer_size"][i].item()

            # 3. The "Overlap and Crop" Realization
            # Slice out the buffer zone to extract only the high-confidence center
            crop_prob = batch_probs_np[i, buffer:buffer + h, buffer:buffer + w]

            # 4. Global Canvas Stitching
            # Write the cropped tile into its exact spatial location
            self.memmap[y0:y0 + h, x0:x0 + w] = crop_prob

    def close(self):
        """Flushes final data to disk and closes the file pointer."""
        self.memmap.flush()
        del self.memmap
        logger.info(f"Probability map successfully flushed and closed at {self.output_path}")