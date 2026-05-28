import math
import logging
from typing import Dict, Tuple, Optional, Callable

import numpy as np
import rasterio
from rasterio.windows import Window
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

class InferenceDataset(Dataset):
    """
    Dataset for sliding-window inference on large SAR swaths.
    Implements the 'Overlap and Crop' strategy to mitigate edge artifacts.
    """
    def __init__(
        self, 
        image_path: str, 
        tile_size: int,         
        buffer_size: int,       
        fill_value: float,      
        precision: str,         
        transform: Optional[Callable] = None,
        channel_fill_values: Optional[list[float]] = None,
        ):
        """
        Args:
            image_path: Path to the input full-swath GeoTIFF.
            transform: Transformations (e.g., standard normalization) applied to the tensor.
            tile_size: The dimension of the 'valid' center crop (e.g., 1024).
            buffer_size: The contextual padding added to all sides (e.g., 128).
                         Total model input size will be (tile_size + 2 * buffer_size).
            channel_fill_values: A list of fill values for each channel (one for each polarization).
                                 This is used when the buffer extends beyond the edge of the GeoTIFF.
        """
        self.image_path = image_path
        self.transform = transform
        self.tile_size = tile_size
        self.buffer_size = buffer_size
        self.fill_value = fill_value
        self.channel_fill_values = channel_fill_values
        self.dtype = np.float32 if precision == "float32" else np.float16
        
        # The full tensor size fed to the RTX 6000 (e.g., 1280x1280)
        self.window_size = self.tile_size + (2 * self.buffer_size)
        
        # Extract swath dimensions
        with rasterio.open(self.image_path) as src:
            self.height = src.height
            self.width = src.width
            self.num_channels = src.count
            
        # Calculate the grid of valid tiles needed to cover the swath
        self.tiles_y = math.ceil(self.height / self.tile_size)
        self.tiles_x = math.ceil(self.width / self.tile_size)
        self.total_tiles = self.tiles_y * self.tiles_x

        logger.info(f"Initialized InferenceDataset for {image_path}")
        logger.info(f"Swath dimensions: {self.width}x{self.height}")
        logger.info(f"Grid: {self.tiles_x}x{self.tiles_y} tiles (Total: {self.total_tiles})")
        logger.info(f"Input tensor shape: {self.num_channels}x{self.window_size}x{self.window_size}")

    def __len__(self) -> int:
        return self.total_tiles

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, int]]:
        # 1. Map 1D index back to 2D grid coordinates
        tile_y_idx = idx // self.tiles_x
        tile_x_idx = idx % self.tiles_x

        # 2. Calculate coordinates for the 'Valid' center crop
        valid_y0 = tile_y_idx * self.tile_size
        valid_x0 = tile_x_idx * self.tile_size
        
        # Calculate valid boundary sizes (handles the edges of the swath where a full 1024 tile doesn't fit)
        valid_h = min(self.tile_size, self.height - valid_y0)
        valid_w = min(self.tile_size, self.width - valid_x0)

        # 3. Calculate coordinates for the 'Padded' input window (1280x1280)
        # Note: These can be negative if we are at the top/left edge of the swath
        window_y0 = valid_y0 - self.buffer_size
        window_x0 = valid_x0 - self.buffer_size

        # 4. Read the buffered window from the GeoTIFF
        # We re-open the file per worker to ensure thread safety in PyTorch DataLoaders
        with rasterio.open(self.image_path) as src:
            if self.channel_fill_values is None:
                window = Window(
                    col_off=window_x0,
                    row_off=window_y0,
                    width=self.window_size,
                    height=self.window_size,
                )

                data = src.read(
                    window=window,
                    boundless=True,
                    fill_value=self.fill_value,
                )

            else:
                fill_values = np.asarray(self.channel_fill_values, dtype=self.dtype)

                if len(fill_values) != self.num_channels:
                    raise ValueError(
                        f"Expected {self.num_channels} channel fill values, "
                        f"but got {len(fill_values)}."
                    )

                data = np.empty(
                    (self.num_channels, self.window_size, self.window_size),
                    dtype=self.dtype,
                )

                data[:] = fill_values[:, None, None]

                src_row_start = max(window_y0, 0)
                src_col_start = max(window_x0, 0)
                src_row_stop = min(window_y0 + self.window_size, self.height)
                src_col_stop = min(window_x0 + self.window_size, self.width)

                read_h = src_row_stop - src_row_start
                read_w = src_col_stop - src_col_start

                if read_h > 0 and read_w > 0:
                    valid_window = Window(
                        col_off=src_col_start,
                        row_off=src_row_start,
                        width=read_w,
                        height=read_h,
                    )

                    real_data = src.read(window=valid_window).astype(self.dtype)

                    dst_row_start = src_row_start - window_y0
                    dst_col_start = src_col_start - window_x0
                    dst_row_stop = dst_row_start + read_h
                    dst_col_stop = dst_col_start + read_w

                    data[
                        :,
                        dst_row_start:dst_row_stop,
                        dst_col_start:dst_col_stop,
                    ] = real_data
                    
        data = data.astype(self.dtype)

        # 5. Apply normalizations
        if self.transform:
            data = self.transform(data)
            
        if not isinstance(data, torch.Tensor):
            data = torch.from_numpy(data)

        # 6. Package the spatial metadata
        # The Stitcher needs these coordinates to know exactly where to write the center crop
        # back into the global Probability Memmap.
        spatial_metadata = {
            "valid_y0": valid_y0,
            "valid_x0": valid_x0,
            "valid_h": valid_h,
            "valid_w": valid_w,
            "buffer_size": self.buffer_size
        }
        
        return data, spatial_metadata