import logging
from typing import Callable, Dict, Optional, Tuple

import numpy as np
import rasterio
from rasterio.windows import Window
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


def _build_axis_starts(
    length: int,
    crop_size: int,
    stride: int,
    edge_policy: str,
) -> list[int]:
    """Build deterministic tile origins for one spatial axis.

    ``shift_inward`` avoids tiny final slivers by appending a final full-size crop
    whose trailing edge coincides with the raster boundary. For example, for
    length=2176 and crop_size=1024, the final origin becomes 1152 instead of
    2048. This prevents edge tiles from being dominated by artificial padding.

    ``allow_sliver`` preserves the previous behavior where the final tile starts
    at the next stride location and may be smaller than ``crop_size``.
    """
    if length <= 0:
        raise ValueError(f"Axis length must be positive, got {length}.")
    if crop_size <= 0:
        raise ValueError(f"crop_size must be positive, got {crop_size}.")
    if stride <= 0:
        raise ValueError(f"stride must be positive, got {stride}.")
    if edge_policy not in {"shift_inward", "allow_sliver"}:
        raise ValueError(
            f"Unsupported edge_policy={edge_policy!r}. "
            "Expected 'shift_inward' or 'allow_sliver'."
        )

    if length <= crop_size:
        return [0]

    if edge_policy == "allow_sliver":
        return list(range(0, length, stride))

    final_start = length - crop_size
    starts = list(range(0, final_start + 1, stride))

    if starts[-1] != final_start:
        starts.append(final_start)

    return sorted(set(starts))


class InferenceDataset(Dataset):
    """
    Dataset for sliding-window inference on large SAR swaths.

    The dataset separates three pieces of inference geometry:

    - ``tile_size``: the valid crop size written to the output probability map.
    - ``buffer_size``: contextual padding added on each side of the valid crop.
    - ``stride``: the spacing between valid crop origins.

    The model input size is therefore ``tile_size + 2 * buffer_size``. Setting
    ``stride < tile_size`` enables overlapping valid zones, which can later be
    averaged or weighted by the stitcher.
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
        stride: Optional[int] = None,
        edge_policy: str = "shift_inward",
    ):
        """
        Args:
            image_path: Path to the input full-swath GeoTIFF.
            tile_size: The dimension of the valid center crop written to the
                final map, e.g. 1024. For 224-native models, a typical setting
                is tile_size=160 and buffer_size=32, giving a 224x224 model
                input while only trusting the center crop strongly.
            buffer_size: Context added to all sides of the valid crop. Total
                model input size is ``tile_size + 2 * buffer_size``.
            fill_value: Scalar NoData padding fallback for out-of-bounds reads.
            precision: Either ``float32`` or ``float16``.
            transform: Optional transform, e.g. normalization.
            channel_fill_values: Optional per-channel fill values used when the
                buffered window extends beyond the GeoTIFF.
            stride: Spacing between valid crop origins. Defaults to
                ``tile_size``. Use smaller values for high-precision overlap.
            edge_policy: ``shift_inward`` avoids tiny final slivers; 
                ``allow_sliver`` preserves the previous behavior.
        """
        self.image_path = image_path
        self.transform = transform
        self.tile_size = int(tile_size)
        self.buffer_size = int(buffer_size)
        self.stride = self.tile_size if stride is None else int(stride)
        self.edge_policy = edge_policy
        self.fill_value = fill_value
        self.channel_fill_values = channel_fill_values

        if self.tile_size <= 0:
            raise ValueError(f"tile_size must be positive, got {self.tile_size}.")
        if self.buffer_size < 0:
            raise ValueError(f"buffer_size must be non-negative, got {self.buffer_size}.")
        if self.stride <= 0:
            raise ValueError(f"stride must be positive, got {self.stride}.")
        if self.stride > self.tile_size:
            raise ValueError(
                f"stride ({self.stride}) must be <= tile_size ({self.tile_size}) "
                "to avoid gaps in the probability map."
            )

        if precision == "float32":
            self.dtype = np.float32
        elif precision == "float16":
            self.dtype = np.float16
        else:
            raise ValueError(f"Unsupported precision: {precision}. Must be 'float32' or 'float16'.")

        # Full tensor size fed to the model, e.g. 1024 + 2*128 = 1280.
        self.window_size = self.tile_size + (2 * self.buffer_size)

        # Extract swath dimensions.
        with rasterio.open(self.image_path) as src:
            self.height = src.height
            self.width = src.width
            self.num_channels = src.count

        # Build edge-safe, stride-aware valid crop origins.
        self.y_starts = _build_axis_starts(
            length=self.height,
            crop_size=self.tile_size,
            stride=self.stride,
            edge_policy=self.edge_policy,
        )
        self.x_starts = _build_axis_starts(
            length=self.width,
            crop_size=self.tile_size,
            stride=self.stride,
            edge_policy=self.edge_policy,
        )

        self.tiles_y = len(self.y_starts)
        self.tiles_x = len(self.x_starts)
        self.total_tiles = self.tiles_y * self.tiles_x

        logger.info(f"Initialized InferenceDataset for {image_path}")
        logger.info(f"Swath dimensions: {self.width}x{self.height}")
        logger.info(
            "Tiling geometry: valid_crop=%s, buffer=%s, model_input=%s, "
            "stride=%s, edge_policy=%s",
            self.tile_size,
            self.buffer_size,
            self.window_size,
            self.stride,
            self.edge_policy,
        )
        logger.info(f"Grid: {self.tiles_x}x{self.tiles_y} tiles (Total: {self.total_tiles})")
        logger.info(f"Input tensor shape: {self.num_channels}x{self.window_size}x{self.window_size}")

    def __len__(self) -> int:
        return self.total_tiles

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, int]]:
        # 1. Map 1D index back to 2D grid coordinates.
        tile_y_idx = idx // self.tiles_x
        tile_x_idx = idx % self.tiles_x

        # 2. Look up the valid center crop origin.
        valid_y0 = self.y_starts[tile_y_idx]
        valid_x0 = self.x_starts[tile_x_idx]

        # Edge-safe shifted tiles are usually full-size; small rasters or the
        # legacy allow_sliver policy can still produce smaller valid regions.
        valid_h = min(self.tile_size, self.height - valid_y0)
        valid_w = min(self.tile_size, self.width - valid_x0)

        # 3. Calculate coordinates for the buffered model input window.
        # These can be negative near the top/left raster boundary.
        window_y0 = valid_y0 - self.buffer_size
        window_x0 = valid_x0 - self.buffer_size

        # 4. Read the buffered window from the GeoTIFF.
        # Re-open per worker for PyTorch DataLoader safety.
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

        # 5. Apply normalization or other configured transforms.
        if self.transform:
            data = self.transform(data)

        if not isinstance(data, torch.Tensor):
            data = torch.from_numpy(data)

        # 6. Package the spatial metadata.
        spatial_metadata = {
            "valid_y0": valid_y0,
            "valid_x0": valid_x0,
            "valid_h": valid_h,
            "valid_w": valid_w,
            "buffer_size": self.buffer_size,
        }

        return data, spatial_metadata
