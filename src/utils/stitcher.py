import logging
import os
from typing import Dict

import numpy as np
import torch

logger = logging.getLogger(__name__)


def _resolve_dtype(precision: str) -> np.dtype:
    if precision == "float32":
        return np.float32
    if precision == "float16":
        return np.float16
    raise ValueError(f"Unsupported precision: {precision}. Must be 'float32' or 'float16'.")


def _make_1d_blend_weights(length: int, window: str, min_weight: float) -> np.ndarray:
    """Create one-dimensional nonzero blend weights.

    The weights are intentionally bounded by ``min_weight`` so that valid pixels
    near the true raster boundary are not assigned zero contribution when a
    tapered window is used.
    """
    if length <= 0:
        raise ValueError(f"length must be positive, got {length}.")
    if min_weight <= 0:
        raise ValueError(f"min_weight must be > 0, got {min_weight}.")
    if min_weight > 1:
        raise ValueError(f"min_weight must be <= 1, got {min_weight}.")

    if window == "constant" or length == 1:
        weights = np.ones(length, dtype=np.float32)
    elif window == "hann":
        weights = np.hanning(length).astype(np.float32)
    elif window == "linear":
        center = (length - 1) / 2.0
        distance = np.abs(np.arange(length, dtype=np.float32) - center)
        max_distance = max(center, 1.0)
        weights = 1.0 - (distance / max_distance)
    else:
        raise ValueError(
            f"Unsupported blend_window={window!r}. "
            "Expected 'constant', 'hann', or 'linear'."
        )

    weights = np.maximum(weights, min_weight)
    weights /= weights.max()
    return weights.astype(np.float32)


def make_blend_weight_2d(
    height: int,
    width: int,
    window: str = "hann",
    min_weight: float = 1e-3,
) -> np.ndarray:
    """Create a 2D separable blending weight map for one valid crop."""
    wy = _make_1d_blend_weights(height, window=window, min_weight=min_weight)
    wx = _make_1d_blend_weights(width, window=window, min_weight=min_weight)
    return (wy[:, None] * wx[None, :]).astype(np.float32)


class ProbabilityStitcher:
    """
    Constructs a global probability map out-of-core using np.memmap.

    Supported modes:

    - ``crop_only``: current behavior. The valid crop is written directly into
      the global probability canvas. If edge-safe shifted tiles overlap, later
      tiles overwrite earlier values.
    - ``weighted_blend``: overlapping valid crops are accumulated with a blend
      window and normalized at close time. This is the high-precision mode for
      architectures that show tile-boundary artifacts.
    """

    def __init__(
        self,
        output_path: str,
        shape: tuple[int, int],
        precision: str,
        mode: str = "crop_only",
        blend_window: str = "hann",
        min_weight: float = 1e-3,
    ):
        """
        Args:
            output_path: Destination path for the final .memmap file.
            shape: Tuple of (height, width) matching the original SAR swath.
            precision: Output probability map dtype, ``float32`` or ``float16``.
            mode: ``crop_only`` or ``weighted_blend``.
            blend_window: Weight window for weighted mode: ``hann``, ``linear``,
                or ``constant``.
            min_weight: Lower bound for taper weights to avoid zero-coverage
                pixels near true image boundaries.
        """
        self.output_path = output_path
        self.shape = shape
        self.dtype = _resolve_dtype(precision)
        self.mode = mode
        self.blend_window = blend_window
        self.min_weight = float(min_weight)

        if self.mode not in {"crop_only", "weighted_blend"}:
            raise ValueError(
                f"Unsupported stitcher mode={self.mode!r}. "
                "Expected 'crop_only' or 'weighted_blend'."
            )

        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)

        # The final probability canvas. 'w+' creates or overwrites cleanly.
        self.memmap = np.memmap(
            self.output_path,
            dtype=self.dtype,
            mode="w+",
            shape=self.shape,
        )

        self.sum_memmap = None
        self.weight_memmap = None

        if self.mode == "weighted_blend":
            self.sum_path = f"{self.output_path}.sum.float32.memmap"
            self.weight_path = f"{self.output_path}.weight.float32.memmap"
            self.sum_memmap = np.memmap(
                self.sum_path,
                dtype=np.float32,
                mode="w+",
                shape=self.shape,
            )
            self.weight_memmap = np.memmap(
                self.weight_path,
                dtype=np.float32,
                mode="w+",
                shape=self.shape,
            )
            self.sum_memmap[:] = 0.0
            self.weight_memmap[:] = 0.0

        logger.info(
            "Initialized Global Probability Canvas: shape=%s | dtype=%s | mode=%s | blend_window=%s",
            self.shape,
            self.dtype,
            self.mode,
            self.blend_window,
        )

    @staticmethod
    def _metadata_item(metadata: Dict[str, torch.Tensor], key: str, index: int) -> int:
        value = metadata[key][index]
        return int(value.item() if hasattr(value, "item") else value)

    def add_batch(self, batch_probs: torch.Tensor, metadata: Dict[str, torch.Tensor]):
        """
        Crop buffered predictions and add valid regions to the global canvas.

        Args:
            batch_probs: Tensor of shape (B, H_padded, W_padded) containing
                probabilities after sigmoid or softmax selection.
            metadata: Batched spatial coordinates from InferenceDataset.
        """
        batch_probs_np = batch_probs.detach().cpu().numpy()
        batch_size = batch_probs_np.shape[0]

        for i in range(batch_size):
            y0 = self._metadata_item(metadata, "valid_y0", i)
            x0 = self._metadata_item(metadata, "valid_x0", i)
            h = self._metadata_item(metadata, "valid_h", i)
            w = self._metadata_item(metadata, "valid_w", i)
            buffer = self._metadata_item(metadata, "buffer_size", i)

            crop_prob = batch_probs_np[i, buffer:buffer + h, buffer:buffer + w]

            if crop_prob.shape != (h, w):
                raise ValueError(
                    f"Cropped probability shape {crop_prob.shape} does not match "
                    f"metadata valid size {(h, w)}. Check tile_size/buffer/model output shape."
                )

            if self.mode == "crop_only":
                self.memmap[y0:y0 + h, x0:x0 + w] = crop_prob.astype(self.dtype, copy=False)
            else:
                weights = make_blend_weight_2d(
                    h,
                    w,
                    window=self.blend_window,
                    min_weight=self.min_weight,
                )
                self.sum_memmap[y0:y0 + h, x0:x0 + w] += crop_prob.astype(np.float32) * weights
                self.weight_memmap[y0:y0 + h, x0:x0 + w] += weights

    def _finalize_weighted_blend(self):
        zero_weight_pixels = int(np.count_nonzero(self.weight_memmap == 0))
        if zero_weight_pixels > 0:
            raise RuntimeError(
                f"Weighted blend left {zero_weight_pixels} pixels without coverage. "
                "Check stride, tile_size, and edge_policy."
            )

        normalized = self.sum_memmap / self.weight_memmap
        self.memmap[:] = normalized.astype(self.dtype)
        self.sum_memmap.flush()
        self.weight_memmap.flush()

    def close(self):
        """Flush final probability data to disk and close file handles."""
        if self.mode == "weighted_blend":
            self._finalize_weighted_blend()

        self.memmap.flush()
        del self.memmap

        if self.sum_memmap is not None:
            del self.sum_memmap
        if self.weight_memmap is not None:
            del self.weight_memmap

        logger.info(f"Probability map successfully flushed and closed at {self.output_path}")
