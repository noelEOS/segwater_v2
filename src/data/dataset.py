import os
from dataclasses import dataclass
from typing import Optional, Callable, Tuple
import logging

import numpy as np
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

@dataclass
class MemmapSpec:
    path: str
    H: int = 224
    W: int = 224
    in_channels: int = 2         # VV, VH
    mask_channel_index: int = 2  # third plane holds the target
    dtype: np.dtype = np.float32

class CoastalMemmapDataset(Dataset):
    """Reads samples from a .memmap with shape (N, 3, H, W):
       [0]=VV (float32), [1]=VH (float32), [2]=mask in {0,1,255} float32.
       Returns dict(pixel_values: FloatTensor [2,H,W], labels: LongTensor [H,W]).
    """

    def __init__(
        self,
        spec: MemmapSpec,
        transforms: Optional[Callable[[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]] = None,
    ):
        super().__init__()
        self.spec = spec
        self.transforms = transforms
        self._mm = None  # lazily opened per process
        self._length = self._compute_length()

    def _bytes_per_sample(self) -> int:
        c = self.spec.in_channels + 1  # +1 for mask plane
        return c * self.spec.H * self.spec.W * np.dtype(self.spec.dtype).itemsize

    def _compute_length(self) -> int:
        size = os.path.getsize(self.spec.path)
        bps = self._bytes_per_sample()
        if size % bps != 0:
            raise ValueError(
                f"Memmap size not divisible by sample size. path={self.spec.path} size={size} bytes_per_sample={bps}"
            )
        n = size // bps
        if n == 0:
            raise ValueError(f"No samples found in memmap: {self.spec.path}")
        return int(n)

    def _ensure_open(self):
        if self._mm is None:
            shape = (self._length, self.spec.in_channels + 1, self.spec.H, self.spec.W)
            self._mm = np.memmap(self.spec.path, dtype=self.spec.dtype, mode="r", shape=shape)
            logger.info(f"Opened memmap {self.spec.path} with shape={shape}")

    def __len__(self):
        return self._length

    def __getitem__(self, idx: int):
        self._ensure_open()
        arr = self._mm[idx]  # (3,H,W)
        x_np = arr[: self.spec.in_channels]  # (2,H,W)
        y_np = arr[self.spec.mask_channel_index]  # (H,W)
        
        # Make a writable copy to avoid PyTorch warning about non-writable NumPy
        x = torch.from_numpy(np.array(x_np, copy=True))  # float32 tensor, owns memory
        # keep 255 as ignore; convert to long without copying where possible
        y = torch.from_numpy(y_np.astype(np.int64, copy=False))
        
        if self.transforms is not None:
            x, y = self.transforms(x, y)
        
        return {"pixel_values": x, "labels": y}

    def close(self):
        if getattr(self, "_mm", None) is not None:
            try:
                # close the underlying mmap explicitly
                self._mm._mmap.close()
            except Exception:
                pass
            self._mm = None
