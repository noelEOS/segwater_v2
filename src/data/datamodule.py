import gc
import os
import random
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.dataset import CoastalMemmapDataset, MemmapSpec
from src.data.transforms import CoastalAug


def _seed_worker(worker_id: int):
    """Make each DataLoader worker's RNG deterministic given the base generator.

    torch seeds each worker's `torch.initial_seed()` from the loader generator +
    worker id; we derive numpy/random from it so any numpy/random-based transform
    is reproducible across runs with the same base seed.
    """
    ws = torch.initial_seed() % 2 ** 32
    np.random.seed(ws)
    random.seed(ws)


class CoastalDataModule:
    """Pure Python DataModule orchestrating Memmap datasets."""
    
    def __init__(
        self,
        root_dir: str,
        train_file: str = "train.memmap",
        val_file: str = "val.memmap",
        test_file: str = "test.memmap",
        H: int = 224,
        W: int = 224,
        batch_size: int = 16,
        val_batch_size: Optional[int] = None,
        num_workers: int = 4,
        pin_memory: bool = True,
        persistent_workers: bool = True,
        augment: bool = True,
        aug_params: Optional[dict] = None,
        seed: Optional[int] = None,
        dtype: str = "float32",
        multiprocessing_context: Optional[str] = None,
    ):
        self.root_dir = root_dir
        self.train_path = os.path.join(root_dir, train_file)
        self.val_path = os.path.join(root_dir, val_file)
        self.test_path = os.path.join(root_dir, test_file)
        self.H = H
        self.W = W
        # On-disk storage dtype of the memmaps under root_dir. "float32" for the
        # chip-based/pair-based datasets; "float16" for mixed80_blocked. Samples are
        # returned as float32 either way. A wrong value raises in _compute_length
        # (size % bytes_per_sample != 0) rather than silently misreading.
        self.dtype = np.dtype(dtype)
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size or batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers and (num_workers > 0)
        # Explicit "spawn" is used by optimize.py so HPO workers start in clean
        # interpreters rather than inheriting the parent's initialized CUDA
        # context through Linux fork. None preserves all non-HPO call sites.
        self.multiprocessing_context = multiprocessing_context
        self.augment = augment
        # Seed for the TRAIN loader's shuffle generator + worker RNGs. None keeps
        # loader construction identical to the pre-seeding behaviour (train.py).
        self.seed = seed
        # Per-aug probabilities forwarded to CoastalAug when augment is enabled.
        # None -> {} -> CoastalAug's own signature defaults.
        self.aug_params = aug_params or {}
        self.train_ds = None
        self.val_ds = None
        self.test_ds = None
        # Track every DataLoader handed out so teardown can join their worker
        # processes (otherwise persistent workers from one Optuna trial outlive
        # the trial and accumulate across the sweep).
        self._loaders = []

    def setup(self):
        """Initializes dataset objects (but delays memmap opening per process)."""
        aug = CoastalAug(**self.aug_params) if self.augment else None
        
        if os.path.exists(self.train_path):
            self.train_ds = CoastalMemmapDataset(
                MemmapSpec(self.train_path, H=self.H, W=self.W, dtype=self.dtype), transforms=aug)
        if os.path.exists(self.val_path):
            self.val_ds = CoastalMemmapDataset(
                MemmapSpec(self.val_path, H=self.H, W=self.W, dtype=self.dtype), transforms=None)
        if os.path.exists(self.test_path):
            self.test_ds = CoastalMemmapDataset(
                MemmapSpec(self.test_path, H=self.H, W=self.W, dtype=self.dtype), transforms=None)

    def _dl(self, dataset, batch_size, shuffle=False):
        if dataset is None:
            return None
        kwargs = dict(
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )
        if self.num_workers > 0:
            kwargs["prefetch_factor"] = 2
            if self.multiprocessing_context is not None:
                kwargs["multiprocessing_context"] = self.multiprocessing_context

        # Seed only the shuffling (train) loader: a fixed generator makes the
        # shuffle order reproducible and worker_init_fn pins per-worker RNGs.
        # val/test are shuffle=False, so they are untouched. seed=None keeps
        # construction byte-identical to the pre-seeding behaviour.
        if shuffle and self.seed is not None:
            kwargs["generator"] = torch.Generator().manual_seed(self.seed)
            kwargs["worker_init_fn"] = _seed_worker

        loader = DataLoader(dataset, **kwargs)
        self._loaders.append(loader)
        return loader

    def train_dataloader(self):
        return self._dl(self.train_ds, self.batch_size, shuffle=True)

    def val_dataloader(self):
        return self._dl(self.val_ds, self.val_batch_size, shuffle=False)

    def test_dataloader(self):
        return self._dl(self.test_ds, self.val_batch_size, shuffle=False)

    def teardown(self):
        """Release DataLoader workers and close open memmaps.

        Persistent workers keep their processes (and memmap file handles) alive
        for the lifetime of the DataLoader. In an Optuna sweep a fresh
        DataModule / set of loaders is built every trial, so without an explicit
        shutdown those worker pools leak across trials and can eventually
        deadlock. Join the workers, drop loader references, then close memmaps.
        """
        for loader in self._loaders:
            # Shut down the live iterator's worker pool if one exists. DataLoader
            # exposes the persistent-workers iterator as `_iterator`.
            iterator = getattr(loader, "_iterator", None)
            if iterator is not None:
                shutdown = getattr(iterator, "_shutdown_workers", None)
                if callable(shutdown):
                    shutdown()
                loader._iterator = None
        self._loaders.clear()
        # Force collection so any DataLoader whose iterator we did not hold is
        # finalized (its __del__ joins remaining workers) before the next trial.
        gc.collect()

        for ds in (self.train_ds, self.val_ds, self.test_ds):
            if isinstance(ds, CoastalMemmapDataset):
                ds.close()
