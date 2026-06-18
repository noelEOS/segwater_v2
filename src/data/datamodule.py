import gc
import os
from typing import Optional
from torch.utils.data import DataLoader

from src.data.dataset import CoastalMemmapDataset, MemmapSpec
from src.data.transforms import CoastalAug

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
    ):
        self.root_dir = root_dir
        self.train_path = os.path.join(root_dir, train_file)
        self.val_path = os.path.join(root_dir, val_file)
        self.test_path = os.path.join(root_dir, test_file)
        self.H = H
        self.W = W
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size or batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers and (num_workers > 0)
        self.augment = augment
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
            self.train_ds = CoastalMemmapDataset(MemmapSpec(self.train_path, H=self.H, W=self.W), transforms=aug)
        if os.path.exists(self.val_path):
            self.val_ds = CoastalMemmapDataset(MemmapSpec(self.val_path, H=self.H, W=self.W), transforms=None)
        if os.path.exists(self.test_path):
            self.test_ds = CoastalMemmapDataset(MemmapSpec(self.test_path, H=self.H, W=self.W), transforms=None)

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
