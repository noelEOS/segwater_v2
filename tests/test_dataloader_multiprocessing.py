import numpy as np
import torch

from src.data.datamodule import CoastalDataModule


def _write_memmap(path, *, n=4, h=2, w=2):
    data = np.zeros((n, 3, h, w), dtype=np.float16)
    data[:, 0] = 1.25
    data[:, 1] = -2.5
    data[1::2, 2] = 1
    memmap = np.memmap(path, dtype=np.float16, mode="w+", shape=data.shape)
    memmap[:] = data
    memmap.flush()
    del memmap


def test_spawn_worker_reads_fp16_memmap_and_tears_down(tmp_path):
    _write_memmap(tmp_path / "train.memmap")
    # Reproduce optimize.py's ordering on CUDA hosts: the parent initializes
    # CUDA before DataLoader workers start. A forked worker would inherit that
    # runtime state; a spawned worker starts clean.
    if torch.cuda.is_available():
        torch.empty(1, device="cuda")

    datamodule = CoastalDataModule(
        root_dir=str(tmp_path),
        H=2,
        W=2,
        batch_size=2,
        num_workers=1,
        pin_memory=False,
        persistent_workers=True,
        augment=False,
        seed=42,
        dtype="float16",
        multiprocessing_context="spawn",
    )
    datamodule.setup()
    loader = datamodule.train_dataloader()

    try:
        batch = next(iter(loader))
        assert loader.multiprocessing_context.get_start_method() == "spawn"
        assert batch["pixel_values"].dtype == torch.float32
        assert batch["labels"].dtype == torch.int64
        assert batch["pixel_values"].shape == (2, 2, 2, 2)
        assert batch["labels"].shape == (2, 2, 2)
    finally:
        datamodule.teardown()

    assert loader._iterator is None


def test_default_context_preserves_existing_dataloader_behavior(tmp_path):
    _write_memmap(tmp_path / "train.memmap")
    datamodule = CoastalDataModule(
        root_dir=str(tmp_path),
        H=2,
        W=2,
        batch_size=2,
        num_workers=1,
        pin_memory=False,
        persistent_workers=False,
        augment=False,
        dtype="float16",
    )
    datamodule.setup()
    loader = datamodule.train_dataloader()

    assert loader.multiprocessing_context is None
    datamodule.teardown()
