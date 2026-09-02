"""data.dtype must agree with the memmap directory's build_manifest.json.

An fp16 array read as fp32 passes the file-size divisibility check whenever
N is even and silently halves the sample count. The v3 / mixed80 builders
record the dtype beside the arrays; the datamodule must refuse a contradiction
and stay silent when no record exists (legacy lineages).
"""
import json

import numpy as np
import pytest

from src.data.datamodule import CoastalDataModule


def _write(root, dtype, n=4, h=2, w=2):
    for name in ("train", "val"):
        mm = np.memmap(root / f"{name}.memmap", dtype=dtype, mode="w+", shape=(n, 3, h, w))
        mm[:] = 0
        mm.flush()


def _dm(root, dtype):
    return CoastalDataModule(root_dir=str(root), H=2, W=2, dtype=dtype, batch_size=2,
                             num_workers=0, persistent_workers=False, augment=False)


def test_matching_dtype_passes(tmp_path):
    _write(tmp_path, np.float16)
    (tmp_path / "build_manifest.json").write_text(json.dumps({"dtype": "float16"}))
    dm = _dm(tmp_path, "float16")
    dm.setup()
    assert len(dm.train_ds) == 4


def test_contradicting_dtype_raises_even_when_size_divides(tmp_path):
    _write(tmp_path, np.float16, n=4)  # 4 fp16 samples == 2 fp32 samples: size check alone passes
    (tmp_path / "build_manifest.json").write_text(json.dumps({"dtype": "float16"}))
    with pytest.raises(ValueError, match="records dtype='float16'"):
        _dm(tmp_path, "float32").setup()


def test_no_manifest_falls_back_to_size_check(tmp_path):
    _write(tmp_path, np.float32)
    dm = _dm(tmp_path, "float32")
    dm.setup()
    assert len(dm.train_ds) == 4
