"""Bit-identity check: ProbabilityStitcher CPU-memmap path vs device path.

The device path (ProbabilityStitcher(device="cuda")) must produce byte-identical
output files to the legacy CPU path for every mode/precision/window combination,
because inference sweeps are compared bit-for-bit for provenance.

Run as pytest or directly:

    python scripts/tests/test_stitcher_device_parity.py            # parity checks
    python scripts/tests/test_stitcher_device_parity.py --bench    # + micro-benchmark

Requires CUDA; parity checks are skipped (exit 0 with a message) without it.
"""

import argparse
import itertools
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch


def _import_stitcher():
    """Import ProbabilityStitcher from the repo, or from a stitcher.py copied
    next to this file (used to test on a machine without the full repo)."""
    local = Path(__file__).resolve().parent / "stitcher.py"
    if local.exists():
        import importlib.util

        spec = importlib.util.spec_from_file_location("stitcher_local", local)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.ProbabilityStitcher

    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root))
    from src.utils.stitcher import ProbabilityStitcher  # noqa: PLC0415
    return ProbabilityStitcher


ProbabilityStitcher = _import_stitcher()


def _tile_origins(extent: int, tile: int, stride: int) -> list[int]:
    """Origins covering [0, extent) with shift_inward-style final tile."""
    origins = list(range(0, extent - tile + 1, stride))
    if not origins or origins[-1] + tile < extent:
        origins.append(extent - tile)
    return origins


def _make_batches(shape, tile, stride, buffer, batch_size, seed, include_slivers):
    """Yield (batch_probs, metadata) covering the full canvas."""
    rng = np.random.default_rng(seed)
    tiles = [
        (y0, x0, tile, tile)
        for y0 in _tile_origins(shape[0], tile, stride)
        for x0 in _tile_origins(shape[1], tile, stride)
    ]
    if include_slivers:
        # A few odd-sized valid regions (allow_sliver-style) to exercise
        # multiple weight-cache keys. Overlap existing coverage.
        tiles += [(0, 0, tile // 2, tile), (shape[0] - tile, 0, tile, tile // 3 * 2)]

    padded = tile + 2 * buffer
    for start in range(0, len(tiles), batch_size):
        chunk = tiles[start:start + batch_size]
        probs = rng.random((len(chunk), padded, padded), dtype=np.float32)
        metadata = {
            "valid_y0": torch.tensor([t[0] for t in chunk]),
            "valid_x0": torch.tensor([t[1] for t in chunk]),
            "valid_h": torch.tensor([t[2] for t in chunk]),
            "valid_w": torch.tensor([t[3] for t in chunk]),
            "buffer_size": torch.tensor([buffer] * len(chunk)),
        }
        yield torch.from_numpy(probs), metadata


def _run_stitcher(out_path, device, *, shape, precision, mode, window, keep, batches):
    stitcher = ProbabilityStitcher(
        output_path=str(out_path),
        shape=shape,
        precision=precision,
        mode=mode,
        blend_window=window,
        min_weight=1e-3,
        keep_accumulator_memmaps=keep,
        device=device,
    )
    for probs, metadata in batches:
        stitcher.add_batch(probs.to(device) if device else probs, metadata)
    stitcher.close()


def _assert_files_equal(a: Path, b: Path, label: str):
    ba, bb = a.read_bytes(), b.read_bytes()
    if ba == bb:
        return
    arr_a = np.frombuffer(ba, dtype=np.float32)
    arr_b = np.frombuffer(bb, dtype=np.float32)
    detail = ""
    if arr_a.shape == arr_b.shape:
        detail = f" | max abs diff {np.abs(arr_a - arr_b).max():.3e}"
    raise AssertionError(f"[{label}] device output differs from CPU output{detail}")


def test_device_parity():
    if not torch.cuda.is_available():
        print("SKIP: CUDA not available; device parity not checked.")
        return

    shape = (300, 421)
    tile, stride, buffer = 64, 17, 8

    cases = list(itertools.product(
        ["float32", "float16"],
        ["weighted_blend", "crop_only"],
        ["hann", "linear", "constant"],
    ))
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for idx, (precision, mode, window) in enumerate(cases):
            keep = mode == "weighted_blend"  # also compare accumulator memmaps
            common = dict(
                shape=shape, precision=precision, mode=mode, window=window, keep=keep
            )
            batch_args = (shape, tile, stride, buffer, 32, 1234 + idx, True)

            cpu_out = tmp / f"case{idx}_cpu.memmap"
            gpu_out = tmp / f"case{idx}_gpu.memmap"
            _run_stitcher(cpu_out, None, batches=_make_batches(*batch_args), **common)
            _run_stitcher(gpu_out, "cuda", batches=_make_batches(*batch_args), **common)

            label = f"{precision}/{mode}/{window}"
            _assert_files_equal(cpu_out, gpu_out, label)
            if keep:
                for suffix in (".sum.float32.memmap", ".weight.float32.memmap"):
                    _assert_files_equal(
                        Path(str(cpu_out) + suffix),
                        Path(str(gpu_out) + suffix),
                        label + suffix,
                    )
            print(f"OK  {label}: byte-identical")
    print(f"PASS: all {len(cases)} cases byte-identical between CPU and device paths.")


def bench():
    if not torch.cuda.is_available():
        print("SKIP bench: CUDA not available.")
        return
    # Production-like: 224px tiles, stride 32, fp32 batch on GPU already.
    shape = (2000, 2000)
    tile, stride, buffer = 224, 32, 0
    with tempfile.TemporaryDirectory() as tmp:
        for device in (None, "cuda"):
            batches = [
                (p.cuda(), m)
                for p, m in _make_batches(shape, tile, stride, buffer, 256, 7, False)
            ]
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            _run_stitcher(
                Path(tmp) / f"bench_{device}.memmap", device,
                shape=shape, precision="float16", mode="weighted_blend",
                window="hann", keep=False, batches=batches,
            )
            torch.cuda.synchronize()
            n_tiles = sum(len(m["valid_h"]) for _, m in batches)
            elapsed = time.perf_counter() - t0
            print(f"{device or 'cpu-memmap':11s}: {elapsed:6.2f}s total, "
                  f"{1e3 * elapsed / n_tiles:.3f} ms/tile ({n_tiles} tiles)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bench", action="store_true")
    args = parser.parse_args()
    test_device_parity()
    if args.bench:
        bench()
