import numpy as np
import pytest
import torch

from src.utils.stitcher import ProbabilityStitcher


def _metadata(y0, x0, h, w, buffer):
    return {
        "valid_y0": torch.tensor([y0]),
        "valid_x0": torch.tensor([x0]),
        "valid_h": torch.tensor([h]),
        "valid_w": torch.tensor([w]),
        "buffer_size": torch.tensor([buffer]),
    }


def test_weighted_blend_averages_overlapping_predictions_with_constant_weights(tmp_path):
    """Overlapping valid zones should be normalized by accumulated weights."""
    output_path = tmp_path / "prob.memmap"
    stitcher = ProbabilityStitcher(
        output_path=str(output_path),
        shape=(4, 6),
        precision="float32",
        mode="weighted_blend",
        blend_window="constant",
    )

    first_tile = torch.full((1, 4, 4), 0.25, dtype=torch.float32)
    second_tile = torch.full((1, 4, 4), 0.75, dtype=torch.float32)

    stitcher.add_batch(first_tile, _metadata(y0=0, x0=0, h=4, w=4, buffer=0))
    stitcher.add_batch(second_tile, _metadata(y0=0, x0=2, h=4, w=4, buffer=0))
    stitcher.close()

    result = np.memmap(output_path, dtype=np.float32, mode="r", shape=(4, 6))

    expected = np.array(
        [
            [0.25, 0.25, 0.50, 0.50, 0.75, 0.75],
            [0.25, 0.25, 0.50, 0.50, 0.75, 0.75],
            [0.25, 0.25, 0.50, 0.50, 0.75, 0.75],
            [0.25, 0.25, 0.50, 0.50, 0.75, 0.75],
        ],
        dtype=np.float32,
    )

    np.testing.assert_allclose(result, expected, rtol=1e-6, atol=1e-6)


def test_weighted_blend_raises_if_stride_leaves_uncovered_pixels(tmp_path):
    """Misconfigured tiling should fail loudly instead of silently writing zeros."""
    output_path = tmp_path / "prob.memmap"
    stitcher = ProbabilityStitcher(
        output_path=str(output_path),
        shape=(4, 6),
        precision="float32",
        mode="weighted_blend",
        blend_window="constant",
    )

    tile = torch.ones((1, 4, 4), dtype=torch.float32)
    stitcher.add_batch(tile, _metadata(y0=0, x0=0, h=4, w=4, buffer=0))

    with pytest.raises(RuntimeError, match="without coverage"):
        stitcher.close()
