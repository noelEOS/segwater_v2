from collections import OrderedDict

import torch

from src.utils.tta import (
    apply_tta_transform,
    invert_tta_transform,
    predict_with_tta,
    validate_tta_transforms,
)


def test_tta_flip_transforms_are_self_inverse():
    x = torch.arange(2 * 1 * 3 * 4, dtype=torch.float32).reshape(2, 1, 3, 4)

    for transform in ["identity", "hflip", "vflip", "hvflip"]:
        transformed = apply_tta_transform(x, transform)
        restored = apply_tta_transform(transformed, transform)
        assert torch.equal(restored, x)


def test_probability_inverse_transforms_are_self_inverse():
    probs = torch.arange(2 * 3 * 4, dtype=torch.float32).reshape(2, 3, 4)

    for transform in ["identity", "hflip", "vflip", "hvflip"]:
        transformed = invert_tta_transform(probs, transform)
        restored = invert_tta_transform(transformed, transform)
        assert torch.equal(restored, probs)


class MeanIntensityModel(torch.nn.Module):
    def forward(self, x):
        # Return one binary-logit channel. This model is flip-equivariant because
        # the logit at each pixel is just the channel mean at that same pixel.
        return x.mean(dim=1, keepdim=True)


def test_predict_with_tta_returns_mean_and_individual_maps():
    model = MeanIntensityModel()
    images = torch.randn(2, 3, 5, 7)

    mean_probs, individual = predict_with_tta(
        model,
        images,
        num_classes=1,
        transforms=["identity", "hflip", "vflip", "hvflip"],
        return_individual=True,
    )

    expected = torch.sigmoid(images.mean(dim=1))

    assert isinstance(individual, OrderedDict)
    assert list(individual.keys()) == ["identity", "hflip", "vflip", "hvflip"]
    assert torch.allclose(mean_probs, expected, atol=1e-6)

    for probs in individual.values():
        assert torch.allclose(probs, expected, atol=1e-6)


def test_validate_tta_transforms_deduplicates_and_rejects_invalid():
    assert validate_tta_transforms(["identity", "hflip", "hflip"]) == ["identity", "hflip"]

    try:
        validate_tta_transforms(["identity", "rot90"])
    except ValueError as exc:
        assert "Unsupported TTA transform" in str(exc)
    else:
        raise AssertionError("Expected invalid transform to raise ValueError")
