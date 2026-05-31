from collections import OrderedDict
from typing import Iterable

import torch

SUPPORTED_TTA_TRANSFORMS = {"identity", "hflip", "vflip", "hvflip"}


def validate_tta_transforms(transforms: Iterable[str]) -> list[str]:
    """Validate and return a de-duplicated list of TTA transform names."""
    valid = []
    seen = set()

    for transform in transforms:
        if transform not in SUPPORTED_TTA_TRANSFORMS:
            raise ValueError(
                f"Unsupported TTA transform={transform!r}. "
                f"Supported transforms: {sorted(SUPPORTED_TTA_TRANSFORMS)}"
            )
        if transform not in seen:
            valid.append(transform)
            seen.add(transform)

    if not valid:
        raise ValueError("At least one TTA transform must be provided when TTA is enabled.")

    return valid


def apply_tta_transform(images: torch.Tensor, transform: str) -> torch.Tensor:
    """Apply a test-time augmentation transform to an image batch.

    Expected image shape: (B, C, H, W).
    """
    if transform == "identity":
        return images
    if transform == "hflip":
        return torch.flip(images, dims=(-1,))
    if transform == "vflip":
        return torch.flip(images, dims=(-2,))
    if transform == "hvflip":
        return torch.flip(images, dims=(-2, -1))

    raise ValueError(f"Unsupported TTA transform={transform!r}.")


def invert_tta_transform(probs: torch.Tensor, transform: str) -> torch.Tensor:
    """Invert a TTA transform on probability maps.

    Expected probability shape: (B, H, W). The supported flip transforms are
    self-inverse, so inversion is the same operation as application.
    """
    if transform == "identity":
        return probs
    if transform == "hflip":
        return torch.flip(probs, dims=(-1,))
    if transform == "vflip":
        return torch.flip(probs, dims=(-2,))
    if transform == "hvflip":
        return torch.flip(probs, dims=(-2, -1))

    raise ValueError(f"Unsupported TTA transform={transform!r}.")


def logits_to_water_probability(logits: torch.Tensor, num_classes: int) -> torch.Tensor:
    """Convert segmentation logits to a water-probability map.

    For one-channel binary models, sigmoid is used. For multi-class models, class
    index 1 is treated as the water class, matching the existing inference code.
    """
    if num_classes == 1:
        return torch.sigmoid(logits).squeeze(1)

    return torch.softmax(logits, dim=1)[:, 1, :, :]


def predict_once(model: torch.nn.Module, images: torch.Tensor, num_classes: int) -> torch.Tensor:
    """Run a single forward pass and return water probabilities."""
    logits = model(images)
    return logits_to_water_probability(logits, num_classes=num_classes)


def predict_with_tta(
    model: torch.nn.Module,
    images: torch.Tensor,
    num_classes: int,
    transforms: Iterable[str],
    return_individual: bool = False,
) -> tuple[torch.Tensor, OrderedDict[str, torch.Tensor]] | torch.Tensor:
    """Run TTA inference and average inverse-transformed probabilities.

    TTA is averaged at the tile level. The returned probability tile is already
    in the original tile orientation and can be passed directly to the stitcher.

    Args:
        model: Segmentation model in eval mode.
        images: Input tensor with shape (B, C, H, W).
        num_classes: Number of output classes configured for the model.
        transforms: Sequence of transform names, e.g. identity/hflip/vflip/hvflip.
        return_individual: If true, also return inverse-transformed probability
            maps for each TTA view. This is useful for development/QGIS exports.
    """
    valid_transforms = validate_tta_transforms(transforms)
    individual = OrderedDict()
    prob_sum = None

    for transform in valid_transforms:
        aug_images = apply_tta_transform(images, transform)
        aug_probs = predict_once(model, aug_images, num_classes=num_classes)
        probs = invert_tta_transform(aug_probs, transform)

        individual[transform] = probs
        prob_sum = probs if prob_sum is None else prob_sum + probs

    mean_probs = prob_sum / len(valid_transforms)

    if return_individual:
        return mean_probs, individual

    return mean_probs
