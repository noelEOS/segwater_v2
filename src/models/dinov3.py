"""DINOv3 ViT + linear segmentation head.

Wraps a HuggingFace DINOv3ViTModel with a single 1×1 conv head and bilinear
×16 upsample. The interface mirrors smp models: forward() accepts a raw pixel
tensor (B, C, H, W) with z-score normalised values and returns logits
(B, num_classes, H, W).

The backbone patch embedding is surgically adapted from 3→in_channels by
averaging the pretrained RGB weights along the channel axis, preserving as
much pretrained signal as possible (same strategy used by smp for in_channels≠3).

freeze_strategy controls what is trained:
  "full"  — backbone frozen, only the linear head is updated (linear probe).
  "none"  — full fine-tuning; backbone and head are both updated, with the
            backbone receiving a scaled-down LR via parameter_groups().
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from transformers import DINOv3ViTModel
except ImportError as e:
    raise ImportError(
        "DINOv3 requires transformers>=5.12.0. "
        "Install it with: pip install 'transformers>=5.12.0'"
    ) from e

FreezeStrategy = Literal["full", "none"]


def _adapt_patch_embedding(backbone: DINOv3ViTModel, in_channels: int) -> None:
    """Replace the 3-channel patch embedding with an in_channels-channel one.

    The pretrained RGB weights are averaged across the channel dimension and
    then tiled to produce a kernel of shape (hidden_size, in_channels, 16, 16).
    This is the standard weight-surgery approach used by smp and timm.

    No-op when in_channels == 3.
    """
    if in_channels == 3:
        return

    proj: nn.Conv2d = backbone.embeddings.patch_embeddings
    old_weight = proj.weight.data  # (hidden, 3, 16, 16)

    # Average across RGB → (hidden, 1, 16, 16), then repeat to in_channels
    new_weight = old_weight.mean(dim=1, keepdim=True).repeat(1, in_channels, 1, 1)

    new_proj = nn.Conv2d(
        in_channels,
        proj.out_channels,
        kernel_size=proj.kernel_size,
        stride=proj.stride,
        padding=proj.padding,
        bias=proj.bias is not None,
    )
    new_proj.weight = nn.Parameter(new_weight)
    if proj.bias is not None:
        new_proj.bias = nn.Parameter(proj.bias.data.clone())

    backbone.embeddings.patch_embeddings = new_proj


class DINOv3LinearSegmentation(nn.Module):
    """DINOv3 ViT backbone + trainable linear segmentation head.

    Args:
        hf_model_name: HuggingFace Hub model id, e.g.
            ``"facebook/dinov3-vitb16-pretrain-lvd1689m"``.
        num_classes: Number of output segmentation classes.
        in_channels: Number of input image channels. Pretrained patch embedding
            weights are averaged down from 3 channels when in_channels != 3.
        patch_size: Patch size of the ViT (default 16). Must match the
            checkpoint; used only to compute the upsample factor.
        freeze_strategy: ``"full"`` freezes the backbone (linear probe);
            ``"none"`` trains everything with differential LR via
            ``parameter_groups()``.
    """

    def __init__(
        self,
        hf_model_name: str,
        num_classes: int,
        in_channels: int = 3,
        patch_size: int = 16,
        freeze_strategy: FreezeStrategy = "none",
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.freeze_strategy: FreezeStrategy = freeze_strategy

        self.backbone = DINOv3ViTModel.from_pretrained(hf_model_name)
        _adapt_patch_embedding(self.backbone, in_channels)

        if freeze_strategy == "full":
            self._freeze_backbone()

        hidden_size = self.backbone.config.hidden_size
        self.num_register_tokens: int = getattr(
            self.backbone.config, "num_register_tokens", 0
        )

        self.head = nn.Conv2d(hidden_size, num_classes, kernel_size=1)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _freeze_backbone(self) -> None:
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.backbone.eval()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train(self, mode: bool = True):
        """Keep backbone in eval mode when freeze_strategy='full'."""
        super().train(mode)
        if self.freeze_strategy == "full":
            self.backbone.eval()
        return self

    def parameter_groups(self, base_lr: float, backbone_lr_scale: float = 0.1) -> list[dict]:
        """Return AdamW parameter groups with differential learning rates.

        The head receives ``base_lr``; the backbone receives
        ``base_lr * backbone_lr_scale``. Use this instead of
        ``model.parameters()`` when freeze_strategy='none' so the pretrained
        weights are not overwritten by a too-large LR.

        Args:
            base_lr: Learning rate for the segmentation head.
            backbone_lr_scale: Multiplier applied to base_lr for the backbone.
                Default 0.1 (backbone LR = head LR / 10).
        """
        return [
            {"params": self.head.parameters(), "lr": base_lr},
            {"params": self.backbone.parameters(), "lr": base_lr * backbone_lr_scale},
        ]

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pixel_values: ``(B, C, H, W)`` float tensor with z-score normalised
                SAR backscatter values (VV, VH). No additional preprocessing needed.

        Returns:
            Logits ``(B, num_classes, H, W)``.
        """
        H, W = pixel_values.shape[-2:]

        outputs = self.backbone(pixel_values=pixel_values)

        # last_hidden_state: (B, 1 + num_register_tokens + N_patches, hidden)
        last_hidden = outputs.last_hidden_state
        skip = 1 + self.num_register_tokens
        patch_tokens = last_hidden[:, skip:, :]  # (B, N, hidden)

        # Reshape to spatial grid
        pH = H // self.patch_size
        pW = W // self.patch_size
        patch_tokens = patch_tokens.permute(0, 2, 1).reshape(
            -1, patch_tokens.shape[-1], pH, pW
        )  # (B, hidden, pH, pW)

        logits_small = self.head(patch_tokens)  # (B, num_classes, pH, pW)
        logits = F.interpolate(
            logits_small, size=(H, W), mode="bilinear", align_corners=False
        )
        return logits

    @property
    def trainable_parameters(self) -> list[nn.Parameter]:
        return [p for p in self.parameters() if p.requires_grad]
