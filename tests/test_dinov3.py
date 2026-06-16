"""Tests for the DINOv3 linear segmentation wrapper.

These tests run without network access by constructing a DINOv3ViTModel from
a minimal random-weight config instead of downloading a checkpoint. This keeps
CI fast and offline.

Run with:
    conda activate eda_hf
    pytest tests/test_dinov3.py -v
"""

import pytest
import torch
import torch.nn as nn

pytest.importorskip("transformers", reason="transformers>=5.12.0 not installed")

from transformers import DINOv3ViTConfig, DINOv3ViTModel  # noqa: E402

from src.models.dinov3 import DINOv3LinearSegmentation, _adapt_patch_embedding  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _minimal_config(num_register_tokens: int = 4) -> DINOv3ViTConfig:
    return DINOv3ViTConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        patch_size=16,
        image_size=224,
        num_register_tokens=num_register_tokens,
    )


def _make_model(
    num_classes: int = 2,
    in_channels: int = 2,
    num_register_tokens: int = 4,
) -> DINOv3LinearSegmentation:
    """Build a DINOv3LinearSegmentation from random weights (no Hub download)."""
    cfg = _minimal_config(num_register_tokens=num_register_tokens)
    backbone = DINOv3ViTModel(cfg)

    model = DINOv3LinearSegmentation.__new__(DINOv3LinearSegmentation)
    nn.Module.__init__(model)
    model.patch_size = 16
    model.num_register_tokens = num_register_tokens
    model.backbone = backbone
    _adapt_patch_embedding(model.backbone, in_channels)
    model._freeze_backbone()
    model.head = nn.Conv2d(cfg.hidden_size, num_classes, kernel_size=1)
    return model


def _random_input(B: int = 2, C: int = 2, H: int = 224, W: int = 224) -> torch.Tensor:
    # Simulate z-score normalised SAR values (mean≈0, std≈1)
    return torch.randn(B, C, H, W)


# ---------------------------------------------------------------------------
# Tests: channel adaptation
# ---------------------------------------------------------------------------

class TestPatchEmbeddingAdaptation:

    def test_no_op_for_3_channels(self):
        """_adapt_patch_embedding must leave the layer untouched for in_channels=3."""
        cfg = _minimal_config()
        backbone = DINOv3ViTModel(cfg)
        original_proj = backbone.embeddings.patch_embeddings
        _adapt_patch_embedding(backbone, 3)
        assert backbone.embeddings.patch_embeddings is original_proj

    def test_weight_shape_after_adaptation(self):
        """Patch embedding weight must be (hidden, in_channels, 16, 16) after surgery."""
        cfg = _minimal_config()
        backbone = DINOv3ViTModel(cfg)
        _adapt_patch_embedding(backbone, 2)
        w = backbone.embeddings.patch_embeddings.weight
        assert w.shape == (64, 2, 16, 16), f"Unexpected shape: {w.shape}"

    def test_adapted_weights_are_mean_of_rgb(self):
        """Adapted weights must equal the per-pixel mean of the original RGB weights."""
        cfg = _minimal_config()
        backbone = DINOv3ViTModel(cfg)
        original_weight = backbone.embeddings.patch_embeddings.weight.data.clone()
        _adapt_patch_embedding(backbone, 2)
        adapted = backbone.embeddings.patch_embeddings.weight.data
        expected = original_weight.mean(dim=1, keepdim=True).repeat(1, 2, 1, 1)
        assert torch.allclose(adapted, expected)

    def test_bias_preserved_after_adaptation(self):
        """Bias tensor must be unchanged after channel surgery."""
        cfg = _minimal_config()
        backbone = DINOv3ViTModel(cfg)
        original_bias = backbone.embeddings.patch_embeddings.bias.data.clone()
        _adapt_patch_embedding(backbone, 2)
        adapted_bias = backbone.embeddings.patch_embeddings.bias.data
        assert torch.allclose(adapted_bias, original_bias)


# ---------------------------------------------------------------------------
# Tests: forward pass
# ---------------------------------------------------------------------------

class TestDINOv3LinearSegmentation:

    def test_output_shape_2ch_input(self):
        """Logits must be (B, num_classes, H, W) for 2-channel SAR input."""
        model = _make_model(num_classes=2, in_channels=2)
        x = _random_input(B=2, C=2)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (2, 2, 224, 224), f"Unexpected shape: {out.shape}"

    def test_output_shape_3_classes(self):
        model = _make_model(num_classes=3, in_channels=2)
        x = _random_input(B=1, C=2)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1, 3, 224, 224)

    def test_no_register_tokens(self):
        model = _make_model(num_classes=2, in_channels=2, num_register_tokens=0)
        x = _random_input(B=1, C=2)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1, 2, 224, 224)

    # ---------------------------------------------------------------------------
    # Tests: frozen backbone / trainable head
    # ---------------------------------------------------------------------------

    def test_backbone_is_frozen(self):
        """No backbone parameter should require gradients."""
        model = _make_model()
        for name, param in model.backbone.named_parameters():
            assert not param.requires_grad, f"Backbone param {name!r} is not frozen"

    def test_head_is_trainable(self):
        model = _make_model()
        for name, param in model.head.named_parameters():
            assert param.requires_grad, f"Head param {name!r} is not trainable"

    def test_trainable_parameters_only_head(self):
        model = _make_model()
        trainable = model.trainable_parameters
        head_params = list(model.head.parameters())
        assert len(trainable) == len(head_params)
        for t, h in zip(trainable, head_params):
            assert t is h

    def test_backbone_stays_in_eval_during_train_mode(self):
        model = _make_model()
        model.train()
        assert not model.backbone.training, "Backbone should stay in eval mode"

    def test_no_gradient_flows_to_backbone(self):
        """Backward pass must not accumulate gradients on backbone weights."""
        model = _make_model(num_classes=2, in_channels=2)
        model.train()
        x = _random_input(B=1, C=2)
        out = model(x)
        out.sum().backward()
        for name, param in model.backbone.named_parameters():
            assert param.grad is None, f"Backbone param {name!r} has gradients"

    # ---------------------------------------------------------------------------
    # Tests: factory integration
    # ---------------------------------------------------------------------------

    def test_factory_returns_dinov3_model(self):
        """SegmentationModelFactory with arch='dinov3-linear' must return our wrapper."""
        from src.models.factory import SegmentationModelFactory

        cfg = _minimal_config()
        backbone = DINOv3ViTModel(cfg)

        original = DINOv3ViTModel.from_pretrained
        DINOv3ViTModel.from_pretrained = classmethod(lambda cls, *a, **kw: backbone)
        try:
            model = SegmentationModelFactory.build(
                arch="dinov3-linear",
                encoder_name="facebook/dinov3-vits16-pretrain-lvd1689m",
                in_channels=2,
                classes=2,
            )
        finally:
            DINOv3ViTModel.from_pretrained = original

        assert isinstance(model, DINOv3LinearSegmentation)

    def test_factory_passes_in_channels(self):
        """Factory must wire in_channels through so patch embedding is adapted."""
        from src.models.factory import SegmentationModelFactory

        cfg = _minimal_config()
        backbone = DINOv3ViTModel(cfg)

        original = DINOv3ViTModel.from_pretrained
        DINOv3ViTModel.from_pretrained = classmethod(lambda cls, *a, **kw: backbone)
        try:
            model = SegmentationModelFactory.build(
                arch="dinov3-linear",
                encoder_name="facebook/dinov3-vits16-pretrain-lvd1689m",
                in_channels=2,
                classes=2,
            )
        finally:
            DINOv3ViTModel.from_pretrained = original

        proj = model.backbone.embeddings.patch_embeddings
        assert proj.weight.shape[1] == 2, (
            f"Patch embedding should have 2 input channels, got {proj.weight.shape[1]}"
        )
