"""Export the UNet(resnet50) checkpoint to a TorchScript artifact for Vertex AI serving.

Traces the model on CPU, verifies the traced module matches the eager model at two
different input sizes (proving the trace stayed fully convolutional), and writes the
serialized TorchScript module to disk.

Example:
    python scripts/export_torchscript.py \
        --checkpoint checkpoints/stage2/unet_resnet50/09-44-49_s42/unet_resnet50_s42_step31200_miou0.9382.pth \
        --out /Users/noel/code/segwater-app/inference-service/model/segwater_unet.ts
"""
import argparse
import os

import torch

from src.models.factory import SegmentationModelFactory

DEFAULT_CKPT = (
    "checkpoints/stage2/unet_resnet50/09-44-49_s42/"
    "unet_resnet50_s42_step31200_miou0.9382.pth"
)
DEFAULT_OUT = "/Users/noel/code/segwater-app/inference-service/model/segwater_unet.ts"


def build_model(checkpoint: str) -> torch.nn.Module:
    model = SegmentationModelFactory.build(
        arch="unet",
        encoder_name="resnet50",
        in_channels=2,
        classes=2,
        encoder_weights=None,
    )
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()
    return model


def max_abs_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a - b).abs().max().item()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=DEFAULT_CKPT)
    parser.add_argument("--out", default=DEFAULT_OUT)
    args = parser.parse_args()

    print(f"Loading checkpoint: {args.checkpoint}")
    model = build_model(args.checkpoint)

    # Trace on a (1, 2, 256, 256) float32 CPU example.
    example = torch.randn(1, 2, 256, 256, dtype=torch.float32)
    with torch.no_grad():
        traced = torch.jit.trace(model, example)
    traced.eval()
    print("Traced model with torch.jit.trace on (1, 2, 256, 256).")

    # --- Verification ---
    tol = 1e-5

    # 1) Match at trained/example size on a fresh random input.
    x256 = torch.randn(1, 2, 256, 256, dtype=torch.float32)
    with torch.no_grad():
        eager256 = model(x256)
        ts256 = traced(x256)
    diff256 = max_abs_diff(eager256, ts256)
    print(f"[256x256] output shape={tuple(ts256.shape)} max_abs_diff={diff256:.3e}")
    assert ts256.shape[-2:] == x256.shape[-2:], "256 output spatial shape mismatch"
    assert diff256 <= tol, f"256 max_abs_diff {diff256} exceeds {tol}"

    # 2) Different size -> proves the trace is size-agnostic (fully convolutional).
    x192 = torch.randn(1, 2, 192, 192, dtype=torch.float32)
    with torch.no_grad():
        eager192 = model(x192)
        ts192 = traced(x192)
    diff192 = max_abs_diff(eager192, ts192)
    print(f"[192x192] output shape={tuple(ts192.shape)} max_abs_diff={diff192:.3e}")
    assert ts192.shape[-2:] == x192.shape[-2:], "192 output spatial shape mismatch"
    assert diff192 <= tol, f"192 max_abs_diff {diff192} exceeds {tol}"

    print("Verification passed: traced module matches eager model at both sizes.")

    # --- Write artifact ---
    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    traced.save(args.out)
    size_mb = os.path.getsize(args.out) / (1024 * 1024)
    print(f"Saved TorchScript artifact: {args.out} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
