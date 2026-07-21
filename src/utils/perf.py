"""Opt-in performance knobs shared by the training/HPO/inference entry points.

All numerics-affecting switches default to OFF so a run without explicit
overrides reproduces current behavior bit-for-bit. Flip them per run via
config (`trainer.perf.*` for training/HPO, `inference.compute.*` for
inference). Background and benchmarks: docs/TRAINING_SPEEDUP_NOTES.md and
docs/INFERENCE_SPEEDUP_NOTES.md.
"""

import logging

import torch

logger = logging.getLogger(__name__)


def apply_perf_flags(perf_cfg) -> None:
    """Apply process-wide opt-in flags (TF32 matmul, cudnn.benchmark).

    Accepts a DictConfig/dict with optional boolean keys ``tf32`` and
    ``cudnn_benchmark``; ``None`` or missing keys leave defaults untouched.
    """
    if perf_cfg is None:
        return
    if bool(perf_cfg.get("tf32", False)):
        torch.set_float32_matmul_precision("high")
        logger.info("[PERF] TF32 matmul enabled (float32_matmul_precision='high').")
    if bool(perf_cfg.get("cudnn_benchmark", False)):
        torch.backends.cudnn.benchmark = True
        logger.info("[PERF] cudnn.benchmark autotuning enabled.")


def maybe_compile(model, perf_cfg):
    """Wrap the model with torch.compile when ``perf.compile`` is set.

    ``perf.compile_mode`` (e.g. "reduce-overhead", "max-autotune") is passed
    through; null/missing uses torch.compile's default mode.
    """
    if perf_cfg is None or not bool(perf_cfg.get("compile", False)):
        return model
    mode = perf_cfg.get("compile_mode", None)
    model = torch.compile(model, mode=mode)
    logger.info(f"[PERF] torch.compile enabled (mode={mode or 'default'}).")
    return model


def unwrap_compiled(model):
    """Return the original nn.Module under a torch.compile OptimizedModule.

    Checkpoints must be saved from / loaded into the unwrapped module so their
    state-dict keys stay free of the `_orig_mod.` prefix and remain loadable
    by uncompiled consumers (inference, evaluation).
    """
    return getattr(model, "_orig_mod", model)


def build_adamw(params, lr: float, weight_decay: float):
    """AdamW using the fused CUDA kernel when available, else the default impl.

    fused=True requires the params to already live on a CUDA device at
    construction time; callers must move the model to the device first.
    """
    params = list(params)
    if torch.cuda.is_available() and params and params[0].is_cuda:
        try:
            return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay, fused=True)
        except (RuntimeError, ValueError) as exc:
            logger.warning(f"[PERF] fused AdamW unavailable ({exc}); using default impl.")
    return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
