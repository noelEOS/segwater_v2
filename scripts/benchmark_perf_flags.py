"""Benchmark trainer.perf flag combinations on the real training pipeline.

Runs ONE flag-cell per process (tf32/cudnn/compile are process-global, so cells
must not share a process). Times steady-state optimizer steps on the real
memmap dataloader at the production batch size, mirroring SpectralTrainer's
step: bf16 autocast forward + composite loss, backward, grad-clip 1.0, AdamW
step. Emits one JSON line with the results.

Usage (from repo root, PYTHONPATH=.):
  python scripts/benchmark_perf_flags.py --encoder tu-swin_base_patch4_window7_224 \
      --cell base --memmap-root /mnt/local_ssd/dataset
"""

import argparse
import json
import statistics
import time

import torch

from src.data.datamodule import CoastalDataModule
from src.models.factory import SegmentationModelFactory
from src.models.losses import CoastalCompositeLoss
from src.utils.perf import apply_perf_flags, build_adamw, maybe_compile

CELLS = {
    # cell name -> (tf32, cudnn_benchmark, fused_adamw, compile)
    "base": (False, False, False, False),
    "fused": (False, False, True, False),
    "nocompile": (True, True, True, False),
    "all": (True, True, True, True),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", required=True)
    ap.add_argument("--arch", default="upernet")
    ap.add_argument("--cell", required=True, choices=sorted(CELLS))
    ap.add_argument("--memmap-root", required=True)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--warmup", type=int, default=25)
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--encoder-weights", default="imagenet")
    args = ap.parse_args()

    tf32, cudnn_bench, fused, compile_ = CELLS[args.cell]
    perf_cfg = {
        "tf32": tf32,
        "cudnn_benchmark": cudnn_bench,
        "fused_adamw": fused,
        "compile": compile_,
        "compile_mode": None,
    }

    device = torch.device("cuda")
    apply_perf_flags(perf_cfg)

    dm = CoastalDataModule(
        root_dir=args.memmap_root,
        batch_size=args.batch_size,
        val_batch_size=args.batch_size,
        num_workers=args.num_workers,
        augment=False,
    )
    dm.setup()
    dl = dm.train_dataloader()

    try:
        model = SegmentationModelFactory.build(
            arch=args.arch,
            encoder_name=args.encoder,
            encoder_weights=args.encoder_weights,
            in_channels=2,
            classes=2,
        )
    except Exception as exc:  # pretrained weights unavailable offline etc.
        print(f"[bench] encoder_weights={args.encoder_weights} failed ({exc}); "
              f"retrying with encoder_weights=None (identical speed)")
        model = SegmentationModelFactory.build(
            arch=args.arch,
            encoder_name=args.encoder,
            encoder_weights=None,
            in_channels=2,
            classes=2,
        )
    model = model.to(device)
    model = maybe_compile(model, perf_cfg)
    model.train()

    loss_fn = CoastalCompositeLoss(ce_weight=0.5, dice_weight=0.5, label_smoothing=0.05)
    optimizer = build_adamw(model.parameters(), lr=3e-4, weight_decay=1e-4, perf_cfg=perf_cfg)

    # Extra warmup for compile (graph capture + autotune) and cudnn autotune.
    warmup = args.warmup + (15 if compile_ else 0)
    data_ms, compute_ms, total_ms = [], [], []
    torch.cuda.reset_peak_memory_stats()

    it = iter(dl)
    for i in range(warmup + args.steps):
        t0 = time.perf_counter()
        batch = next(it)
        t1 = time.perf_counter()

        x = batch["pixel_values"].to(device, non_blocking=True)
        y = batch["labels"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(x)
            loss = loss_fn(logits, y)["loss"]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        torch.cuda.synchronize()
        t2 = time.perf_counter()

        if i >= warmup:
            data_ms.append((t1 - t0) * 1e3)
            compute_ms.append((t2 - t1) * 1e3)
            total_ms.append((t2 - t0) * 1e3)

    result = {
        "encoder": args.encoder,
        "cell": args.cell,
        **{k: v for k, v in perf_cfg.items() if k != "compile_mode"},
        "batch_size": args.batch_size,
        "steps_timed": args.steps,
        "compute_ms_median": round(statistics.median(compute_ms), 1),
        "compute_ms_mean": round(statistics.mean(compute_ms), 1),
        "data_wait_ms_median": round(statistics.median(data_ms), 1),
        "total_ms_median": round(statistics.median(total_ms), 1),
        "samples_per_s": round(args.batch_size / (statistics.median(total_ms) / 1e3), 1),
        "peak_mem_gb": round(torch.cuda.max_memory_allocated() / 1024**3, 2),
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(0),
    }
    print("RESULT " + json.dumps(result), flush=True)

    dm.teardown()


if __name__ == "__main__":
    main()
