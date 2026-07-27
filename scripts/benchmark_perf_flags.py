"""Benchmark trainer.perf flag combinations on the real training pipeline.

Runs ONE flag-cell per process (tf32/cudnn/compile are process-global, so cells
must not share a process). Times steady-state optimizer steps on the real
memmap dataloader at the production batch size, mirroring SpectralTrainer's
step: bf16 autocast forward + composite loss, backward, grad-clip 1.0, AdamW
step. Emits one JSON line with the results.

Usage (from repo root, PYTHONPATH=.):
  python scripts/benchmark_perf_flags.py --encoder tu-swin_base_patch4_window7_224 \
      --cell base --memmap-root /mnt/local_ssd/dataset

Mixed-forward ConvNeXtV2-Small compiled benchmark (runtime only; Small has no
pretrained timm weights):
  python scripts/benchmark_perf_flags.py --encoder tu-convnextv2_small \
      --encoder-weights none --cell all \
      --memmap-root /mnt/local_ssd/dataset_mixed80 --dtype float16 \
      --warmup 25 --steps 150
"""

import argparse
import glob
import json
import os
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


def optional_weights(value: str):
    return None if value.lower() in {"none", "null"} else value


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", required=True)
    ap.add_argument("--arch", default="upernet")
    ap.add_argument("--cell", required=True, choices=sorted(CELLS))
    ap.add_argument("--memmap-root", required=True)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument(
        "--dtype",
        default="float32",
        choices=("float16", "float32"),
        help="On-disk memmap dtype (mixed80_blocked uses float16).",
    )
    ap.add_argument("--warmup", type=int, default=25)
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument(
        "--encoder-weights",
        default="imagenet",
        type=optional_weights,
        help='Pretrained weight key, or "none"/"null" for random initialization.',
    )
    # Concurrency measurement: all N processes finish warmup, rendezvous at the
    # barrier, and only then start their timed windows; each finisher keeps
    # running untimed "linger" steps until every process has finished timing,
    # so no timed step runs on a quieter GPU than intended.
    ap.add_argument("--barrier-file", default=None)
    ap.add_argument("--barrier-count", type=int, default=1)
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
        dtype=args.dtype,
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

    it = iter(dl)

    def one_step(timed: bool):
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

        if timed:
            data_ms.append((t1 - t0) * 1e3)
            compute_ms.append((t2 - t1) * 1e3)
            total_ms.append((t2 - t0) * 1e3)

    def wait_for(pattern, n):
        while len(glob.glob(pattern)) < n:
            time.sleep(0.5)

    for _ in range(warmup):
        one_step(timed=False)

    if args.barrier_file:
        open(f"{args.barrier_file}.ready.{os.getpid()}", "w").close()
        wait_for(f"{args.barrier_file}.ready.*", args.barrier_count)

    torch.cuda.reset_peak_memory_stats()
    for _ in range(args.steps):
        one_step(timed=True)

    if args.barrier_file:
        open(f"{args.barrier_file}.done.{os.getpid()}", "w").close()
        # keep the GPU loaded until every process has finished its timed window
        while len(glob.glob(f"{args.barrier_file}.done.*")) < args.barrier_count:
            one_step(timed=False)

    result = {
        "arch": args.arch,
        "encoder": args.encoder,
        "cell": args.cell,
        "concurrency": args.barrier_count,
        "mps": bool(os.environ.get("CUDA_MPS_PIPE_DIRECTORY")),
        **{k: v for k, v in perf_cfg.items() if k != "compile_mode"},
        "batch_size": args.batch_size,
        "memmap_dtype": args.dtype,
        "encoder_weights": args.encoder_weights,
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
