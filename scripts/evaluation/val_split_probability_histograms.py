"""Phase 1 of the threshold-calibration plans: val-split margin histograms.

For every (architecture, seed) and every per-architecture 3-seed ensemble, run a
forward pass over the training validation split (val.memmap) and accumulate two
fine histograms of the softmax margin m = z_water - z_land (for 2-class softmax,
p_water = sigmoid(m)): one over GT-water pixels, one over GT-land pixels.
Pixels labeled 255 are excluded and counted.

These histograms are the sufficient statistic for both calibration plans
(docs/threshold_calibration/):
  - Option 1: exact IoU/F1 at every threshold via cumulative sums
    (threshold tau on p corresponds to the margin cut logit(tau));
  - Option 2: Platt scaling fitted on binned margins with class counts.

The ensemble histogram bins the margin of the MEAN seed probability,
m_ens = logit(mean_p), matching how pooled Hampyeong rasters are built
(mean of the 3 seed probability rasters).

Checkpoints: the as-shipped best.pth per (model, seed) — run
scripts/evaluation/ensure_best_ckpts.py first. Checkpoint provenance (resolved
path, step, stored val_miou) is recorded in every meta.json.

Outputs, per (model, seed) and per model ensemble:
    {out_root}/{model}_s{seed}/hist_margin.npz + meta.json
    {out_root}/{model}_ensemble/hist_margin.npz + meta.json

Run (GPU VM):
    python scripts/evaluation/val_split_probability_histograms.py \
        --stage2-root outputs/stage2 \
        --memmap-root /mnt/local_ssd/dataset \
        --out-root outputs/evaluation/val_split_calibration

Smoke test: add --max-batches 3 (writes to a separate out-root or use
--overwrite afterwards; partial histograms are marked "partial" in meta.json).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

_REPO = Path(__file__).resolve().parent.parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from src.data.datamodule import CoastalDataModule  # noqa: E402
from src.models.factory import SegmentationModelFactory  # noqa: E402

# Histogram design (fixed ex ante in both plans): 4096 bins over m in [-20, 20],
# overflow clamped into the end bins. m = 0 (p = 0.5) falls exactly on the edge
# between bins 2047 and 2048, so the deployment operating point is exactly
# representable: p >= 0.5 <=> bins 2048..4095.
N_BINS = 4096
MARGIN_LO, MARGIN_HI = -20.0, 20.0
BIN_WIDTH = (MARGIN_HI - MARGIN_LO) / N_BINS
ENSEMBLE_EPS = 1e-7  # clamp for logit(mean_p); |logit(eps)| ~ 16.1 < 20

# model dir name under stage2 -> (smp arch, encoder). The 9 multi-seed
# architectures of record (Hampyeong 8 + ConvNeXtV2-Large).
MODEL_SPECS = OrderedDict([
    ("upernet_tu-swin_base_patch4_window7_224", ("upernet", "tu-swin_base_patch4_window7_224")),
    ("upernet_tu-swin_large_patch4_window7_224", ("upernet", "tu-swin_large_patch4_window7_224")),
    ("upernet_tu-convnextv2_base", ("upernet", "tu-convnextv2_base")),
    ("upernet_tu-convnextv2_large", ("upernet", "tu-convnextv2_large")),
    ("deeplabv3plus_resnet50", ("deeplabv3plus", "resnet50")),
    ("dpt_tu-vit_base_patch16_224.mae", ("dpt", "tu-vit_base_patch16_224.mae")),
    ("segformer_mit_b4", ("segformer", "mit_b4")),
    ("unet_resnet50", ("unet", "resnet50")),
    ("unetplusplus_resnet50", ("unetplusplus", "resnet50")),
])


class MarginHistogram:
    """GT-water / GT-land histograms of the margin, accumulated on device."""

    def __init__(self, device: torch.device):
        self.water = torch.zeros(N_BINS, dtype=torch.int64, device=device)
        self.land = torch.zeros(N_BINS, dtype=torch.int64, device=device)

    def update(self, m: torch.Tensor, water_mask: torch.Tensor, land_mask: torch.Tensor):
        idx = ((m - MARGIN_LO) * (1.0 / BIN_WIDTH)).long().clamp_(0, N_BINS - 1)
        self.water += torch.bincount(idx[water_mask], minlength=N_BINS)
        self.land += torch.bincount(idx[land_mask], minlength=N_BINS)

    def numpy(self) -> tuple[np.ndarray, np.ndarray]:
        return self.water.cpu().numpy(), self.land.cpu().numpy()


def hist_metrics_at_half(hist_water: np.ndarray, hist_land: np.ndarray) -> dict:
    """Confusion at p >= 0.5 (margin >= 0, bins 2048+) -> water IoU, mIoU."""
    cut = N_BINS // 2
    tp = int(hist_water[cut:].sum())
    fn = int(hist_water[:cut].sum())
    fp = int(hist_land[cut:].sum())
    tn = int(hist_land[:cut].sum())
    iou_water = tp / max(tp + fp + fn, 1)
    iou_land = tn / max(tn + fp + fn, 1)
    return {
        "iou_water@0.5": iou_water,
        "iou_land@0.5": iou_land,
        "miou@0.5": (iou_water + iou_land) / 2.0,
        "TP": tp, "FP": fp, "FN": fn, "TN": tn,
    }


def load_checkpoint_meta(best_path: Path, device: torch.device) -> tuple[dict, dict]:
    """Load a best.pth; return (state_dict, provenance dict)."""
    if not (best_path.exists() or best_path.is_symlink()):
        raise SystemExit(
            f"Missing {best_path} — run scripts/evaluation/ensure_best_ckpts.py first")
    resolved = best_path.resolve()
    ck = torch.load(resolved, map_location=device, weights_only=True)
    prov = {
        "checkpoint_path": str(best_path),
        "checkpoint_resolved": str(resolved),
        "checkpoint_step": int(ck.get("step", -1)),
        "checkpoint_val_miou_stored": float(ck.get("val_miou", float("nan"))),
    }
    return ck["model_state_dict"], prov


def save_outputs(out_dir: Path, hist: MarginHistogram, n_ignored: int,
                 n_pixels_total: int, meta: dict) -> dict:
    hist_water, hist_land = hist.numpy()
    assert int(hist_water.sum()) + int(hist_land.sum()) + n_ignored == n_pixels_total, (
        f"{out_dir.name}: histogram counts {hist_water.sum()} + {hist_land.sum()} "
        f"+ ignored {n_ignored} != total {n_pixels_total}")
    out_dir.mkdir(parents=True, exist_ok=True)
    bin_edges = MARGIN_LO + BIN_WIDTH * np.arange(N_BINS + 1, dtype=np.float64)
    np.savez_compressed(
        out_dir / "hist_margin.npz",
        hist_water=hist_water, hist_land=hist_land, bin_edges=bin_edges,
        n_ignored=np.int64(n_ignored), n_pixels_total=np.int64(n_pixels_total),
    )
    m = hist_metrics_at_half(hist_water, hist_land)
    meta = dict(meta)
    meta.update({
        "n_pixels_total": n_pixels_total,
        "n_ignored": n_ignored,
        "n_water": int(hist_water.sum()),
        "n_land": int(hist_land.sum()),
        "hist_bins": N_BINS,
        "margin_range": [MARGIN_LO, MARGIN_HI],
        **{k: m[k] for k in ("iou_water@0.5", "iou_land@0.5", "miou@0.5",
                             "TP", "FP", "FN", "TN")},
    })
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    return m


def git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=_REPO,
                              capture_output=True, text=True).stdout.strip()
    except Exception:
        return "unknown"


def outputs_exist(out_root: Path, model: str, seeds: list[str], with_ensemble: bool) -> bool:
    dirs = [out_root / f"{model}_{s}" for s in seeds]
    if with_ensemble:
        dirs.append(out_root / f"{model}_ensemble")
    return all((d / "hist_margin.npz").exists() and (d / "meta.json").exists()
               for d in dirs)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage2-root", type=Path, default=Path("outputs/stage2"))
    ap.add_argument("--memmap-root", type=str, required=True)
    ap.add_argument("--val-file", default="val.memmap")
    ap.add_argument("--out-root", type=Path,
                    default=Path("outputs/evaluation/val_split_calibration"))
    ap.add_argument("--models", nargs="+", default=list(MODEL_SPECS.keys()),
                    choices=list(MODEL_SPECS.keys()))
    ap.add_argument("--seeds", nargs="+", default=["s19", "s42", "s58"])
    ap.add_argument("--no-ensemble", action="store_true",
                    help="Skip the 3-seed ensemble histogram")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--precision", choices=["fp32", "bf16", "fp16"], default="fp32",
                    help="fp32 = no autocast (default; calibration-grade margins). "
                         "bf16/fp16 = autocast forward, margins cast to fp32.")
    ap.add_argument("--in-channels", type=int, default=2)
    ap.add_argument("--num-classes", type=int, default=2)
    ap.add_argument("--max-batches", type=int, default=None,
                    help="Smoke-test cap; outputs are marked partial in meta.json")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_autocast = args.precision != "fp32"
    amp_dtype = torch.bfloat16 if args.precision == "bf16" else torch.float16
    with_ensemble = (not args.no_ensemble) and len(args.seeds) > 1

    dm = CoastalDataModule(
        root_dir=args.memmap_root,
        val_file=args.val_file,
        H=224, W=224,
        batch_size=args.batch_size,
        val_batch_size=args.batch_size,
        num_workers=args.num_workers,
        augment=False,
    )
    dm.setup()
    if dm.val_ds is None:
        raise SystemExit(f"No {args.val_file} under {args.memmap_root}")
    n_chips = len(dm.val_ds)
    print(f"Val split: {n_chips} chips of 224x224 "
          f"({n_chips * 224 * 224 / 1e9:.2f}e9 pixels), device={device}, "
          f"precision={args.precision}")

    # One loader reused across all architectures (persistent workers would
    # otherwise accumulate, one pool per pass).
    loader = dm.val_dataloader()

    base_meta = {
        "val_file": str(Path(args.memmap_root) / args.val_file),
        "n_chips": n_chips,
        "batch_size": args.batch_size,
        "precision": args.precision,
        "torch_version": torch.__version__,
        "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
        "matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "git_commit": git_commit(),
        "partial": args.max_batches is not None,
        "max_batches": args.max_batches,
    }

    for model_key in args.models:
        arch, encoder = MODEL_SPECS[model_key]
        if not args.overwrite and outputs_exist(args.out_root, model_key,
                                                args.seeds, with_ensemble):
            print(f"\n=== {model_key}: outputs exist, skipping (use --overwrite) ===")
            continue

        print(f"\n=== {model_key} (arch={arch}, encoder={encoder}) ===")
        models, provs = [], []
        for seed in args.seeds:
            best = args.stage2_root / model_key / seed / "best.pth"
            state, prov = load_checkpoint_meta(best, device)
            # encoder_weights=None: every parameter comes from the checkpoint.
            net = SegmentationModelFactory.build(
                arch=arch, encoder_name=encoder, encoder_weights=None,
                in_channels=args.in_channels, classes=args.num_classes)
            net.load_state_dict(state)
            net.to(device).eval()
            models.append(net)
            provs.append(prov)
            print(f"  {seed}: {prov['checkpoint_resolved']} "
                  f"(step {prov['checkpoint_step']}, "
                  f"stored val mIoU {prov['checkpoint_val_miou_stored']:.6f})")
        del state

        seed_hists = [MarginHistogram(device) for _ in args.seeds]
        ens_hist = MarginHistogram(device) if with_ensemble else None
        n_ignored = 0
        n_pixels_total = 0
        n_bad_labels = 0
        t0 = time.time()

        with torch.inference_mode():
            for bi, batch in enumerate(tqdm(loader, desc=model_key, leave=False)):
                if args.max_batches is not None and bi >= args.max_batches:
                    break
                x = batch["pixel_values"].to(device, non_blocking=True)
                y = batch["labels"].to(device, non_blocking=True)

                valid = y != 255
                water_mask = valid & (y == 1)
                land_mask = valid & (y == 0)
                n_pixels_total += y.numel()
                n_ignored += int((~valid).sum())
                n_bad_labels += int((valid & (y != 0) & (y != 1)).sum())

                p_sum = None
                for net, hist in zip(models, seed_hists):
                    if use_autocast:
                        with torch.autocast(device_type=device.type, dtype=amp_dtype):
                            logits = net(x)
                    else:
                        logits = net(x)
                    logits = logits.float()
                    m = logits[:, 1] - logits[:, 0]  # p_water = sigmoid(m)
                    hist.update(m, water_mask, land_mask)
                    if ens_hist is not None:
                        p = torch.sigmoid(m)
                        p_sum = p if p_sum is None else p_sum + p
                if ens_hist is not None:
                    p_mean = (p_sum / len(models)).clamp_(ENSEMBLE_EPS, 1.0 - ENSEMBLE_EPS)
                    m_ens = torch.log(p_mean) - torch.log1p(-p_mean)
                    ens_hist.update(m_ens, water_mask, land_mask)

        elapsed = time.time() - t0
        if n_bad_labels:
            raise SystemExit(f"{model_key}: {n_bad_labels} labels outside {{0,1,255}}")
        print(f"  pass done in {elapsed/60:.1f} min "
              f"({n_pixels_total / max(elapsed, 1e-9) / 1e6:.0f} Mpx/s)")

        for seed, hist, prov in zip(args.seeds, seed_hists, provs):
            meta = {**base_meta, "model": model_key, "arch": arch,
                    "encoder": encoder, "seed": seed, "kind": "seed", **prov,
                    "elapsed_s": elapsed}
            m = save_outputs(args.out_root / f"{model_key}_{seed}", hist,
                             n_ignored, n_pixels_total, meta)
            stored = prov["checkpoint_val_miou_stored"]
            diff = m["miou@0.5"] - stored
            flag = "" if abs(diff) < 0.005 else "  <-- LARGE, investigate"
            print(f"  {seed}: hist mIoU@0.5 {m['miou@0.5']:.6f} vs stored "
                  f"val mIoU {stored:.6f} (diff {diff:+.6f}){flag}")

        if ens_hist is not None:
            meta = {**base_meta, "model": model_key, "arch": arch,
                    "encoder": encoder, "seed": "ensemble", "kind": "ensemble",
                    "member_checkpoints": provs, "ensemble_eps": ENSEMBLE_EPS,
                    "elapsed_s": elapsed}
            m = save_outputs(args.out_root / f"{model_key}_ensemble", ens_hist,
                             n_ignored, n_pixels_total, meta)
            print(f"  ensemble: hist mIoU@0.5 {m['miou@0.5']:.6f} "
                  f"(water IoU {m['iou_water@0.5']:.6f})")

        for net in models:
            del net
        models.clear()
        torch.cuda.empty_cache()

    dm.teardown()
    print("\nAll requested models done.")


if __name__ == "__main__":
    main()
