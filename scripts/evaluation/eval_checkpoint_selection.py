"""Score every retained top-k checkpoint on the held-out test memmap.

Training keeps the top-3 checkpoints by val mIoU and promotes the argmax as
`best.pth` (src/engine/trainer.py). This script asks whether that val-argmax is
also the test-argmax, i.e. whether selecting on val generalises or is just
fitting val noise.

Usage:
    python scripts/evaluation/eval_checkpoint_selection.py \
        --ckpt-root outputs/stage2/upernet_tu-swin_base_patch4_window7_224 \
        --memmap-root ~/dataset \
        --arch upernet --encoder tu-swin_base_patch4_window7_224 \
        --seeds s19 s42 s58 \
        --out results.json
"""
import argparse
import json
import os
import re
from pathlib import Path

import torch

from src.data.datamodule import CoastalDataModule
from src.engine.trainer import SpectralTrainer
from src.models.factory import SegmentationModelFactory
from src.models.losses import CoastalCompositeLoss

CKPT_RE = re.compile(r"_step(\d+)_miou([0-9.]+)\.pth$")


def discover(ckpt_root: Path, seeds):
    """Map seed -> [(step, name_miou, path)], excluding the best.pth alias."""
    found = {}
    for seed in seeds:
        seed_dir = ckpt_root / seed
        ckpts = []
        for p in sorted(seed_dir.glob("*.pth")):
            m = CKPT_RE.search(p.name)
            if m is None:
                continue  # skips best.pth, which duplicates one of the three
            ckpts.append((int(m.group(1)), float(m.group(2)), p))
        if not ckpts:
            raise SystemExit(f"No step/miou checkpoints under {seed_dir}")
        found[seed] = sorted(ckpts)
    return found


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-root", type=Path, required=True)
    ap.add_argument("--memmap-root", type=str, required=True)
    ap.add_argument("--arch", required=True)
    ap.add_argument("--encoder", required=True)
    ap.add_argument("--seeds", nargs="+", default=["s19", "s42", "s58"])
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--in-channels", type=int, default=2)
    ap.add_argument("--num-classes", type=int, default=2)
    ap.add_argument("--precision", default="bf16")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_root = args.ckpt_root.expanduser()
    memmap_root = os.path.expanduser(args.memmap_root)

    ckpts = discover(ckpt_root, args.seeds)
    total = sum(len(v) for v in ckpts.values())
    print(f"Found {total} checkpoints across {len(ckpts)} seeds under {ckpt_root}")

    # augment=False: test is evaluated clean, exactly as trainer.test() does.
    dm = CoastalDataModule(
        root_dir=memmap_root,
        H=224, W=224,
        batch_size=args.batch_size,
        val_batch_size=args.batch_size,
        num_workers=args.num_workers,
        augment=False,
    )
    dm.setup()
    if dm.test_ds is None:
        raise SystemExit(f"No test.memmap under {memmap_root}")
    test_dl = dm.test_dataloader()
    print(f"Test samples: {len(dm.test_ds)}  batches: {len(test_dl)}")

    # Built once and re-used; each checkpoint overwrites the weights in place.
    # encoder_weights=None: every parameter comes from the checkpoint, so there
    # is no point paying for the pretrained download.
    model = SegmentationModelFactory.build(
        arch=args.arch,
        encoder_name=args.encoder,
        encoder_weights=None,
        in_channels=args.in_channels,
        classes=args.num_classes,
    )
    loss_fn = CoastalCompositeLoss(ce_weight=0.5, dice_weight=0.5, label_smoothing=0.05)

    trainer = SpectralTrainer(
        model=model,
        optimizer=None,
        scheduler=None,
        loss_fn=loss_fn,
        device=device,
        use_amp=True,
        precision=args.precision,
        num_classes=args.num_classes,
        arch=args.arch,
        encoder=args.encoder,
    )

    rows = []
    for seed, entries in ckpts.items():
        for step, name_miou, path in entries:
            ckpt = torch.load(path, map_location=device, weights_only=True)
            trainer.model.load_state_dict(ckpt["model_state_dict"])
            # The checkpoint stores the full-precision val mIoU; the filename
            # only carries 4dp. The float is what the selection actually sorted
            # on, so record both.
            val_miou = float(ckpt.get("val_miou", float("nan")))

            print(f"\n[{seed} step{step}] val_miou={val_miou:.6f} -> testing", flush=True)
            m = trainer.test(test_dl)
            print(f"[{seed} step{step}] test mIoU={m['mIoU']:.6f} f1={m['f1_macro']:.6f}", flush=True)

            rows.append({
                "seed": seed,
                "step": step,
                "ckpt": path.name,
                "val_miou_filename": name_miou,
                "val_miou_exact": val_miou,
                "test_miou": m["mIoU"],
                "test_f1_macro": m["f1_macro"],
                "test_loss": m["loss"],
                "TN": m.get("TN"), "FP": m.get("FP"),
                "FN": m.get("FN"), "TP": m.get("TP"),
            })
            args.out.write_text(json.dumps(rows, indent=2))

    dm.teardown()

    # --- verdict: does val-argmax == test-argmax, per seed? ---
    print("\n" + "=" * 78)
    print("VAL-SELECTION vs TEST-TRUTH")
    print("=" * 78)
    agree = 0
    for seed, entries in ckpts.items():
        srows = [r for r in rows if r["seed"] == seed]
        # Reproduce the trainer's rule: argmax on the exact val float, ties to
        # the later step (stable sort, take last).
        val_pick = sorted(srows, key=lambda r: (r["val_miou_exact"], r["step"]))[-1]
        test_pick = max(srows, key=lambda r: r["test_miou"])
        hit = val_pick["step"] == test_pick["step"]
        agree += hit
        regret = test_pick["test_miou"] - val_pick["test_miou"]
        print(f"\n{seed}:")
        for r in sorted(srows, key=lambda r: r["step"]):
            tag = []
            if r["step"] == val_pick["step"]:
                tag.append("VAL-PICK")
            if r["step"] == test_pick["step"]:
                tag.append("TEST-BEST")
            print(f"  step{r['step']:<6} val={r['val_miou_exact']:.6f}  "
                  f"test={r['test_miou']:.6f}  {' '.join(tag)}")
        print(f"  -> {'AGREE' if hit else 'DISAGREE'};  regret = {regret:.6f} test mIoU")

    print(f"\nval-argmax picked the test-best checkpoint in {agree}/{len(ckpts)} seeds.")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
