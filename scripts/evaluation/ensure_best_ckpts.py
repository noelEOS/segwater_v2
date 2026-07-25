"""Ensure every stage2 (model, seed) dir has a best.pth, using the as-shipped rule.

The as-shipped best.pth (the checkpoint every downstream eval and inference run
loads) is selected by the rule in the original make_best_ckpt_path.py: highest
4-dp mIoU parsed from the FILENAME, tie-break on highest step, final fallback on
name. This script applies that same rule wherever best.pth is missing, and
audits existing best.pth files against it.

It also loads each candidate's stored full-float val_miou and reports where the
filename rule disagrees with the trainer's full-float argmax (known case:
Swin-Base s42). Disagreements are REPORTED, not resolved: reproducing the
as-shipped selection is the point.

Usage:
    python scripts/evaluation/ensure_best_ckpts.py \
        --stage2-root outputs/stage2 \
        --models upernet_tu-swin_base_patch4_window7_224 ... \
        --seeds s19 s42 s58 \
        [--report-out outputs/evaluation/val_split_calibration/best_ckpt_audit.json]
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch

STEP_RE = re.compile(r"step(\d+)")
MIOU_RE = re.compile(r"miou([0-9]*\.?[0-9]+)")
# SWA checkpoints (build_swa_checkpoint.py: `_swa{K}_step{min}-{max}.pth`) are a
# derived, competing checkpoint-selection ARM, not a candidate for best.pth --
# keeping them out of the pool is what makes best-vs-SWA an independent contrast.
# They also carry val_miou=None, which the float cross-check below must survive.
SWA_RE = re.compile(r"_swa\d+_step")

DEFAULT_MODELS = [
    "upernet_tu-swin_base_patch4_window7_224",
    "upernet_tu-swin_large_patch4_window7_224",
    "upernet_tu-convnextv2_base",
    "upernet_tu-convnextv2_large",
    "deeplabv3plus_resnet50",
    "dpt_tu-vit_base_patch16_224.mae",
    "segformer_mit_b4",
    "unet_resnet50",
    "unetplusplus_resnet50",
]


def pick_best_filename_rule(pths: list[Path]) -> Path:
    """Highest 4-dp filename mIoU; tie-break on highest step; fallback on name."""
    def key(p: Path):
        miou_m = MIOU_RE.search(p.name)
        step_m = STEP_RE.search(p.name)
        miou = float(miou_m.group(1)) if miou_m else -1.0
        step = int(step_m.group(1)) if step_m else -1
        return (miou, step, p.name)
    return max(pths, key=key)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage2-root", type=Path, default=Path("outputs/stage2"))
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    ap.add_argument("--seeds", nargs="+", default=["s19", "s42", "s58"])
    ap.add_argument("--report-out", type=Path, default=None)
    ap.add_argument("--dry-run", action="store_true",
                    help="Report only; do not create any best.pth")
    args = ap.parse_args()

    report = []
    n_created = n_present = n_disagree = 0

    for model in args.models:
        for seed in args.seeds:
            seed_dir = args.stage2_root / model / seed
            if not seed_dir.is_dir():
                raise SystemExit(f"Missing seed dir: {seed_dir}")
            step_ckpts = [p for p in sorted(seed_dir.glob("*.pth"))
                          if p.name != "best.pth"
                          and "_snap_" not in p.name
                          and not SWA_RE.search(p.name)
                          and not p.name.endswith("_last.pth")]
            if not step_ckpts:
                raise SystemExit(f"No step checkpoints in {seed_dir}")

            filename_pick = pick_best_filename_rule(step_ckpts)

            # Full-float cross-check: what the trainer's argmax (exact val_miou,
            # ties to later step) would have picked.
            floats = []
            for p in step_ckpts:
                ck = torch.load(p, map_location="cpu", weights_only=True)
                # val_miou may be absent or explicitly None (model-only/derived
                # checkpoints); treat both as "no score" rather than crashing.
                raw_miou, raw_step = ck.get("val_miou"), ck.get("step")
                floats.append((float(raw_miou) if raw_miou is not None else float("-inf"),
                               int(raw_step) if raw_step is not None else -1, p))
            float_pick = sorted(floats, key=lambda t: (t[0], t[1]))[-1][2]

            best = seed_dir / "best.pth"
            entry = {
                "model": model, "seed": seed,
                "filename_rule_pick": filename_pick.name,
                "float_argmax_pick": float_pick.name,
                "rules_agree": filename_pick == float_pick,
                "candidates": [
                    {"name": p.name, "val_miou_exact": v, "step": s}
                    for v, s, p in sorted(floats, key=lambda t: t[1])
                ],
            }
            if filename_pick != float_pick:
                n_disagree += 1
                print(f"NOTE {model}/{seed}: filename rule -> {filename_pick.name} "
                      f"but float argmax -> {float_pick.name} (as-shipped = filename rule)")

            if best.exists() or best.is_symlink():
                target = best.resolve() if best.is_symlink() else None
                if target is not None:
                    matches = target.name == filename_pick.name
                else:
                    # regular-file copy: identify by stored step
                    ck = torch.load(best, map_location="cpu", weights_only=True)
                    best_step = int(ck.get("step", -1))
                    pick_step = int(STEP_RE.search(filename_pick.name).group(1))
                    matches = best_step == pick_step
                entry["best_pth"] = "pre-existing"
                entry["best_matches_filename_rule"] = matches
                status = "matches filename rule" if matches else "!! DIFFERS from filename rule"
                print(f"{model}/{seed}: best.pth pre-existing ({status})")
                n_present += 1
            else:
                entry["best_pth"] = "created" if not args.dry_run else "would create"
                entry["best_matches_filename_rule"] = True
                if not args.dry_run:
                    best.symlink_to(filename_pick.name)  # relative, stays portable
                print(f"{model}/{seed}: best.pth -> {filename_pick.name} "
                      f"({'created' if not args.dry_run else 'dry-run'})")
                n_created += 1
            report.append(entry)

    print(f"\n{n_present} pre-existing, {n_created} created, "
          f"{n_disagree} filename-vs-float disagreements")
    if args.report_out is not None:
        args.report_out.parent.mkdir(parents=True, exist_ok=True)
        args.report_out.write_text(json.dumps(report, indent=2))
        print(f"Wrote {args.report_out}")


if __name__ == "__main__":
    main()
