"""Recreate best.pth symlinks under a run tree.

Object stores (R2/S3 via rclone) have no symlink concept: syncing a run dir up
dereferences best.pth into a full second copy of the checkpoint, and syncing
back down leaves a regular file where the symlink was. This script restores the
symlink, so a downloaded tree matches what the trainer wrote.

Selection rule: the trainer's own rule (src/engine/trainer.py) -- the argmax of
the full-float `val_miou` stored INSIDE each checkpoint, ties broken by the
later step. Filenames are never parsed for the mIoU, so this is unaffected by
the 4-dp -> 6-dp filename change. Use scripts/evaluation/ensure_best_ckpts.py
instead when you need the legacy as-shipped filename rule reproduced.

Every directory containing at least one *.pth (other than best.pth) is treated
as a checkpoint dir, so this works for any run layout.

Usage:
    python scripts/relink_best_ckpts.py --root outputs/stage2 --dry-run
    python scripts/relink_best_ckpts.py --root outputs/stage2
    python scripts/relink_best_ckpts.py --root outputs/stage2 --replace-files
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

# Shared checkpoint-selection rules (one implementation for the whole repo).
# `step_of_checkpoint` is the payload-first rule this script has always used --
# deliberately different from ckptsel.step_of (filename-only); see its docstring.
sys.path.insert(0, str(Path(__file__).resolve().parent / "evaluation" / "vm"))
import ckptsel  # noqa: E402
from ckptsel import step_of_checkpoint  # noqa: E402


def pick_best(candidates: list[Path]) -> tuple[Path, list[dict]]:
    """Full-float val_miou argmax, ties to the later step. Mirrors the trainer.

    Loading and the val_miou None-refusal stay local (this script REFUSES a
    scoreless checkpoint where ensure_best_ckpts maps it to -inf); only the final
    argmax is shared, via ckptsel.pick_best_float_rule.
    """
    scored = []
    for p in candidates:
        ckpt = torch.load(p, map_location="cpu", weights_only=True)
        val_miou = ckpt.get("val_miou")
        if val_miou is None:
            raise SystemExit(
                f"{p} has no val_miou; cannot apply the trainer rule. "
                f"Use scripts/evaluation/ensure_best_ckpts.py (filename rule) instead."
            )
        scored.append({"name": p.name, "val_miou": float(val_miou),
                       "step": step_of_checkpoint(p, ckpt), "path": p})
    scored.sort(key=lambda d: (d["val_miou"], d["step"]))
    best = ckptsel.pick_best_float_rule(
        (d["val_miou"], d["step"], d["path"]) for d in scored
    )
    return best, [{k: v for k, v in d.items() if k != "path"} for d in scored]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, required=True,
                    help="Tree to walk; every dir holding *.pth is a checkpoint dir")
    ap.add_argument("--dry-run", action="store_true",
                    help="Report only; touch nothing")
    ap.add_argument("--replace-files", action="store_true",
                    help="Also replace a regular-file best.pth (the dereferenced "
                         "copy an object-store round-trip leaves behind) with a "
                         "symlink. Off by default so nothing is deleted unasked.")
    ap.add_argument("--report-out", type=Path, default=None)
    args = ap.parse_args()

    if not args.root.is_dir():
        raise SystemExit(f"Not a directory: {args.root}")

    ckpt_dirs = sorted({p.parent for p in args.root.rglob("*.pth")
                        if p.name != "best.pth"})
    if not ckpt_dirs:
        raise SystemExit(f"No checkpoints found under {args.root}")

    report = []
    n_linked = n_ok = n_skipped = n_mismatch = 0

    for d in ckpt_dirs:
        candidates = sorted(p for p in d.glob("*.pth") if p.name != "best.pth")
        best_path, scored = pick_best(candidates)
        link = d / "best.pth"
        rel = str(best_path.name)
        entry = {"dir": str(d), "pick": rel, "candidates": scored}

        if link.is_symlink():
            current = link.readlink().name
            if current == rel:
                entry["action"] = "already-correct"
                n_ok += 1
            else:
                entry["action"] = "would-repoint" if args.dry_run else "repointed"
                entry["previous"] = current
                n_mismatch += 1
                if not args.dry_run:
                    link.unlink()
                    link.symlink_to(rel)
                print(f"{d}: best.pth {current} -> {rel}")
        elif link.exists():
            # Regular file: the dereferenced copy from an object-store round-trip.
            if args.replace_files:
                entry["action"] = "would-replace-file" if args.dry_run else "replaced-file"
                n_linked += 1
                if not args.dry_run:
                    link.unlink()
                    link.symlink_to(rel)
                print(f"{d}: replaced file best.pth with symlink -> {rel}")
            else:
                entry["action"] = "skipped-regular-file"
                n_skipped += 1
                print(f"{d}: best.pth is a regular file; pass --replace-files to relink "
                      f"(would point to {rel})")
        else:
            entry["action"] = "would-create" if args.dry_run else "created"
            n_linked += 1
            if not args.dry_run:
                link.symlink_to(rel)
            print(f"{d}: best.pth -> {rel}")
        report.append(entry)

    print(f"\n{len(ckpt_dirs)} checkpoint dirs: {n_ok} already correct, "
          f"{n_linked} linked, {n_mismatch} repointed, {n_skipped} skipped"
          f"{' (dry run: nothing written)' if args.dry_run else ''}")
    if args.report_out is not None:
        args.report_out.parent.mkdir(parents=True, exist_ok=True)
        args.report_out.write_text(json.dumps(report, indent=2))
        print(f"Wrote {args.report_out}")


if __name__ == "__main__":
    main()
