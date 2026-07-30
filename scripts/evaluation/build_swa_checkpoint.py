"""Average snapshot-spine weights into a gate-ready SWA checkpoint.

Stage-2 runs save a model-only snapshot spine (`*_snap_*.pth`) at every
validation in the post-LR-decay window (src/engine/trainer.py). When those
snapshots sit in one loss basin (flat val mIoU), a Stochastic Weight Averaging
(SWA) mean of their weights (Izmailov et al. 2018) is a natural fourth candidate
in the checkpoint-selection gate alongside {early-best, last, best-among-late}.

This CLI:
  1. discovers `*_snap_*.pth` in a seed dir (parsing the trainer's naming),
  2. averages their `model_state_dict` (float tensors in float32, cast back;
     non-float buffers taken from the highest-step ingredient),
  3. writes a model-only checkpoint that run_inference.py's loader consumes
     (`build_model_and_load_checkpoint` reads `model_state_dict` + optional
     step/val_miou) plus a provenance JSON sidecar,
  4. optionally refreshes BatchNorm running stats over the project memmap
     dataset via torch.optim.swa_utils.update_bn (--bn-refresh),
  5. prints (never runs) a four-way gate command manifest for
     scripts/run_inference.py using the production native224-s32 preset.

Usage:
    python scripts/evaluation/build_swa_checkpoint.py \
        --seed-dir outputs/stage2/upernet_tu-swin_base_patch4_window7_224/s19 \
        --min-step 31200

    # optional BatchNorm refresh (ResNet decoders benefit most):
    python scripts/evaluation/build_swa_checkpoint.py \
        --seed-dir outputs/stage2/unet_resnet50/s19 \
        --bn-refresh --model-arch unet --encoder resnet50 \
        --bn-memmap-root ~/dataset --bn-file train_10pct.memmap

    # synthetic self-test (no data needed):
    python scripts/evaluation/build_swa_checkpoint.py --self-test
"""
import argparse
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

# Shared checkpoint resolution / distinctness guards -- scripts/evaluation/vm/ckptsel.py
# is stdlib-only, so it imports here without an editable install.
sys.path.insert(0, str(Path(__file__).resolve().parent / "vm"))
import ckptsel  # noqa: E402

# Seed dirs are named `s19` / `s42` / `s58`, or `13-37-33_s19` when the trainer
# stamped them. Used to derive the seed token every ingredient filename must carry.
SEED_DIR_RE = re.compile(r"(?:^|_)(s\d+)$")

# Parses the step out of the trainer's snapshot naming:
#   {arch}_{encoder}_s{seed}_step{N}_snap_miou{v:.6f}.pth
SNAP_STEP_RE = re.compile(r"_step(\d+)_snap_")
# The val mIoU embedded in the same name (best-effort, for provenance only).
SNAP_MIOU_RE = re.compile(r"_snap_miou([0-9.]+)\.pth$")


# --------------------------------------------------------------------------- #
# Discovery / selection
# --------------------------------------------------------------------------- #
def discover_snapshots(seed_dir: Path, min_step, last_k):
    """Return [(step, path)] of `*_snap_*.pth` in seed_dir, sorted by step.

    Applies `--min-step` (keep step >= min_step) then `--last-k` (keep the K
    highest-step survivors). Order matters: --min-step first bounds the window,
    then --last-k trims from the top of what remains.
    """
    found = []
    for p in sorted(seed_dir.glob("*_snap_*.pth")):
        m = SNAP_STEP_RE.search(p.name)
        if m is None:
            continue
        found.append((int(m.group(1)), p))
    found.sort(key=lambda t: t[0])

    if min_step is not None:
        found = [(s, p) for (s, p) in found if s >= min_step]
    if last_k is not None:
        found = found[-last_k:]
    return found


def parse_step(path: Path):
    m = SNAP_STEP_RE.search(path.name)
    return int(m.group(1)) if m else None


def parse_miou(path: Path):
    m = SNAP_MIOU_RE.search(path.name)
    return float(m.group(1)) if m else None


# --------------------------------------------------------------------------- #
# Averaging
# --------------------------------------------------------------------------- #
def average_state_dicts(state_dicts):
    """Equal-weight average a list of model_state_dicts.

    - Floating-point tensors: accumulate in float32, divide by n, cast back to
      the original dtype.
    - Non-floating entries (e.g. BatchNorm `num_batches_tracked`, int64): take
      the value from the LAST state dict (assumed highest-step ingredient) — a
      count/index is meaningless to average.
    - Aborts if key sets or tensor shapes differ across ingredients.

    Ingredients must be ordered so state_dicts[-1] is the highest-step snapshot.
    """
    if len(state_dicts) < 2:
        raise ValueError(f"Need >= 2 ingredients to average, got {len(state_dicts)}.")

    ref_keys = set(state_dicts[0].keys())
    for i, sd in enumerate(state_dicts[1:], start=1):
        if set(sd.keys()) != ref_keys:
            missing = ref_keys - set(sd.keys())
            extra = set(sd.keys()) - ref_keys
            raise ValueError(
                f"Key set mismatch in ingredient {i}: "
                f"missing={sorted(missing)} extra={sorted(extra)}"
            )

    n = len(state_dicts)
    last = state_dicts[-1]
    out = {}
    for key in state_dicts[0].keys():
        ref = state_dicts[0][key]
        # Verify shapes agree across all ingredients.
        for i, sd in enumerate(state_dicts[1:], start=1):
            if sd[key].shape != ref.shape:
                raise ValueError(
                    f"Shape mismatch for '{key}': ingredient 0 has {tuple(ref.shape)}, "
                    f"ingredient {i} has {tuple(sd[key].shape)}"
                )

        if torch.is_floating_point(ref):
            acc = torch.zeros_like(ref, dtype=torch.float32)
            for sd in state_dicts:
                acc += sd[key].to(torch.float32)
            acc /= n
            out[key] = acc.to(ref.dtype)
        else:
            # Non-float buffer: take from the highest-step ingredient verbatim.
            out[key] = last[key].clone()
    return out


# --------------------------------------------------------------------------- #
# BatchNorm refresh
# --------------------------------------------------------------------------- #
def refresh_batchnorm(model_state_dict, args):
    """Load averaged weights into a model and refresh BN running stats.

    Returns (refreshed_state_dict, bn_refreshed: bool). Skips gracefully (warns,
    returns the input unchanged with bn_refreshed=False) if the memmap file is
    absent. LayerNorm-only encoders (Swin/ConvNeXt) only carry BN in their
    decoder; pure-ResNet backbones benefit most.
    """
    from src.models.factory import SegmentationModelFactory
    from src.data.dataset import CoastalMemmapDataset, MemmapSpec
    from torch.utils.data import DataLoader

    memmap_path = Path(args.bn_memmap_root).expanduser() / args.bn_file
    if not memmap_path.exists():
        print(
            f"[BN] WARNING: memmap not found at {memmap_path}; skipping BatchNorm "
            "refresh (bn_refreshed=false).",
            file=sys.stderr,
        )
        return model_state_dict, False

    device = torch.device(args.bn_device if torch.cuda.is_available() or args.bn_device == "cpu" else "cpu")

    model = SegmentationModelFactory.build(
        arch=args.model_arch,
        encoder_name=args.encoder,
        encoder_weights=None,
        in_channels=2,
        classes=2,
    )
    model.load_state_dict(model_state_dict, strict=True)
    model.to(device)

    # Snapshot BN buffers so we can report whether any actually moved.
    def bn_snapshot():
        return {
            name: (buf.detach().clone())
            for name, buf in model.named_buffers()
            if name.endswith("running_mean") or name.endswith("running_var")
        }

    before = bn_snapshot()
    if not before:
        print(
            "[BN] No BatchNorm running-stat buffers found in this model "
            "(LayerNorm-only encoder with no decoder BN); nothing to refresh.",
            file=sys.stderr,
        )
        return model_state_dict, False

    spec = MemmapSpec(path=str(memmap_path))
    dataset = CoastalMemmapDataset(spec)
    loader = DataLoader(
        dataset,
        batch_size=args.bn_batch_size,
        shuffle=True,
        num_workers=args.bn_num_workers,
        pin_memory=(device.type == "cuda"),
    )

    def input_generator(dataloader, max_batches, dev):
        """Yield pixel-values tensors; update_bn only needs the model inputs."""
        for i, batch in enumerate(dataloader):
            if i >= max_batches:
                break
            yield batch["pixel_values"].to(dev, non_blocking=True)

    print(
        f"[BN] Refreshing BatchNorm running stats over up to {args.bn_batches} "
        f"batches of {args.bn_batch_size} from {memmap_path} on {device}...",
        file=sys.stderr,
    )
    torch.optim.swa_utils.update_bn(
        input_generator(loader, args.bn_batches, device), model, device=device
    )

    after = bn_snapshot()
    moved = sum(
        1 for k in before if not torch.equal(before[k].cpu(), after[k].cpu())
    )
    print(
        f"[BN] Refresh complete: {moved}/{len(before)} running-stat buffers changed.",
        file=sys.stderr,
    )

    # Return weights on CPU for a portable, gate-ready checkpoint.
    refreshed = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    return refreshed, True


# --------------------------------------------------------------------------- #
# Command manifest
# --------------------------------------------------------------------------- #
def find_best_pth(seed_dir: Path):
    """Plain existence check, deliberately NOT ckptsel.resolve_best.

    The result only feeds the PRINTED gate-command manifest, where `None` becomes
    a "# SKIPPED: no candidate" line. Resolving/validating the symlink here would
    make an unrelated best.pth problem abort a build that never loads it.
    """
    p = seed_dir / "best.pth"
    return p if p.exists() else None


def find_last_pth(seed_dir: Path):
    """The single `*_last.pth`, via ckptsel.resolve_last.

    HARDENING: the old code took `sorted(...)[-1]`, so with more than one
    `*_last.pth` in the dir it silently picked by sort order. ckptsel.resolve_last
    RAISES on 0 or >1 and lists the candidates; 0 is mapped back to `None` here
    because a missing `last` is a legitimate SKIP line in the printed manifest,
    whereas an ambiguous one is a wiring error that must surface.
    """
    if not sorted(seed_dir.glob("*_last.pth")):
        return None
    try:
        return ckptsel.resolve_last(seed_dir)
    except ckptsel.CkptSelError as exc:
        raise SystemExit(f"[SWA] ambiguous 'last' arm for the gate manifest: {exc}") from exc


def best_among_late(snapshots):
    """Highest val-mIoU snapshot in the selected window (best-effort by name)."""
    scored = [(parse_miou(p), s, p) for (s, p) in snapshots]
    scored = [t for t in scored if t[0] is not None]
    if not scored:
        return None
    scored.sort(key=lambda t: (t[0], t[1]))  # by miou, then step
    return scored[-1][2]


NATIVE224_S32_OVERRIDES = [
    "inference.data.tile_size=224",
    "inference.data.buffer_size=0",
    "inference.data.stride=32",
    'inference.stitching.mode="weighted_blend"',
    'inference.stitching.blend_window="hann"',
    "inference.stitching.min_weight=0.001",
    'inference.data.edge_policy="shift_inward"',
    "inference.data.batch_size=256",
]


def print_command_manifest(seed_dir: Path, swa_path: Path, snapshots, model_arch, encoder):
    """Print (do NOT execute) the four-way gate command manifest to stdout."""
    best = find_best_pth(seed_dir)
    last = find_last_pth(seed_dir)
    late = best_among_late(snapshots)

    variants = [
        ("early_best", best, "best.pth (val-argmax early checkpoint)"),
        ("last", last, "*_last.pth (final-step checkpoint)"),
        ("best_among_late", late, "highest-val snapshot in the SWA window"),
        ("swa", swa_path, "SWA-averaged snapshot spine (this file)"),
    ]

    model_overrides = []
    if model_arch:
        model_overrides.append(f"model.arch={model_arch}")
    if encoder:
        model_overrides.append(f"model.encoder_name={encoder}")

    lines = []
    lines.append("")
    lines.append("=" * 78)
    lines.append("FOUR-WAY CHECKPOINT-SELECTION GATE  (native224-s32 preset) — COMMANDS ONLY")
    lines.append("=" * 78)
    lines.append("# Set these machine-specific paths, then run each block below.")
    lines.append("# (nothing here is executed automatically)")
    lines.append('INPUT_DIR="<PLACEHOLDER: dir of S1 scenes, e.g. /home/noel/data_semarang_concurrent>"')
    lines.append('INPUT_GLOB="<PLACEHOLDER: *.tif>"')
    lines.append('OUTPUT_ROOT="<PLACEHOLDER: inference output root, e.g. outputs/inference/swa_gate>"')
    lines.append("")

    for name, path, desc in variants:
        lines.append(f"# --- {name}: {desc} ---")
        if path is None:
            lines.append(f"#   SKIPPED: no candidate found for '{name}' in {seed_dir}")
            lines.append("")
            continue
        parts = [
            "python scripts/run_inference.py",
            f"inference.checkpoint_path={path}",
        ]
        parts += model_overrides
        parts += [
            'inference.data.input_dir="$INPUT_DIR"',
            'inference.data.input_glob="$INPUT_GLOB"',
            'inference.output.root_dir="$OUTPUT_ROOT"',
            f"inference.output.run_name=gate_{name}",
        ]
        parts += NATIVE224_S32_OVERRIDES
        lines.append(" \\\n    ".join(parts))
        lines.append("")

    lines.append("# Then score each run's 6-scene AUC-ROC with:")
    lines.append("#   python scripts/evaluate_indonesia_inference_run_aucroc.py <config-or-run-args>")
    lines.append("=" * 78)
    print("\n".join(lines))


# --------------------------------------------------------------------------- #
# Self-test
# --------------------------------------------------------------------------- #
def run_self_test():
    """Synthetic averaging correctness: 1/2/3 floats -> 2.0; int64 -> last."""
    sds = []
    for c in (1.0, 2.0, 3.0):
        sds.append({
            "w": torch.full((2, 3), float(c), dtype=torch.float32),
            "num_batches_tracked": torch.tensor(int(c * 10), dtype=torch.int64),
        })
    out = average_state_dicts(sds)
    assert torch.allclose(out["w"], torch.full((2, 3), 2.0)), out["w"]
    assert out["w"].dtype == torch.float32
    assert out["num_batches_tracked"].item() == 30, out["num_batches_tracked"]
    assert out["num_batches_tracked"].dtype == torch.int64

    # Key-set mismatch must abort.
    try:
        average_state_dicts([{"a": torch.zeros(1)}, {"b": torch.zeros(1)}])
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError on key-set mismatch")

    # Shape mismatch must abort.
    try:
        average_state_dicts([{"a": torch.zeros(2)}, {"a": torch.zeros(3)}])
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError on shape mismatch")

    print("SELF-TEST PASS: float mean == 2.0, int64 buffer taken from last (30), "
          "key/shape mismatches abort.")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def build_parser():
    p = argparse.ArgumentParser(
        description="Average snapshot-spine weights into a gate-ready SWA checkpoint.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--seed-dir", type=Path,
                   help="Directory containing *_snap_*.pth snapshots (and best.pth / *_last.pth).")
    p.add_argument("--snapshots", type=Path, nargs="+", default=None,
                   help="Explicit snapshot paths (overrides discovery in --seed-dir).")
    p.add_argument("--min-step", type=int, default=None,
                   help="Keep only snapshots with step >= this (window lower bound).")
    p.add_argument("--last-k", type=int, default=None,
                   help="Keep only the K highest-step snapshots (applied after --min-step).")
    p.add_argument("--out", type=Path, default=None,
                   help="Output checkpoint path (default: inside --seed-dir with an auto name).")
    p.add_argument("--prefix", type=str, default=None,
                   help="Filename prefix for the auto-named output (default: derived from ingredient names).")

    # BN refresh
    p.add_argument("--bn-refresh", action="store_true",
                   help="Refresh BatchNorm running stats over the project memmap "
                        "after averaging. LayerNorm-only encoders (Swin/ConvNeXt) "
                        "need this only for decoder BN layers; ResNet models benefit most.")
    p.add_argument("--bn-memmap-root", type=Path, default=None,
                   help="Directory holding the BN-refresh memmap file.")
    p.add_argument("--bn-file", type=str, default="train_10pct.memmap",
                   help="Memmap filename under --bn-memmap-root for BN refresh.")
    p.add_argument("--bn-batches", type=int, default=200,
                   help="Max number of batches to stream through update_bn.")
    p.add_argument("--bn-batch-size", type=int, default=256,
                   help="Batch size for the BN-refresh dataloader.")
    p.add_argument("--bn-num-workers", type=int, default=4,
                   help="DataLoader workers for BN refresh.")
    p.add_argument("--bn-device", type=str, default="cuda",
                   help="Device for BN refresh (falls back to cpu if no CUDA).")
    p.add_argument("--model-arch", type=str, default=None,
                   help="Model arch (required for --bn-refresh; also used in the manifest).")
    p.add_argument("--encoder", type=str, default=None,
                   help="Encoder name (required for --bn-refresh; also used in the manifest).")

    p.add_argument("--self-test", action="store_true",
                   help="Run the synthetic averaging correctness test and exit.")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)

    if args.self_test:
        run_self_test()
        return

    # Resolve ingredients.
    if args.snapshots:
        snaps = sorted(
            [(parse_step(p) if parse_step(p) is not None else -1, p) for p in args.snapshots],
            key=lambda t: t[0],
        )
        seed_dir = args.snapshots[0].parent
    else:
        if args.seed_dir is None:
            raise SystemExit("Provide --seed-dir (for discovery) or --snapshots (explicit).")
        seed_dir = args.seed_dir
        snaps = discover_snapshots(seed_dir, args.min_step, args.last_k)

    if len(snaps) < 2:
        raise SystemExit(
            f"Need >= 2 snapshot ingredients, found {len(snaps)} in {seed_dir} "
            f"(min_step={args.min_step}, last_k={args.last_k})."
        )

    if args.bn_refresh and (not args.model_arch or not args.encoder):
        raise SystemExit("--bn-refresh requires --model-arch and --encoder.")

    paths = [p for (_, p) in snaps]
    steps = [s for (s, _) in snaps]
    print(f"[SWA] {len(paths)} ingredient(s) from {seed_dir}:")
    for s, p in snaps:
        print(f"       step {s:>7}  {p.name}")

    # --- pre-average guards (both catch an already-mislabeled ingredient set) ---
    # (i) sha256 distinctness: a duplicated snapshot file (same weights under two
    # names) silently down-weights the rest of the average, and the output is
    # indistinguishable from a correct SWA build.
    try:
        ckptsel.assert_distinct_weights({p.name: p for p in paths})
    except ckptsel.CkptSelError as exc:
        raise SystemExit(f"[SWA] {exc}") from exc
    # (ii) seed token: an ingredient copied in from another seed's dir passes every
    # path check. Only checkable when the seed is derivable from the dir NAME --
    # `--snapshots` with an arbitrary parent dir is skipped silently rather than
    # guessed at.
    seed_m = SEED_DIR_RE.search(seed_dir.name)
    if seed_m:
        seed_token = seed_m.group(1)
        try:
            for p in paths:
                ckptsel.require_seed_token(p, seed_token)
        except ckptsel.CkptSelError as exc:
            raise SystemExit(f"[SWA] {exc}") from exc
        print(f"[SWA] Guards OK: {len(paths)} distinct sha256, all carry _{seed_token}_.")
    else:
        print(f"[SWA] Guards OK: {len(paths)} distinct sha256 "
              f"(seed-token check skipped: no s<N> in dir name {seed_dir.name!r}).")

    # Load and average (loaded in ascending-step order; last == highest step).
    state_dicts = []
    for _, p in snaps:
        ck = torch.load(str(p), map_location="cpu", weights_only=True)
        if "model_state_dict" not in ck:
            raise SystemExit(f"Snapshot missing 'model_state_dict': {p}")
        state_dicts.append(ck["model_state_dict"])

    averaged = average_state_dicts(state_dicts)
    print(f"[SWA] Averaged {len(averaged)} tensors "
          f"({sum(1 for v in averaged.values() if torch.is_floating_point(v))} float, "
          f"{sum(1 for v in averaged.values() if not torch.is_floating_point(v))} non-float).")

    # Optional BN refresh.
    bn_refreshed = False
    if args.bn_refresh:
        averaged, bn_refreshed = refresh_batchnorm(averaged, args)

    # Resolve output path.
    # DOCUMENTED TENSION (unchanged on purpose): the default output lands INSIDE
    # the training seed dir, so a derived artifact sits among the trainer's own
    # checkpoints. Downstream selectors therefore have to exclude it explicitly
    # (ckptsel.SWA_RE / is_best_candidate). Moving it is a behavior change that
    # would orphan the existing SWA builds and their provenance sidecars, so it is
    # out of scope here -- pass --out to write elsewhere.
    min_step, max_step = steps[0], steps[-1]
    if args.out is not None:
        out_path = args.out
    else:
        prefix = args.prefix
        if prefix is None:
            # Derive from the highest-step ingredient name, up to the _step token.
            base = paths[-1].name
            prefix = base.split("_step")[0] if "_step" in base else "model"
        out_name = f"{prefix}_swa{len(paths)}_step{min_step}-{max_step}.pth"
        out_path = seed_dir / out_name
    out_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "model_state_dict": averaged,
        "step": max_step,
        "val_miou": None,
        "swa": {
            "ingredients": [p.name for p in paths],
            "n": len(paths),
            "bn_refreshed": bn_refreshed,
        },
    }
    torch.save(payload, str(out_path))
    print(f"[SWA] Wrote checkpoint -> {out_path}")

    # Provenance sidecar.
    try:
        from src.utils.inference_outputs import get_git_commit
        git_commit = get_git_commit()
    except Exception:
        git_commit = None

    provenance = {
        "output_checkpoint": str(out_path),
        "seed_dir": str(seed_dir),
        "n_ingredients": len(paths),
        "step_window": {"min": min_step, "max": max_step},
        "selection": {
            "min_step": args.min_step,
            "last_k": args.last_k,
            "explicit_snapshots": bool(args.snapshots),
        },
        "ingredients": [
            {"name": p.name, "step": parse_step(p), "val_miou": parse_miou(p)}
            for p in paths
        ],
        "bn_refresh": {
            "enabled": bool(args.bn_refresh),
            "bn_refreshed": bn_refreshed,
            "memmap_root": str(args.bn_memmap_root) if args.bn_memmap_root else None,
            "bn_file": args.bn_file,
            "bn_batches": args.bn_batches,
            "bn_batch_size": args.bn_batch_size,
            "model_arch": args.model_arch,
            "encoder": args.encoder,
        },
        "created_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "git_commit": git_commit,
    }
    prov_path = Path(str(out_path) + ".provenance.json")
    prov_path.write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    print(f"[SWA] Wrote provenance -> {prov_path}")

    # Command manifest (stdout, no execution).
    print_command_manifest(seed_dir, out_path, snaps, args.model_arch, args.encoder)


if __name__ == "__main__":
    main()
