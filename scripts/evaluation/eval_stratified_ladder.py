"""Per-checkpoint stratified evaluation over the pair-based val memmap.

Reproduces the trainer's exact val forward path (fp16 autocast, argmax on raw
fp16 logits) so the pooled mIoU we recompute matches each checkpoint's stored
`val_miou` to <1e-4 (the anchor that certifies the harness). On top of that it
accumulates stratified / pair-macro / boundary-band confusion matrices over a
threshold sweep on fp32 softmax water probabilities, for the downstream
H1/H2/H3 checkpoint-selection audit.

Everything is written under experiments/CHECKPOINT_LADDER/raw/<seed>/ — NEVER
into the training output dirs. The running training job is not touched.

Confusion-matrix convention throughout: code = 2*target + pred, i.e.
  0 = target0/pred0 (TN, land-as-land)
  1 = target0/pred1 (FP, land-as-water)
  2 = target1/pred0 (FN, water-as-land)
  3 = target1/pred1 (TP, water-as-water)
Reshaped to [[0,1],[2,3]] this is rows=target, cols=pred, matching
torchmetrics ConfusionMatrix, so class-1 = water.

Usage (VM):
    /home/noel/miniforge3/envs/torch211_cu128/bin/python \
        scripts/evaluation/eval_stratified_ladder.py \
        --ckpt-root outputs/stage2/upernet_tu-swin_base_patch4_window7_224 \
        --memmap-root /mnt/local_ssd/dataset \
        --strata-index data/full_val_strata/strata_index.npz \
        --out-root experiments/CHECKPOINT_LADDER/raw \
        --seeds s19 s58
"""
import argparse
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from src.data.datamodule import CoastalDataModule
from src.models.factory import SegmentationModelFactory

# Filename role parsing. Order matters: _last / _snap before the generic _miou.
STEP_RE = re.compile(r"_step(\d+)_")


def role_of(name: str) -> str:
    if name.endswith("_last.pth"):
        return "last"
    if "_snap_" in name:
        return "snap"
    if "_miou" in name:
        return "best"  # a retained top-k checkpoint
    return "other"


def resolve_seed_dir(ckpt_root: Path, seed: str) -> Path:
    """Seed dirs may be bare ('s58') or timestamped ('13-37-33_s19'). Match a
    subdir equal to the seed token or ending in '_<seed>'."""
    exact = ckpt_root / seed
    if exact.is_dir():
        return exact
    cands = sorted(
        d for d in ckpt_root.iterdir()
        if d.is_dir() and (d.name == seed or d.name.endswith("_" + seed))
    )
    if len(cands) == 1:
        return cands[0]
    if not cands:
        raise SystemExit(f"No seed dir for {seed} under {ckpt_root}")
    raise SystemExit(f"Ambiguous seed dirs for {seed}: {[c.name for c in cands]}")


def discover(ckpt_root: Path, seeds):
    """seed -> list of (step, role, path), best.pth symlink dropped by realpath."""
    found = {}
    for seed in seeds:
        seed_dir = resolve_seed_dir(ckpt_root, seed)
        if not seed_dir.is_dir():
            raise SystemExit(f"Missing seed dir {seed_dir}")
        seen_real = set()
        entries = []
        for p in sorted(seed_dir.glob("*.pth")):
            real = p.resolve()
            if real in seen_real:
                continue  # best.pth symlink duplicating a top-k file
            seen_real.add(real)
            if p.name == "best.pth":
                continue
            m = STEP_RE.search(p.name)
            if m is None:
                print(f"  [skip] no step in {p.name}", file=sys.stderr)
                continue
            entries.append((int(m.group(1)), role_of(p.name), p))
        if not entries:
            raise SystemExit(f"No checkpoints under {seed_dir}")
        entries.sort(key=lambda e: e[0])
        found[seed] = entries
    return found


def pooled_miou_from_cm4(cm4: np.ndarray) -> float:
    """cm4 = [TN, FP, FN, TP] (int64). Macro IoU over classes with union>0,
    float64, exactly as SegmentationMetrics.compute()."""
    tn, fp, fn, tp = [float(x) for x in cm4]
    cm = np.array([[tn, fp], [fn, tp]], dtype=np.float64)
    diag = np.diag(cm)
    pred_tot = cm.sum(0)
    tgt_tot = cm.sum(1)
    union = pred_tot + tgt_tot - diag
    m = union > 0
    return float((diag[m] / union[m]).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-root", type=Path, required=True)
    ap.add_argument("--memmap-root", type=str, required=True)
    ap.add_argument("--strata-index", type=Path, required=True)
    ap.add_argument("--out-root", type=Path, required=True)
    ap.add_argument("--arch", default="upernet")
    ap.add_argument("--encoder", default="tu-swin_base_patch4_window7_224")
    ap.add_argument("--seeds", nargs="+", default=["s19", "s58"])
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--anchor-tol", type=float, default=1e-4)
    args = ap.parse_args()

    device = torch.device("cuda")
    ckpt_root = args.ckpt_root.expanduser()
    out_root = args.out_root.expanduser()

    # --- strata index ---
    idx = np.load(args.strata_index, allow_pickle=True)
    N = int(idx["N"])
    stratum_id_np = idx["stratum_id"].astype(np.int64)
    pair_id_np = idx["pair_id"].astype(np.int64)
    P = len(idx["pair_names"])
    stratum_id = torch.from_numpy(stratum_id_np).to(device)   # [N] int64
    pair_id = torch.from_numpy(pair_id_np).to(device)         # [N] int64
    print(f"Strata index: N={N} pairs={P}", flush=True)

    # --- tau grid ---
    tau_grid = torch.linspace(0.05, 0.95, 19, device=device)  # [T]
    T = tau_grid.numel()
    tau_grid_np = tau_grid.cpu().numpy()

    # --- ckpts ---
    ckpts = discover(ckpt_root, args.seeds)
    total = sum(len(v) for v in ckpts.values())
    for seed, entries in ckpts.items():
        print(f"[{seed}] {len(entries)} ckpts: "
              + ", ".join(f"{s}({r})" for s, r, _ in entries), flush=True)
    print(f"Total {total} checkpoints", flush=True)

    # --- data (once) ---
    dm = CoastalDataModule(
        root_dir=args.memmap_root, H=224, W=224,
        batch_size=args.batch_size, val_batch_size=args.batch_size,
        num_workers=args.num_workers, augment=False,
        persistent_workers=True,
    )
    dm.setup()
    if dm.val_ds is None:
        raise SystemExit(f"No val.memmap under {args.memmap_root}")
    if len(dm.val_ds) != N:
        raise SystemExit(f"val_ds len {len(dm.val_ds)} != strata N {N}")
    val_dl = dm.val_dataloader()
    print(f"Val samples: {len(dm.val_ds)}  batches: {len(val_dl)}", flush=True)

    # --- model (once) ---
    model = SegmentationModelFactory.build(
        arch=args.arch, encoder_name=args.encoder,
        encoder_weights=None, in_channels=2, classes=2,
    ).to(device)

    all_anchor = []
    for seed, entries in ckpts.items():
        seed_out = out_root / seed
        seed_out.mkdir(parents=True, exist_ok=True)
        for step, role, path in entries:
            ck = torch.load(path, map_location=device, weights_only=True)
            model.load_state_dict(ck["model_state_dict"], strict=True)
            model.eval()
            payload_miou = float(ck.get("val_miou", float("nan")))

            # per-chip argmax CM (anchor): [N,4] int32
            chip_cm = torch.zeros((N, 4), dtype=torch.int32, device=device)
            # sweep accumulators (int64)
            strat_cm = torch.zeros((3, T, 4), dtype=torch.int64, device=device)
            pair_cm = torch.zeros((P, T, 4), dtype=torch.int64, device=device)
            band_cm = torch.zeros((2, T, 4), dtype=torch.int64, device=device)

            t0 = time.time()
            base = 0
            with torch.no_grad():
                for batch in val_dl:
                    x = batch["pixel_values"].to(device, non_blocking=True)
                    y = batch["labels"].to(device, non_blocking=True)  # [B,H,W] int64
                    B, H, W = y.shape
                    chip_idx = base + torch.arange(B, device=device)  # [B]

                    with torch.autocast(device_type="cuda", enabled=True):
                        logits = model(x)                       # fp16 under autocast
                    preds = torch.argmax(logits, dim=1)         # [B,H,W] argmax on raw fp16
                    probs = torch.softmax(logits.float(), dim=1)[:, 1]  # [B,H,W] fp32 water prob

                    valid = y != 255                            # [B,H,W]
                    yb = torch.clamp(y, 0, 1)                    # ignore rows dropped by valid mask

                    # ---- per-chip argmax CM ----
                    code = (2 * yb + preds).to(torch.int64)     # [B,H,W] in {0..3}
                    b_expand = chip_idx.view(B, 1, 1).expand(B, H, W)
                    flat_idx = (b_expand * 4 + code)[valid]      # valid pixels only
                    counts = torch.bincount(flat_idx, minlength=B * 4 + base * 4)
                    # counts is length >= (base+B)*4; slice this batch's window.
                    lo = base * 4
                    hi = (base + B) * 4
                    chip_cm.view(-1)[lo:hi] += counts[lo:hi].to(torch.int32)

                    # ---- boundary bands from GT (GPU) ----
                    water = (yb == 1) & valid                    # [B,H,W] bool
                    land = (yb == 0) & valid
                    wf = water.float().unsqueeze(1)               # [B,1,H,W]
                    lf = land.float().unsqueeze(1)
                    band_of = torch.zeros((B, H, W), dtype=torch.int8, device=device)
                    # k=2 first so k=1 overwrites (band_id: -1 none, 0 => within 1px, 1 => within 2px)
                    band_of.fill_(-1)
                    for k in (2, 1):
                        dw = F.max_pool2d(wf, 2 * k + 1, 1, k) > 0
                        dl = F.max_pool2d(lf, 2 * k + 1, 1, k) > 0
                        bandk = (dw & dl).squeeze(1) & valid      # [B,H,W]
                        band_of[bandk] = (k - 1)  # k=1 -> id 0 (+-1px), k=2 -> id 1 (+-2px)
                    # Note: k=1 band is a subset of k=2 band, so writing k=2 then
                    # k=1 leaves +-1px pixels tagged 0 and the +-2px ring tagged 1.
                    # For "IoU within +-2px" downstream we union bands {0,1}.

                    # ---- tau sweep ----
                    # Precompute per-pixel stratum & pair via chip broadcast, and
                    # the fixed pixel masks / gathered indices (tau-independent).
                    strat_px = stratum_id[chip_idx].view(B, 1, 1).expand(B, H, W)  # [B,H,W]
                    pair_px = pair_id[chip_idx].view(B, 1, 1).expand(B, H, W)
                    mixed_px = (strat_px == 2) & valid
                    yb_v = yb[valid]                              # target of valid px
                    strat_v = strat_px[valid]
                    pair_m = pair_px[mixed_px]                    # pair of mixed valid px
                    yb_m = yb[mixed_px]
                    band0_mask = (band_of == 0)                  # +-1px core
                    band1_mask = (band_of == 1)                  # +-2px ring
                    yb_b0 = yb[band0_mask]
                    yb_b1 = yb[band1_mask]
                    for ti in range(T):
                        tau = tau_grid[ti]
                        pt = (probs >= tau).to(torch.int64)       # [B,H,W]

                        # strat_cm[3,4]: index = strat*4 + (2*y+pred), valid px
                        code_v = 2 * yb_v + pt[valid]
                        s_idx = strat_v * 4 + code_v
                        strat_cm[:, ti, :] += torch.bincount(
                            s_idx, minlength=3 * 4).view(3, 4)

                        # pair_cm[P,4]: mixed chips only
                        code_m = 2 * yb_m + pt[mixed_px]
                        p_idx = pair_m * 4 + code_m
                        pair_cm[:, ti, :] += torch.bincount(
                            p_idx, minlength=P * 4).view(P, 4)

                        # band_cm[2,4]: within-1px core (id0), +-2px ring (id1)
                        band_cm[0, ti, :] += torch.bincount(
                            2 * yb_b0 + pt[band0_mask], minlength=4)
                        band_cm[1, ti, :] += torch.bincount(
                            2 * yb_b1 + pt[band1_mask], minlength=4)

                    base += B

            assert base == N, f"iterated {base} chips != {N}"

            # anchor: pooled mIoU from chip_cm
            chip_cm_np = chip_cm.cpu().numpy()          # [N,4] int32
            cm4 = chip_cm_np.sum(0).astype(np.int64)    # [4]
            pooled = pooled_miou_from_cm4(cm4)
            delta = abs(pooled - payload_miou)
            ok = delta < args.anchor_tol
            dt = time.time() - t0
            fn = path.name
            m = re.search(r"_miou(\d+\.\d+)", fn)
            fname_miou = float(m.group(1)) if m else float("nan")
            print(f"[{seed} step{step:<6} {role:>4}] pooled={pooled:.6f} "
                  f"payload={payload_miou:.6f} d={delta:.2e} "
                  f"{'OK' if ok else 'FAIL'}  ({dt:.0f}s)", flush=True)
            all_anchor.append((seed, step, role, payload_miou, pooled, delta, ok))

            # save
            stem = path.stem  # filename w/o .pth
            np.save(seed_out / f"{stem}_chipcm.npy", chip_cm_np)
            np.savez(
                seed_out / f"{stem}.npz",
                strat_cm=strat_cm.cpu().numpy(),
                pair_cm=pair_cm.cpu().numpy(),
                band_cm=band_cm.cpu().numpy(),
                tau_grid=tau_grid_np,
                step=np.int64(step),
                role=str(role),
                val_miou=np.float64(payload_miou),
                filename_miou=np.float64(fname_miou),
                pooled_miou_fp16argmax=np.float64(pooled),
                anchor_delta=np.float64(delta),
                ckpt_name=str(fn),
            )

            # ANCHOR GATE: if the FIRST ckpt fails, stop before burning GPU time.
            if not ok and len(all_anchor) == 1:
                dm.teardown()
                raise SystemExit(
                    f"ANCHOR FAILED on first checkpoint {fn}: pooled={pooled:.6f} "
                    f"payload={payload_miou:.6f} delta={delta:.2e} >= {args.anchor_tol}. "
                    "Harness invalid; stopping."
                )

    dm.teardown()

    n_ok = sum(1 for a in all_anchor if a[6])
    print("\n" + "=" * 60)
    print(f"ANCHOR: {n_ok}/{len(all_anchor)} checkpoints reproduced val_miou "
          f"to < {args.anchor_tol}", flush=True)
    if n_ok != len(all_anchor):
        print("SOME ANCHORS FAILED:", flush=True)
        for seed, step, role, pay, pool, d, ok in all_anchor:
            if not ok:
                print(f"  {seed} step{step} {role}: payload={pay:.6f} "
                      f"pooled={pool:.6f} d={d:.2e}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
