"""Paired 2 km block-bootstrap of each new architecture vs the legacy ResNet50-UNet.

Regenerates `new_vs_legacy_allmodels_2km_FULL.csv`: for every new architecture (3-seed
MEAN probability map) against each legacy reference (baseline, finetuned), on each of the
3 legacy-comparable dates, the paired IoU difference (new - legacy) with a 95% percentile
CI from a paired spatial block bootstrap (2 km blocks, shared draws so shared spatial
variance cancels). This is the same method used for ConvNeXtV2-vs-Swin-B in
`hampyeong_block_bootstrap.py`, extended to all architectures vs the legacy models.

The legacy models are single runs (no seeds), so each new architecture is paired against
a legacy run using the architecture's 3-seed-MEAN thresholded map. Threshold 0.5, all
1,179,967 valid pixels.

Run-dir maps live in MODEL_RUNS below (not imported from hampyeong_model_comparison, so
that Swin-Large / ConvNeXtV2-Large / any per-scene raster override can be set here
explicitly). To swap a defective run's raster, override its path in RASTER_OVERRIDES.

Run with an interpreter that has rasterio + scipy, e.g.
    /opt/homebrew/Caskroom/miniforge/base/envs/eda/bin/python
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import from_bounds

_SCRIPTS = Path(__file__).resolve().parent.parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from evaluation.hampyeong_model_comparison import gt_path, valid_mask_path, EXPECTED_VALID_PIXELS  # noqa: E402

DEFAULT_NAS_ROOT = "/Volumes/WD_8tb_RedPlus_NAS_A/MACKBOOK_AIR_M2_BACKUP/Documents/EOS/ACDC"
DEFAULT_RUNS_ROOT = "experiments/hampyeong/runs"
DEFAULT_OUT = "experiments/hampyeong/evaluation/new_vs_legacy_allmodels_2km_FULL.csv"

BLOCK_PX = 200          # 2 km at ~10 m/px
N_BOOT = 2000
BOOT_SEED = 42
SEEDS = (19, 42, 58)
DATES = ["20210305", "20210422", "20210621"]   # dates the legacy models were run on

SCENE = {
    "20210305": "S1B_IW_GRDH_1SDV_20210305T213224_20210305T213249_025885_031658_6933_Clipped",
    "20210422": "S1B_IW_GRDH_1SDV_20210422T213225_20210422T213250_026585_032CBA_CF3C_Clipped",
    "20210621": "S1B_IW_GRDH_1SDV_20210621T213228_20210621T213253_027460_034782_6C41_Clipped",
}

# New architectures: label -> the architecture token in the run-dir name. Run dirs are
# DISCOVERED by pattern (arch token + seed), newest timestamp wins -- so a re-inference
# that replaces a run dir (e.g. the Swin-B s42 defect re-run) is picked up automatically
# with NO code change here. Run-dir names look like:
#   dev_sweep_all_hampyeong_s{seed}_{ts}_{ARCH_TOKEN}_native224_weighted_224_b0_s32
ARCH_TOKEN: dict[str, str] = {
    "Swin-B": "upernet_swin_base_224",
    "Swin-Large": "upernet_swin_large_224",
    "ConvNeXtV2": "upernet_tu-convnextv2_base",
    "ConvNeXtV2-Large": "upernet_tu-convnextv2_large",
    "DPT-ViT-B": "dpt_vit_b_16",
    "SegFormer-B4": "segformer_mit_b4",
    "DeepLabV3+": "deeplabv3plus_resnet50",
    "UNet-R50": "unet_resnet50",
    "UNet++-R50": "unetplusplus_resnet50",
}

# Which architectures to include (and output order). The original FULL csv had these 8
# (no ConvNeXtV2-Large). Add "ConvNeXtV2-Large" to include it.
MODELS = ["Swin-B", "Swin-Large", "ConvNeXtV2", "DPT-ViT-B", "SegFormer-B4",
          "DeepLabV3+", "UNet-R50", "UNet++-R50"]


def _resolve_run_dir(runs_root: Path, model: str, seed: int) -> Path:
    """Newest run dir for (arch, seed). unet_resnet50 must not match unetplusplus_resnet50."""
    token = ARCH_TOKEN[model]
    prefix = f"dev_sweep_all_hampyeong_s{seed}_"
    suffix = f"_{token}_native224_weighted_224_b0_s32"
    matches = sorted(p for p in runs_root.glob(f"{prefix}*{suffix}")
                     if p.is_dir() and p.name.startswith(prefix) and p.name.endswith(suffix))
    if not matches:
        raise FileNotFoundError(f"no run dir for {model} (token {token}) seed {seed} under {runs_root}")
    # sort key = the ...T######Z timestamp between prefix and token; lexicographic works
    return matches[-1]

# Per-(model, seed, date) absolute raster overrides, e.g. to substitute a re-run that
# replaced a defective inference. Key: (model, seed, date_str) -> absolute .tif path.
# Leave empty to use the MODEL_RUNS path. Any override is asserted to exist and is echoed.
RASTER_OVERRIDES: dict[tuple[str, int, str], str] = {
    # Example (NOT active): swap Swin-B s42 @20210422 to a re-run raster. Uncomment + set
    # path to activate, and DOCUMENT the swap (see docs / defect record).
    # ("Swin-B", 42, "20210422"):
    #     "/absolute/path/to/..._probability_water.tif",
}

LEGACY = {
    "baseline": "Pretrained_Baseline_Resnet50_for_reproj",
    "finetuned": "Baseline_Resnet50_reproj_FINETUNED",
}


def _new_raster_path(runs_root: Path, model: str, seed: int, date: str) -> Path:
    key = (model, seed, date)
    if key in RASTER_OVERRIDES:
        p = Path(RASTER_OVERRIDES[key])
        if not p.exists():
            raise FileNotFoundError(f"override raster missing: {p}")
        print(f"  [override] {model} s{seed} {date} -> {p}")
        return p
    scene = SCENE[date]
    return _resolve_run_dir(runs_root, model, seed) / scene / f"{scene}_probability_water.tif"


def _legacy_raster_path(nas_root: Path, ref: str, date: str) -> Path:
    return (nas_root / "sen12coast/Inference/Hampeyeong/Models" / LEGACY[ref]
            / "DESC/Geotiffs_clipped_Val_AOI" / f"Inference_DESC_{date}_Clipped_VAL_AOI.tif")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nas-root", type=Path, default=Path(DEFAULT_NAS_ROOT))
    ap.add_argument("--runs-root", type=Path, default=Path(DEFAULT_RUNS_ROOT))
    ap.add_argument("--out", type=Path, default=Path(DEFAULT_OUT))
    args = ap.parse_args()
    if not args.nas_root.exists():
        raise SystemExit(f"NAS root not found: {args.nas_root} (is the volume mounted?)")

    # valid mask -> boolean, bounds, and 2 km block ids in row-major valid-pixel order
    with rasterio.open(valid_mask_path(args.nas_root)) as m:
        valid = m.read(1) == 1
        mb = m.bounds
    if int(valid.sum()) != EXPECTED_VALID_PIXELS:
        raise AssertionError(f"valid mask has {valid.sum()} px, expected {EXPECTED_VALID_PIXELS}")
    r0, c0 = np.nonzero(valid)
    braw = (r0 // BLOCK_PX) * 1_000_000 + (c0 // BLOCK_PX)
    _, bidx = np.unique(braw, return_inverse=True)
    n_blocks = int(bidx.max() + 1)
    n_valid = int(valid.sum())

    def read_valid(path: Path) -> np.ndarray:
        with rasterio.open(path) as s:
            w = from_bounds(*mb, transform=s.transform).round_offsets().round_lengths()
            return s.read(1, window=w)[valid]

    def new_mean_binary(model: str, date: str) -> np.ndarray:
        acc = np.zeros(n_valid)
        for seed in SEEDS:
            acc += read_valid(_new_raster_path(args.runs_root, model, seed, date))
        return (acc / len(SEEDS)) >= 0.5

    def legacy_binary(ref: str, date: str) -> np.ndarray:
        return read_valid(_legacy_raster_path(args.nas_root, ref, date)) > 0

    def gt_binary(date: str) -> np.ndarray:
        return read_valid(gt_path(args.nas_root, date)) > 0

    def block_counts(yt: np.ndarray, yp: np.ndarray) -> np.ndarray:
        # per-block [tp, fp, fn]
        return np.stack([
            np.bincount(bidx, weights=yt & yp, minlength=n_blocks),
            np.bincount(bidx, weights=(~yt) & yp, minlength=n_blocks),
            np.bincount(bidx, weights=yt & (~yp), minlength=n_blocks),
        ], axis=1)

    def iou(counts: np.ndarray, mvec: np.ndarray) -> float:
        agg = mvec @ counts
        d = agg.sum()
        return float(agg[0] / d) if d > 0 else float("nan")

    def paired_boot(cN: np.ndarray, cL: np.ndarray) -> tuple[float, float, float, bool]:
        rng = np.random.default_rng(BOOT_SEED)
        diffs = np.empty(N_BOOT)
        for i in range(N_BOOT):
            draw = rng.integers(0, n_blocks, n_blocks)
            mvec = np.bincount(draw, minlength=n_blocks).astype(np.float64)
            diffs[i] = iou(cN, mvec) - iou(cL, mvec)
        full = np.ones(n_blocks)
        point = iou(cN, full) - iou(cL, full)
        lo, hi = np.percentile(diffs, [2.5, 97.5])
        return point, float(lo), float(hi), bool(lo > 0 or hi < 0)

    # echo the resolved run dir per (model, seed) so re-inferred/replaced dirs are visible
    print("Resolved run dirs (newest per arch+seed):")
    for m in MODELS:
        for seed in SEEDS:
            print(f"  {m:18s} s{seed}: {_resolve_run_dir(args.runs_root, m, seed).name}")
    print()

    rows: list[dict] = []
    for date in DATES:
        yt = gt_binary(date)
        cN = {m: block_counts(yt, new_mean_binary(m, date)) for m in MODELS}
        for ref in ("baseline", "finetuned"):
            cL = block_counts(yt, legacy_binary(ref, date))
            for m in MODELS:
                d, lo, hi, sig = paired_boot(cN[m], cL)
                rows.append({"date": int(date), "model": m, "legacy_ref": ref,
                             "delta": d, "ci_lo": lo, "ci_hi": hi, "sig": sig})
        print(f"{date} done")

    df = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    n_sig = int(df.groupby("legacy_ref").sig.sum().to_dict().get("baseline", 0)), int(df.groupby("legacy_ref").sig.sum().to_dict().get("finetuned", 0))
    print(f"\nWrote {len(df)} rows to {args.out}  (n_blocks={n_blocks}, n_boot={N_BOOT})")
    print(f"significant contrasts: baseline {n_sig[0]}, finetuned {n_sig[1]}")


if __name__ == "__main__":
    main()
