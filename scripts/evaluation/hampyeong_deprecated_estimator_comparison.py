"""Reproduce the DEPRECATED manuscript pixel-bootstrap estimator on current models.

    ┌───────────────────────────────────────────────────────────────────────────┐
    │ NOT A MANUSCRIPT ARTIFACT. These numbers reproduce the *retired* estimator │
    │ purely to show co-authors WHY it was retired (its CIs are uninformatively  │
    │ wide). They must go NOWHERE near the current manuscript. The defensible     │
    │ inference lives in STATISTICAL_ASSESSMENT.md + the pooled/block-bootstrap   │
    │ CSVs. Output: deprecated_estimator_comparison.csv (for co-author discussion).│
    └───────────────────────────────────────────────────────────────────────────┘


The manuscript's Table 2 was produced by sen12coast_global_dl's
`subsampled_two_proportion_ztest.py`, which bootstraps by SUBSAMPLING 400 pixels with
replacement (10,000 iterations) and reports 2.5/97.5 percentile CIs. Its p-value was
degenerate (pinned ~0.5) and is deliberately omitted here.

This script applies that exact estimator -- 400-pixel subsample, 10,000 bootstraps,
percentile CI, six metrics (oa/f1/precision/recall/iou/mcc) -- to the CURRENT models:
Swin-B and ConvNeXtV2 (3-seed-mean, stride 32) vs the legacy ResNet50-UNet baseline and
finetuned, on the 3 dates the legacy runs cover.

WHY: to show, apples-to-apples, what the old methodology would conclude. The takeaway is
that a 400-pixel subsample gives CIs so wide (~±0.025 IoU) that no model is distinguishable
from any other -- the exact weakness the 2 km block bootstrap was built to fix. These numbers
are for a methodology comparison only; the defensible inference is in STATISTICAL_ASSESSMENT.md.

Faithful to the original: np.random.seed(i) is reset inside the loop, so the same 400-index
draw is reused across models each iteration (which pairs the models); sample_size=400;
n_bootstraps=10000; metrics via sklearn; percentile CI. sklearn is used (not the repo's
compute_binary_metrics) to match the original script exactly.

Run: /opt/homebrew/Caskroom/miniforge/base/envs/eda/bin/python <this file>
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import from_bounds
from sklearn.metrics import (accuracy_score, f1_score, jaccard_score,
                             matthews_corrcoef, precision_score, recall_score)

_SCRIPTS = Path(__file__).resolve().parent.parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

NAS = Path("/Volumes/WD_8tb_RedPlus_NAS_A/MACKBOOK_AIR_M2_BACKUP/Documents/EOS/ACDC")
RUNS = Path("experiments/hampyeong/runs")
OUT = Path("experiments/hampyeong/evaluation/deprecated_estimator_comparison.csv")

SAMPLE_SIZE = 400        # original: sample_size=400
N_BOOTSTRAPS = 10000     # original: num_bootstraps=10000 (line 189 overrides the def default)
DATES = ["20210305", "20210422", "20210621"]   # only dates the legacy runs cover
SEEDS = (19, 42, 58)
MODELS = ["legacy_baseline", "legacy_finetuned", "Swin-B", "ConvNeXtV2"]
METRICS = ["oa", "f1", "precision", "recall", "iou", "mcc"]

SCENE = {
    "20210305": "S1B_IW_GRDH_1SDV_20210305T213224_20210305T213249_025885_031658_6933_Clipped",
    "20210422": "S1B_IW_GRDH_1SDV_20210422T213225_20210422T213250_026585_032CBA_CF3C_Clipped",
    "20210621": "S1B_IW_GRDH_1SDV_20210621T213228_20210621T213253_027460_034782_6C41_Clipped",
}
RUN_DIR = {
    ("Swin-B", 19): "dev_sweep_all_hampyeong_s19_20260710T100323Z_upernet_swin_base_224_native224_weighted_224_b0_s32",
    ("Swin-B", 42): "dev_sweep_all_hampyeong_s42_20260716T063710Z_upernet_swin_base_224_native224_weighted_224_b0_s32",
    ("Swin-B", 58): "dev_sweep_all_hampyeong_s58_20260710T101618Z_upernet_swin_base_224_native224_weighted_224_b0_s32",
    ("ConvNeXtV2", 19): "dev_sweep_all_hampyeong_s19_20260710T100323Z_upernet_tu-convnextv2_base_native224_weighted_224_b0_s32",
    ("ConvNeXtV2", 42): "dev_sweep_all_hampyeong_s42_20260710T100817Z_upernet_tu-convnextv2_base_native224_weighted_224_b0_s32",
    ("ConvNeXtV2", 58): "dev_sweep_all_hampyeong_s58_20260710T101618Z_upernet_tu-convnextv2_base_native224_weighted_224_b0_s32",
}
LEGACY_SUBDIR = {"legacy_baseline": "Pretrained_Baseline_Resnet50_for_reproj",
                 "legacy_finetuned": "Baseline_Resnet50_reproj_FINETUNED"}


def _metrics(y_true, y_pred) -> dict:
    """Full-population point metrics via sklearn (used once per model/date)."""
    return {"oa": accuracy_score(y_true, y_pred),
            "f1": f1_score(y_true, y_pred, zero_division=0),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "iou": jaccard_score(y_true, y_pred, zero_division=0),
            "mcc": matthews_corrcoef(y_true, y_pred)}


def _metrics_from_counts(tp, fp, fn, tn):
    """Vectorised metrics from confusion counts (arrays over bootstrap iterations).

    Analytically identical to the sklearn calls in _metrics, computed from the four
    count cells so 10,000 iterations cost four bincounts instead of 60,000 sklearn calls.
    zero_division=0 semantics matched: undefined ratios -> 0; MCC denom 0 -> 0.
    """
    tp, fp, fn, tn = (x.astype(np.float64) for x in (tp, fp, fn, tn))
    total = tp + fp + fn + tn
    with np.errstate(invalid="ignore", divide="ignore"):
        precision = np.where(tp + fp > 0, tp / (tp + fp), 0.0)
        recall = np.where(tp + fn > 0, tp / (tp + fn), 0.0)
        f1 = np.where(2 * tp + fp + fn > 0, 2 * tp / (2 * tp + fp + fn), 0.0)
        iou = np.where(tp + fp + fn > 0, tp / (tp + fp + fn), 0.0)
        oa = (tp + tn) / total
        mcc_den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        mcc = np.where(mcc_den > 0, (tp * tn - fp * fn) / mcc_den, 0.0)
    return {"oa": oa, "f1": f1, "precision": precision, "recall": recall, "iou": iou, "mcc": mcc}


def main() -> None:
    if not NAS.exists():
        raise SystemExit(f"NAS root not found: {NAS} — is the external volume mounted?")

    with rasterio.open(NAS / "Tide_Gauge/Korean_Peninsula/DEM_wrt_WGS84_TBM/DEM_VALID_MASK_aoi.tif") as m:
        valid = m.read(1) == 1
        bounds = m.bounds
    n_valid = int(valid.sum())

    def new_pred(arch, date):        # 3-seed-mean probability, thresholded at 0.5
        acc = np.zeros(n_valid)
        for seed in SEEDS:
            with rasterio.open(RUNS / RUN_DIR[(arch, seed)] / SCENE[date] / f"{SCENE[date]}_probability_water.tif") as src:
                w = from_bounds(*bounds, transform=src.transform).round_offsets().round_lengths()
                acc += src.read(1, window=w)[valid]
        return ((acc / len(SEEDS)) >= 0.5).astype(np.uint8)

    def old_pred(key, date):
        p = NAS / "sen12coast/Inference/Hampeyeong/Models" / LEGACY_SUBDIR[key] / "DESC/Geotiffs_clipped_Val_AOI" / f"Inference_DESC_{date}_Clipped_VAL_AOI.tif"
        return (rasterio.open(p).read(1)[valid] > 0).astype(np.uint8)

    def gt(date):
        p = NAS / "sen12coast/Validation/Hampyeong/DEM_FLOOD_MASKS/Descending_reproj" / f"DEM_FLOOD_S1_DESC_{date}_VAL_AOI.tif"
        return (rasterio.open(p).read(1)[valid] > 0).astype(np.uint8)

    rows = []
    for date in DATES:
        y_true = gt(date)
        preds = {"legacy_baseline": old_pred("legacy_baseline", date),
                 "legacy_finetuned": old_pred("legacy_finetuned", date),
                 "Swin-B": new_pred("Swin-B", date),
                 "ConvNeXtV2": new_pred("ConvNeXtV2", date)}
        full = {m: _metrics(y_true, preds[m]) for m in MODELS}

        # Per-pixel confusion category as one int code per pixel per model:
        # 0=tn, 1=fp, 2=fn, 3=tp. Bincount over the sampled codes gives all 4 counts.
        ytb = y_true.astype(bool)
        code = {m: (ytb.astype(np.int8) * 2 + preds[m].astype(bool).astype(np.int8)).astype(np.int8)
                for m in MODELS}
        # code = 2*y + p  ->  y=0,p=0:0(tn)  y=0,p=1:1(fp)  y=1,p=0:2(fn)  y=1,p=1:3(tp)

        # Faithful seeded draws, but gather counts for all iterations, then vectorise metrics.
        counts = {m: np.empty((N_BOOTSTRAPS, 4), dtype=np.int32) for m in MODELS}  # [tn,fp,fn,tp]
        for i in range(N_BOOTSTRAPS):
            np.random.seed(i)                                    # faithful: reset per iteration
            idx = np.random.choice(n_valid, SAMPLE_SIZE, replace=True)
            for m in MODELS:
                counts[m][i] = np.bincount(code[m][idx], minlength=4)

        boot = {}
        for m in MODELS:
            tn, fp, fn, tp = (counts[m][:, j] for j in range(4))
            boot[m] = _metrics_from_counts(tp, fp, fn, tn)

        wf = float(y_true.mean())
        for m in MODELS:
            for k in METRICS:
                b = boot[m][k]
                rows.append({"date": date, "water_frac": wf, "model": m, "metric": k,
                             "boot_mean": float(b.mean()), "boot_std": float(b.std()),
                             "ci_lo": float(np.percentile(b, 2.5)), "ci_hi": float(np.percentile(b, 97.5)),
                             "full_pop": float(full[m][k]),
                             "sample_size": SAMPLE_SIZE, "n_bootstraps": N_BOOTSTRAPS})
        print(f"{date} done (wf {wf:.3f})")

    df = pd.DataFrame(rows)
    # Faithfulness: the bootstrap mean must track the full-population value. MCC is
    # small-sample biased under 400-pixel subsampling on the 94%-water date (few true
    # negatives), so it drifts ~0.013 there -- a real property of this estimator, not an
    # error. All other metrics must stay tight.
    drift = (df.boot_mean - df.full_pop).abs()
    non_mcc = drift[df.metric != "mcc"].max()
    mcc = drift[df.metric == "mcc"].max()
    assert non_mcc < 0.002, f"non-MCC bootstrap mean drifted {non_mcc:.4f} — estimator not faithful"
    assert mcc < 0.02, f"MCC drifted {mcc:.4f} — larger than expected small-sample bias"
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"\nfaithful: non-MCC drift {non_mcc:.4f}, MCC drift {mcc:.4f} (small-sample bias)\nWrote {OUT}")


if __name__ == "__main__":
    main()
