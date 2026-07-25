"""Score the two LEGACY Sen1Coast ResNet50-UNet Hampyeong predictions.

Separate job from the pair-based scorer: it scores the pre-computed legacy
rasters (Pretrained_Baseline / FINETUNED) under the NAS mirror and asserts they
still reproduce the deprecated-manuscript Table-2 point estimates (tol 0.0015).
Same loader / metrics / guards as score_pairbased_hampyeong.py (threshold 0.5,
greater_equal, resolution_atol 1e-6, valid-pixel count 1,179,967).

Rescued from ~/score_hampyeong_legacy.py — the only change is that the repo,
NAS-root and output paths are now flags instead of VM-hardcoded constants, so it
runs anywhere the legacy rasters are mirrored.

    python scripts/evaluation/vm/score_hampyeong_legacy.py \
        --repo /home/noel/segwater_v2 \
        --nas-root /home/noel/ancillary/hampyeong/nas_root \
        --output /home/noel/hampyeong_vm_legacy_metrics.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

DATES = ["20210305", "20210422", "20210621"]
EXPECTED_VALID_PIXELS = 1_179_967
REPRO_TOLERANCE = 0.0015

LEGACY = [
    ("legacy_baseline", "Pretrained_Baseline_Resnet50_for_reproj"),
    ("legacy_finetuned", "Baseline_Resnet50_reproj_FINETUNED"),
]

# Deprecated-manuscript Table 2 (Hampyeong Bay) point estimates, from
# scripts/evaluation/hampyeong_model_comparison.py PUBLISHED.
PUBLISHED = {
    ("20210305", "legacy_baseline"): dict(oa=0.947, recall=0.997, precision=0.949, f1=0.973, iou=0.947),
    ("20210305", "legacy_finetuned"): dict(oa=0.941, recall=0.987, precision=0.952, f1=0.969, iou=0.941),
    ("20210422", "legacy_baseline"): dict(oa=0.854, recall=0.943, precision=0.806, f1=0.869, iou=0.769),
    ("20210422", "legacy_finetuned"): dict(oa=0.909, recall=0.925, precision=0.901, f1=0.912, iou=0.839),
    ("20210621", "legacy_baseline"): dict(oa=0.881, recall=0.947, precision=0.833, f1=0.886, iou=0.796),
    ("20210621", "legacy_finetuned"): dict(oa=0.931, recall=0.929, precision=0.929, f1=0.929, iou=0.867),
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", type=Path, default=Path("/home/noel/segwater_v2"))
    ap.add_argument("--nas-root", type=Path, default=Path("/home/noel/ancillary/hampyeong/nas_root"))
    ap.add_argument("--output", type=Path, default=Path("/home/noel/hampyeong_vm_legacy_metrics.csv"))
    args = ap.parse_args()

    sys.path.insert(0, str(args.repo / "scripts"))
    from evaluation.metrics import METRIC_NAMES, compute_binary_metrics  # noqa: E402
    from inference_overlap_utils import load_overlap_reference_and_prediction  # noqa: E402

    nas = args.nas_root
    mask = nas / "Tide_Gauge/Korean_Peninsula/DEM_wrt_WGS84_TBM/DEM_VALID_MASK_aoi.tif"
    rows = []
    for date in DATES:
        ref = nas / "sen12coast/Validation/Hampyeong/DEM_FLOOD_MASKS/Descending_reproj" / f"DEM_FLOOD_S1_DESC_{date}_VAL_AOI.tif"
        for model, subdir in LEGACY:
            pred = nas / "sen12coast/Inference/Hampeyeong/Models" / subdir / "DESC/Geotiffs_clipped_Val_AOI" / f"Inference_DESC_{date}_Clipped_VAL_AOI.tif"
            assert pred.exists(), pred
            y_true, y_pred, diag = load_overlap_reference_and_prediction(
                reference_path=str(ref),
                prediction_path=str(pred),
                reference_water_values=[1],
                reference_nodata_values=None,
                probability_threshold=0.5,
                probability_comparison="greater_equal",
                resolution_atol=1e-6,
                valid_mask_path=str(mask),
                valid_mask_value=1,
            )
            n_valid = diag["valid_pixels_after_all_masks"]
            assert n_valid == EXPECTED_VALID_PIXELS, f"{date} {model}: valid px {n_valid}"
            m = compute_binary_metrics(y_true, y_pred, include_counts=True)
            pub = PUBLISHED[(date, model)]
            for k, v in pub.items():
                d = abs(m[k] - v)
                assert d <= REPRO_TOLERANCE, f"{date} {model} {k}: computed {m[k]:.4f} vs published {v} (|d|={d:.4f})"
            rows.append({"date": date, "model": model,
                         **{k: m[k] for k in METRIC_NAMES},
                         **{c: m[c] for c in ("tn", "fp", "fn", "tp")},
                         "n_valid": n_valid})
            print(f"{date} {model:17s} IoU={m['iou']:.4f} F1={m['f1']:.4f} (published repro OK)")
    pd.DataFrame(rows).to_csv(args.output, index=False)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
