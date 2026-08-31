"""Score the hamp24 campaign — 21 arms x 24 dates, extent-parameterized.

A thin generalization of score_pairbased_hampyeong.py (imported, not forked:
guards, audit, loaders and metric spine are reused verbatim) for the 24-scene
Hampyeong campaign scored at three valid-mask extents. What the original
hardcodes, the spec now carries:

    dates_csv:    CSV with scene_id + date mapping (pairing_metadata.csv);
                  dates/scene stems are derived from it (24 rows).
    gt_dir:       directory holding DEM_FLOOD_S1_DESC_<date>_VAL_AOI.tif for
                  every date (one of the three extent sets).
    valid_mask:   abs path to the valid-mask raster for this extent.
    expected_valid_pixels, entries, output: as in the original spec schema.

Guards preserved: threshold 0.5 greater_equal, resolution_atol 1e-6, expected
valid-pixel count per load, identical y_true across entries per date,
prediction-digest distinctness per date, full provenance audit (config ==
summary == manifest checkpoint, stride 32, threshold 0.5, per-entry expected
checkpoint, no shared checkpoints), refuse-to-clobber output.

Run:
    python scripts/evaluation/vm/score_hamp24.py --spec <spec.yaml>
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from score_pairbased_hampyeong import (  # noqa: E402
    ProvenanceError,
    audit,
    require,
    _tag,
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--spec", required=True, type=Path)
    args = ap.parse_args()

    spec = yaml.safe_load(args.spec.read_text())
    for key in ("dates_csv", "gt_dir", "valid_mask", "expected_valid_pixels",
                "output", "entries"):
        require(key in spec, f"spec must set {key}")
    spec.setdefault("repo", "/home/noel/segwater_v2")
    spec.setdefault("pair_root", "outputs/inference")
    spec.setdefault("variant_column", False)
    spec.setdefault("lineage_slug", "pairbased")
    spec.setdefault("training_data", "pair-based")

    repo = Path(spec["repo"])
    sys.path.insert(0, str(repo / "scripts"))
    from evaluation.metrics import METRIC_NAMES, compute_binary_metrics  # noqa: E402
    from inference_overlap_utils import load_overlap_reference_and_prediction  # noqa: E402

    pair_root = repo / spec["pair_root"]
    gt_dir = Path(spec["gt_dir"])
    mask = Path(spec["valid_mask"])
    expected_valid = int(spec["expected_valid_pixels"])
    dest = Path(spec["output"])
    require(not dest.exists(), f"refusing to clobber existing {dest}")
    require(mask.exists(), f"valid mask not found: {mask}")

    pairing = pd.read_csv(spec["dates_csv"])
    require(len(pairing) == 24, f"expected 24 rows in {spec['dates_csv']}, got {len(pairing)}")
    scenes = {}  # date -> scene stem (with _Clipped, as the run dirs name them)
    for sid in pairing["scene_id"]:
        date = str(sid)[17:25]
        require(date not in scenes, f"duplicate date {date} in dates_csv")
        scenes[date] = f"{sid}_Clipped"
    dates = sorted(scenes)

    entries = audit(spec, repo, pair_root)

    rows = []
    for date in dates:
        ref = gt_dir / f"DEM_FLOOD_S1_DESC_{date}_VAL_AOI.tif"
        require(ref.exists(), f"missing GT {ref}")
        scene = scenes[date]
        digests: dict[str, str] = {}
        y_true_reference = None
        for e in entries:
            tag = _tag(e, spec["variant_column"])
            pred = pair_root / e["resolved_run_dir"] / scene / f"{scene}_probability_water.tif"
            require(pred.exists(), f"missing prediction {pred}")
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
            require(n_valid == expected_valid,
                    f"{date} {tag}: valid px {n_valid} != {expected_valid}")
            if y_true_reference is None:
                y_true_reference = y_true
            else:
                require(np.array_equal(y_true, y_true_reference),
                        f"{date}: y_true differs for {tag}")
            d = hashlib.sha1(y_pred.tobytes()).hexdigest()
            if d in digests:
                raise ProvenanceError(
                    f"{date}: {tag} identical prediction to {digests[d]} -- miswiring")
            digests[d] = tag
            m = compute_binary_metrics(y_true, y_pred, include_counts=True)
            rows.append({
                "date": int(date),
                "model": f"{e['slug']}_s{e['seed']}_{spec['lineage_slug']}",
                "arch": e["label"],
                "training_data": spec["training_data"],
                "seed": e["seed"],
                **{k: m[k] for k in METRIC_NAMES},
                **{c: m[c] for c in ("tn", "fp", "fn", "tp")},
                "n_valid": n_valid,
                "water_fraction_gt": float(y_true.mean()),
            })
        print(f"  {date}: {len(entries)} arms scored (water frac {float(y_true_reference.mean()):.4f})")

    out = pd.DataFrame(rows)
    out.to_csv(dest, index=False)
    print(f"\nwrote {dest} ({len(out)} rows)")


if __name__ == "__main__":
    main()
