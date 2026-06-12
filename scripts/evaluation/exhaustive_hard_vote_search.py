"""
Exhaustively evaluate every majority-vote ensemble subset (size >= 2) per
tiling group, using the confusion rasters and the same external valid mask
as analyze_error_decorrelation.py. Tags each subset with its mechanism
composition (CNN / Transformer / Hybrid) to answer:

  - does mixing CNN + Hybrid + Transformer ever beat pure subsets?
  - does adding CNN members help or hurt transformer/hybrid ensembles?

Writes <group>__exhaustive_ensembles.csv (one row per subset, sorted by IoU)
into the output dir.

Usage
-----
    python scripts/evaluation/exhaustive_ensemble_search.py \
        path/to/confusion_rasters --output-dir path \
        [--valid-mask-path mask.tif --valid-mask-value 1]
"""

import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd

import sys

sys.path.insert(0, str(Path(__file__).parent))
from analyze_error_decorrelation import (
    CM_FN, CM_FP, CM_NODATA, CM_TP, MECHANISM, TILING_SUFFIXES,
    discover, load_scene_stack, metrics_from_counts,
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("confusion_rasters_dir")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--valid-mask-path", default=None)
    p.add_argument("--valid-mask-value", type=int, default=1)
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = discover(Path(args.confusion_rasters_dir))

    for tiling in TILING_SUFFIXES:
        sub = df[df["tiling"] == tiling]
        if sub.empty:
            continue
        scenes = sorted(sub["scene"].unique())
        models = sorted(sub["base"].unique())
        n = len(models)

        scene_data = []
        for scene in scenes:
            df_scene = sub[sub["scene"] == scene].sort_values("base")
            scene_models, cm = load_scene_stack(
                df_scene, args.valid_mask_path, args.valid_mask_value
            )
            assert scene_models == models
            valid = cm[0] != CM_NODATA
            pred = ((cm == CM_TP) | (cm == CM_FP)).astype(np.uint8)
            ref = ((cm[0] == CM_TP) | (cm[0] == CM_FN)).astype(np.uint8)
            scene_data.append((pred, ref, valid))

        rows = []
        for k in range(1, n + 1):
            for members in itertools.combinations(range(n), k):
                tp = fp = fn = 0
                scene_ious = []
                for pred, ref, valid in scene_data:
                    votes = pred[list(members)].sum(axis=0)
                    if k % 2 == 0:
                        ens = (2 * votes >= k)
                    else:
                        ens = (2 * votes > k)
                    s_tp = int(np.sum(valid & ens & (ref == 1)))
                    s_fp = int(np.sum(valid & ens & (ref == 0)))
                    s_fn = int(np.sum(valid & ~ens & (ref == 1)))
                    tp, fp, fn = tp + s_tp, fp + s_fp, fn + s_fn
                    scene_ious.append(metrics_from_counts(s_tp, s_fp, s_fn)["iou"])
                m = metrics_from_counts(tp, fp, fn)
                m["macro_iou"] = float(np.mean(scene_ious))
                names = [models[i] for i in members]
                mechs = sorted({MECHANISM[x] for x in names})
                comp = "+".join(mechs)
                rows.append(
                    {
                        "k": k,
                        "members": ",".join(names),
                        "composition": comp,
                        "n_cnn": sum(MECHANISM[x] == "CNN" for x in names),
                        "n_transformer": sum(
                            MECHANISM[x] == "Transformer" for x in names
                        ),
                        "n_hybrid": sum(MECHANISM[x] == "Hybrid" for x in names),
                        **{kk: m[kk] for kk in ["iou", "macro_iou", "precision",
                                                "recall"]},
                    }
                )
        res = pd.DataFrame(rows).sort_values("iou", ascending=False)
        out_csv = out_dir / f"{tiling}__exhaustive_ensembles.csv"
        res.to_csv(out_csv, index=False)

        print(f"\n=== {tiling}: {len(res)} subsets ===")
        print("Top 10 by IoU:")
        print(
            res.head(10)[["k", "members", "composition", "iou", "precision",
                          "recall"]].to_string(index=False)
        )
        print("\nBest per composition:")
        best_comp = res.loc[res.groupby("composition")["iou"].idxmax()]
        print(
            best_comp.sort_values("iou", ascending=False)[
                ["composition", "k", "members", "iou"]
            ].to_string(index=False)
        )
        print("\nBest per ensemble size:")
        best_k = res.loc[res.groupby("k")["iou"].idxmax()]
        print(
            best_k[["k", "members", "composition", "iou"]].to_string(index=False)
        )


if __name__ == "__main__":
    main()
