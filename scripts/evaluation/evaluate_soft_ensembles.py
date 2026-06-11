"""
Soft-ensemble evaluation on probability rasters.

Tests whether averaging member probabilities beats (a) single models and
(b) hard majority voting, on the 6 reference scenes of the Semarang benchmark,
with the external GSHHG valid mask applied.

For every candidate member set (all subsets within each tiling group, all
pairs across the 12 members, and ALL-12) and every single model the script
reports, pooled over scenes:

  - IoU/P/R at the DEFAULT threshold 0.5 (greater_than), computed exactly
  - IoU at a leave-one-scene-out tuned threshold (tune on 5, eval on 1)
  - AUC-PR of the (ensemble) probability, threshold-free

plus weighted-mean pairs and a per-pixel logistic-regression stacker
(LOO by scene), and the oracle upper bound at threshold 0.5.

Sanity gate: each member thresholded at 0.5 must reproduce the pooled IoU of
the masked confusion-raster analysis before any ensemble numbers are emitted.

Usage
-----
    python scripts/evaluation/evaluate_soft_ensembles.py \
        path/to/evaluation_manifest.csv \
        --valid-mask-path mask.tif \
        --sanity-csv path/to/<masked>__single_model_pooled_metrics.csv ... \
        --output-dir path
"""

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import from_bounds as window_from_bounds

sys.path.insert(0, str(Path(__file__).parent))
from analyze_error_decorrelation import DIR_RENAMES, MECHANISM, TILING_SUFFIXES, short_name

N_BINS = 1000
EDGES = np.linspace(0.0, 1.0, N_BINS + 1)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_scene(
    reference_path: str,
    member_paths: dict[str, str],
    valid_mask_path: str | None,
    valid_mask_value: int,
    reference_water_values=(1,),
    reference_nodata_values=(255,),
) -> tuple[np.ndarray, np.ndarray]:
    """Returns (ref_binary[v], probs[n_members, v]) at valid pixels."""
    with rasterio.open(reference_path) as src:
        reference = src.read(1)
        bounds, transform = src.bounds, src.transform

    valid = ~np.isin(reference, reference_nodata_values)
    if valid_mask_path:
        with rasterio.open(valid_mask_path) as src:
            win = window_from_bounds(*bounds, transform=src.transform)
            win = win.round_offsets().round_lengths()
            mask = src.read(1, window=win, boundless=True, fill_value=0)
        if mask.shape != reference.shape:
            raise ValueError(f"mask window {mask.shape} != ref {reference.shape}")
        valid &= mask == valid_mask_value

    ref_binary = np.isin(reference, reference_water_values)[valid]

    probs = np.empty((len(member_paths), int(valid.sum())), dtype=np.float32)
    for i, (member, path) in enumerate(member_paths.items()):
        with rasterio.open(path) as src:
            win = window_from_bounds(*bounds, transform=src.transform)
            win = win.round_offsets().round_lengths()
            arr = src.read(1, window=win)
        if arr.shape != reference.shape:
            raise ValueError(
                f"{member}: prob window {arr.shape} != ref {reference.shape}"
            )
        probs[i] = arr[valid]
    return ref_binary, probs


# ---------------------------------------------------------------------------
# Metric machinery
# ---------------------------------------------------------------------------

def iou_pr(tp, fp, fn):
    union = tp + fp + fn
    return (
        tp / union if union else float("nan"),
        tp / (tp + fp) if tp + fp else 0.0,
        tp / (tp + fn) if tp + fn else 0.0,
    )


def counts_at_05(score: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Exact (tp, fp, fn) for prediction = score > 0.5."""
    pred = score > 0.5
    return np.array(
        [int(np.sum(pred & ref)), int(np.sum(pred & ~ref)), int(np.sum(~pred & ref))],
        dtype=np.int64,
    )


def score_curves(score: np.ndarray, ref: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Suffix-sum histograms: (tp_curve[N_BINS], fp_curve[N_BINS]) where index i
    approximates counts for prediction = score > EDGES[i]."""
    hw, _ = np.histogram(score[ref], bins=EDGES)
    hn, _ = np.histogram(score[~ref], bins=EDGES)
    tp = hw[::-1].cumsum()[::-1]
    fp = hn[::-1].cumsum()[::-1]
    return tp.astype(np.int64), fp.astype(np.int64)


def loo_tuned_iou(per_scene_tp, per_scene_fp, per_scene_nw):
    """per_scene_tp/fp: list of curves; per_scene_nw: list of total water counts.
    Returns (pooled held-out counts (tp,fp,fn), list of chosen thresholds)."""
    n = len(per_scene_tp)
    pooled = np.zeros(3, dtype=np.int64)
    chosen = []
    for held in range(n):
        tr = [i for i in range(n) if i != held]
        tp_c = sum(per_scene_tp[i] for i in tr)
        fp_c = sum(per_scene_fp[i] for i in tr)
        w = sum(per_scene_nw[i] for i in tr)
        fn_c = w - tp_c
        with np.errstate(divide="ignore", invalid="ignore"):
            iou = tp_c / np.maximum(tp_c + fp_c + fn_c, 1)
        best = int(np.argmax(iou))
        chosen.append(float(EDGES[best]))
        h_tp = int(per_scene_tp[held][best])
        h_fp = int(per_scene_fp[held][best])
        h_fn = int(per_scene_nw[held] - h_tp)
        pooled += (h_tp, h_fp, h_fn)
    return pooled, chosen


def loo_tuned_macro(per_scene_tp, per_scene_fp, per_scene_nw):
    """LOO threshold tuning with a MACRO objective: pick the threshold that
    maximizes the mean of the train scenes' per-scene IoU curves, evaluate the
    held-out scene's IoU at it. Returns (macro_iou, list of thresholds)."""
    n = len(per_scene_tp)
    with np.errstate(divide="ignore", invalid="ignore"):
        iou_curves = [
            per_scene_tp[i]
            / np.maximum(per_scene_tp[i] + per_scene_fp[i]
                         + (per_scene_nw[i] - per_scene_tp[i]), 1)
            for i in range(n)
        ]
    held_ious, chosen = [], []
    for held in range(n):
        train_mean = np.mean([iou_curves[i] for i in range(n) if i != held],
                             axis=0)
        best = int(np.argmax(train_mean))
        chosen.append(float(EDGES[best]))
        held_ious.append(float(iou_curves[held][best]))
    return float(np.mean(held_ious)), chosen


def auc_pr(tp_curve: np.ndarray, fp_curve: np.ndarray, total_w: int) -> float:
    with np.errstate(divide="ignore", invalid="ignore"):
        precision = tp_curve / np.maximum(tp_curve + fp_curve, 1)
        recall = tp_curve / max(total_w, 1)
    order = np.argsort(recall)
    return float(np.trapezoid(precision[order], recall[order]))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("manifest_csv")
    p.add_argument("--valid-mask-path", required=True)
    p.add_argument("--valid-mask-value", type=int, default=1)
    p.add_argument("--sanity-csv", nargs="+", default=[],
                   help="masked single_model_pooled_metrics.csv files to check against")
    p.add_argument("--sanity-atol", type=float, default=2e-3)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--stacker-samples-per-scene", type=int, default=300_000)
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.manifest_csv)
    df = df[df["status"] == "success"].copy()
    df["model_name"] = df["model_name"].replace(DIR_RENAMES)
    members = sorted(df["model_name"].unique())
    scenes = sorted(df["s1_id"].unique())
    midx = {m: i for i, m in enumerate(members)}
    print(f"{len(members)} members, {len(scenes)} scenes")

    def tiling_of(m):
        return next(t for t, s in TILING_SUFFIXES.items() if m.endswith(s))

    # ---- load all scenes
    scene_ref, scene_probs = [], []
    for scene in scenes:
        sub = df[df["s1_id"] == scene].set_index("model_name")
        paths = {m: sub.loc[m, "prediction_path"] for m in members}
        ref_path = sub["reference_path"].iloc[0]
        ref, probs = load_scene(
            ref_path, paths, args.valid_mask_path, args.valid_mask_value
        )
        scene_ref.append(ref)
        scene_probs.append(probs)
        print(f"  loaded {scene}: {ref.size} valid px")

    total_w = [int(r.sum()) for r in scene_ref]

    # ---- sanity gate: members @0.5 vs confusion-raster analysis
    member_05 = {}
    for m in members:
        c = sum(counts_at_05(scene_probs[s][midx[m]], scene_ref[s])
                for s in range(len(scenes)))
        member_05[m] = c
    if args.sanity_csv:
        expected = pd.concat([pd.read_csv(f) for f in args.sanity_csv])
        failures = []
        for m in members:
            iou, _, _ = iou_pr(*member_05[m])
            exp = expected[expected["model"] == short_name(m)]
            # tiling groups share base names; match on IoU closeness to either
            diffs = (exp["iou"] - iou).abs()
            if diffs.min() > args.sanity_atol:
                failures.append((m, iou, exp["iou"].tolist()))
        if failures:
            for f in failures:
                print(f"SANITY FAIL: {f}", file=sys.stderr)
            sys.exit(1)
        print("Sanity gate passed: member IoUs @0.5 match confusion-raster analysis")

    # ---- candidate member sets
    candidates: dict[str, list[int]] = {}
    for tiling in TILING_SUFFIXES:
        group = [m for m in members if tiling_of(m) == tiling]
        for k in range(1, len(group) + 1):
            for combo in itertools.combinations(group, k):
                candidates[",".join(combo)] = [midx[m] for m in combo]
    for a, b in itertools.combinations(members, 2):  # adds cross-tiling pairs
        candidates.setdefault(f"{a},{b}", [midx[a], midx[b]])
    candidates["ALL_12"] = list(range(len(members)))

    # ---- evaluate all candidates (soft mean)
    rows = []
    per_scene_single_rows = []
    for name, idxs in candidates.items():
        c05 = np.zeros(3, dtype=np.int64)
        scene_iou05 = []
        tp_curves, fp_curves = [], []
        for s in range(len(scenes)):
            score = scene_probs[s][idxs].mean(axis=0) if len(idxs) > 1 \
                else scene_probs[s][idxs[0]]
            sc = counts_at_05(score, scene_ref[s])
            c05 += sc
            scene_iou05.append(iou_pr(*sc)[0])
            tpc, fpc = score_curves(score, scene_ref[s])
            tp_curves.append(tpc)
            fp_curves.append(fpc)
            if len(idxs) == 1:
                per_scene_single_rows.append(
                    {"member": name, "scene": scenes[s], "iou_05": iou_pr(*sc)[0],
                     "precision_05": iou_pr(*sc)[1], "recall_05": iou_pr(*sc)[2]}
                )
        iou05, p05, r05 = iou_pr(*c05)
        loo_counts, thresholds = loo_tuned_iou(tp_curves, fp_curves, total_w)
        iou_loo, p_loo, r_loo = iou_pr(*loo_counts)
        iou_loo_macro, thresholds_macro = loo_tuned_macro(
            tp_curves, fp_curves, total_w
        )
        pooled_tp = sum(tp_curves)
        pooled_fp = sum(fp_curves)
        names = name.split(",") if name != "ALL_12" else members
        mechs = sorted({MECHANISM[short_name(x)] for x in names})
        tilings = sorted({tiling_of(x) for x in names})
        rows.append({
            "members": name,
            "k": len(idxs),
            "composition": "+".join(mechs),
            "tilings": "+".join(tilings),
            "iou_05": iou05, "precision_05": p05, "recall_05": r05,
            "iou_05_macro": float(np.mean(scene_iou05)),
            "iou_loo_tuned": iou_loo, "precision_loo": p_loo, "recall_loo": r_loo,
            "iou_loo_tuned_macro": iou_loo_macro,
            "mean_tuned_threshold": float(np.mean(thresholds)),
            "mean_tuned_threshold_macro": float(np.mean(thresholds_macro)),
            "auc_pr": auc_pr(pooled_tp, pooled_fp, sum(total_w)),
        })
    res = pd.DataFrame(rows).sort_values("iou_loo_tuned", ascending=False)
    res.to_csv(out_dir / "soft_ensemble_results.csv", index=False)
    pd.DataFrame(per_scene_single_rows).to_csv(
        out_dir / "single_model_per_scene_metrics_05.csv", index=False
    )

    # ---- weighted pairs: top-5 pairs by iou_loo_tuned
    top_pairs = res[(res["k"] == 2)].head(5)["members"].tolist()
    wrows = []
    weights = np.arange(0.1, 0.91, 0.1)
    for name in top_pairs:
        ia, ib = [midx[m] for m in name.split(",")]
        # per-weight curves per scene
        curves = {}
        for w in weights:
            tps, fps, c05s = [], [], np.zeros(3, dtype=np.int64)
            for s in range(len(scenes)):
                score = w * scene_probs[s][ia] + (1 - w) * scene_probs[s][ib]
                tpc, fpc = score_curves(score, scene_ref[s])
                tps.append(tpc); fps.append(fpc)
                c05s += counts_at_05(score, scene_ref[s])
            curves[round(w, 1)] = (tps, fps, c05s)
        # LOO over (w, t) jointly
        pooled = np.zeros(3, dtype=np.int64)
        for held in range(len(scenes)):
            tr = [i for i in range(len(scenes)) if i != held]
            best = (-1.0, None, None)
            for w, (tps, fps, _) in curves.items():
                tp_c = sum(tps[i] for i in tr)
                fp_c = sum(fps[i] for i in tr)
                wtr = sum(total_w[i] for i in tr)
                with np.errstate(divide="ignore", invalid="ignore"):
                    iou = tp_c / np.maximum(tp_c + fp_c + (wtr - tp_c), 1)
                bi = int(np.argmax(iou))
                if iou[bi] > best[0]:
                    best = (float(iou[bi]), w, bi)
            _, w, bi = best
            tps, fps, _ = curves[w]
            h_tp = int(tps[held][bi]); h_fp = int(fps[held][bi])
            pooled += (h_tp, h_fp, total_w[held] - h_tp)
        iou_loo, p_loo, r_loo = iou_pr(*pooled)
        wrows.append({"members": name, "iou_loo_tuned_weighted": iou_loo,
                      "precision": p_loo, "recall": r_loo})
    pd.DataFrame(wrows).to_csv(out_dir / "weighted_pairs.csv", index=False)

    # ---- stacker: logistic regression, LOO by scene
    from sklearn.linear_model import LogisticRegression
    rng = np.random.default_rng(42)
    pooled05 = np.zeros(3, dtype=np.int64)
    pooled_tuned = np.zeros(3, dtype=np.int64)
    for held in range(len(scenes)):
        tr = [i for i in range(len(scenes)) if i != held]
        Xs, ys = [], []
        for s in tr:
            n = scene_ref[s].size
            sel = rng.choice(n, size=min(args.stacker_samples_per_scene, n),
                             replace=False)
            Xs.append(scene_probs[s][:, sel].T)
            ys.append(scene_ref[s][sel])
        clf = LogisticRegression(max_iter=1000)
        clf.fit(np.vstack(Xs), np.concatenate(ys))
        # tune threshold on train subsample
        tr_score = clf.predict_proba(np.vstack(Xs))[:, 1].astype(np.float32)
        tr_ref = np.concatenate(ys)
        tpc, fpc = score_curves(tr_score, tr_ref)
        w = int(tr_ref.sum())
        with np.errstate(divide="ignore", invalid="ignore"):
            iou = tpc / np.maximum(tpc + fpc + (w - tpc), 1)
        bi = int(np.argmax(iou))
        score = clf.predict_proba(scene_probs[held].T)[:, 1].astype(np.float32)
        pooled05 += counts_at_05(score, scene_ref[held])
        h_tpc, h_fpc = score_curves(score, scene_ref[held])
        h_tp = int(h_tpc[bi]); h_fp = int(h_fpc[bi])
        pooled_tuned += (h_tp, h_fp, total_w[held] - h_tp)
    stack = pd.DataFrame([
        {"combiner": "stacker_logreg@0.5", **dict(zip(["iou", "precision", "recall"],
                                                      iou_pr(*pooled05)))},
        {"combiner": "stacker_logreg@loo_tuned", **dict(zip(["iou", "precision",
                                                             "recall"],
                                                            iou_pr(*pooled_tuned)))},
    ])
    stack.to_csv(out_dir / "stacker_results.csv", index=False)

    # ---- oracle @0.5 (any member correct), per tiling group and all-12
    orows = []
    for label, group in [("native224", [m for m in members
                                        if tiling_of(m) == "native224"]),
                         ("large_crop_1024", [m for m in members
                                              if tiling_of(m) == "large_crop_1024"]),
                         ("ALL_12", members)]:
        idxs = [midx[m] for m in group]
        tp = fp = fn = 0
        for s in range(len(scenes)):
            preds = scene_probs[s][idxs] > 0.5
            ref = scene_ref[s]
            correct = (preds == ref[None]).any(axis=0)
            tp += int(np.sum(correct & ref))
            fp += int(np.sum(~correct & ~ref))
            fn += int(np.sum(~correct & ref))
        iou, pr, rc = iou_pr(tp, fp, fn)
        orows.append({"group": label, "iou": iou, "precision": pr, "recall": rc})
    pd.DataFrame(orows).to_csv(out_dir / "oracle_05.csv", index=False)

    # ---- console summary
    print("\nTop 12 by LOO-tuned IoU (micro):")
    print(res.head(12)[["members", "k", "composition", "tilings", "iou_05",
                        "iou_loo_tuned", "iou_loo_tuned_macro",
                        "mean_tuned_threshold", "auc_pr"]]
          .to_string(index=False))
    print("\nTop 12 by IoU at default 0.5 (micro):")
    print(res.sort_values("iou_05", ascending=False).head(12)
          [["members", "k", "composition", "iou_05", "iou_05_macro",
            "precision_05", "recall_05"]]
          .to_string(index=False))
    print("\nSingles (k=1):")
    print(res[res["k"] == 1].sort_values("iou_05", ascending=False)
          [["members", "iou_05", "iou_05_macro", "iou_loo_tuned",
            "iou_loo_tuned_macro", "mean_tuned_threshold", "auc_pr"]]
          .to_string(index=False))
    print("\nWeighted pairs:")
    print(pd.DataFrame(wrows).to_string(index=False))
    print("\nStacker:")
    print(stack.to_string(index=False))
    print("\nOracle @0.5:")
    print(pd.DataFrame(orows).to_string(index=False))
    print(f"\nOutputs in {out_dir}")


if __name__ == "__main__":
    main()
