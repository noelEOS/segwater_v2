"""Per-scene AP + leave-one-scene-out tau* over the 23 scored Hampyeong scenes.

Named hamp23, NOT hamp24, deliberately. It reads the hamp24 campaign's 24-scene
inference tree but every reported number rests on 23 scenes: 20210329 is a
training-overlap scene and is excluded from all aggregates and from tau*
selection (see "The 20210329 exclusion" below). Artifacts carry hamp23 so that a
later reader cannot mistake them for 24-scene quantities and pool them with the
campaign's own hamp24_* outputs, which ARE 24-scene.

A sibling of score_hamp24.py (imported, not forked: the guards, audit and
loaders are reused verbatim) that answers a different question. score_hamp24.py
binarizes at 0.5 and reports metrics AT that operating point; this script sweeps
the probability grid so the threshold-FREE quantity (average precision) and the
threshold-CHOICE quantity (tau*) can both be measured.

Why this exists
---------------
The manuscript contrasts Demak with Hampyeong: at Demak, average precision
separates the architectures and the leave-one-scene-out optimal threshold is
architecture-dependent; at Hampyeong the published AP is flat. That published AP
was computed on 3-5 same-season 2021 scenes by
scripts/evaluation/hampyeong_pooled_ap_iou.py. This script recomputes it on the
24-scene campaign so both sites rest on comparable evidence.

Estimand -- deliberately matched to the Demak side
--------------------------------------------------
  - AP is computed PER SCENE, then macro-averaged over scenes (equal weight per
    scene), per (seed, architecture). NOT pixel-pooled across scenes.
    Rationale: pooling mixes a high-water scene (IoU ~0.99) with a low-water one
    (IoU ~0.65) into a single number and hides whether AP is flat at every tide
    state or only on average. Calibration is a per-scene property.
  - AP rule: step-interpolated (rectangular) integral of precision over recall,
    sum_k (R_k - R_{k+1}) * P_k, on the threshold grid, endpoints anchored so the
    curve is closed. This is the sklearn `average_precision_score` estimator
    restricted to the grid -- NOT the trapezoidal AUC of the PR curve. The two
    differ by up to ~0.15 on individual scenes, so the rule must travel with the
    number.
  - tau* rule: for each held-out scene, the IoU-maximizing threshold selected on
    ALL OTHER retained scenes (leave-one-scene-out), then averaged over scenes.
    NOTE an asymmetry with Demak that must be disclosed, not papered over: Demak
    holds out 1 of 6 scenes, this holds out 1 of 23. tau* here is selected on far
    more data and is correspondingly better determined. The two tau* values are
    comparable in construction but not identically conditioned.

The 20210329 exclusion
----------------------
Date 20210329 (PAIR_3442) overlaps the training set and is excluded from every
figure and quoted number. It is NOT encoded anywhere in the repo, so this script
carries it explicitly. The scene is still scored and written to the per-scene CSV
with excluded=True, so the exclusion is auditable rather than invisible, but it
is dropped BEFORE leave-one-scene-out selection and before every aggregate --
it contributes to neither the scoring nor the threshold choice.

Guards inherited from score_hamp24.py / score_pairbased_hampyeong.py: expected
valid-pixel count per load, identical y_true across entries within a date,
prediction-digest distinctness per date, full provenance audit (config ==
summary == manifest checkpoint, stride 32, per-entry expected checkpoint, no
shared checkpoints), refuse-to-clobber output.

Egress: this writes CSVs only. No rasters leave the VM.

Run:
    python scripts/evaluation/vm/score_hamp23_ap_tau.py \
        --spec scripts/evaluation/vm/specs/hamp24_full_bay.yaml \
        --out-prefix /home/noel/hamp23_ap_tau_full_bay

The --spec stays hamp24_*: it is the campaign's own inference spec, describing
the 24-scene tree this reads. The hamp23 in --out-prefix is what the outputs are.
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

# Training-overlap scene: excluded from all aggregates and from tau* selection.
EXCLUDED_DATES = {20210329}

# Probability grid. 101 points matches the Demak threshold_sweep.csv grid so the
# AP estimator sees the same resolution at both sites.
GRID = np.round(np.linspace(0.0, 1.0, 101), 2)


def sweep_counts(y_true: np.ndarray, prob: np.ndarray) -> pd.DataFrame:
    """Confusion counts at every grid threshold, via a single histogram pass.

    Rather than thresholding the full array 101 times, bin the probabilities once
    for water and land pixels separately; the suffix sums of those histograms give
    tp/fp at every threshold. Comparison is `greater_equal`, matching
    score_hamp24.py's operating-point convention.
    """
    edges = np.concatenate([GRID, [np.inf]])
    pos = np.histogram(prob[y_true == 1], bins=edges)[0]
    neg = np.histogram(prob[y_true == 0], bins=edges)[0]

    # tp[k] = #{water pixels with prob >= GRID[k]} = suffix sum of pos from k.
    tp = np.cumsum(pos[::-1])[::-1]
    fp = np.cumsum(neg[::-1])[::-1]
    n_pos = int((y_true == 1).sum())
    n_neg = int((y_true == 0).sum())
    fn = n_pos - tp
    tn = n_neg - fp

    with np.errstate(divide="ignore", invalid="ignore"):
        precision = np.where(tp + fp > 0, tp / (tp + fp), 1.0)
        recall = np.where(n_pos > 0, tp / max(n_pos, 1), 0.0)
        iou = np.where(tp + fp + fn > 0, tp / (tp + fp + fn), 0.0)

    return pd.DataFrame(
        {
            "threshold": GRID,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "precision": precision,
            "recall": recall,
            "iou": iou,
        }
    )


def average_precision(precision: np.ndarray, recall: np.ndarray) -> float:
    """Step-interpolated AP; see the module docstring for the exact rule."""
    order = np.argsort(-recall, kind="stable")
    r = np.concatenate([recall[order], [0.0]])
    p = np.concatenate([precision[order], [1.0]])
    return float(np.sum((r[:-1] - r[1:]) * p[:-1]))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--spec", required=True, type=Path)
    ap.add_argument("--out-prefix", required=True, type=Path)
    args = ap.parse_args()

    spec = yaml.safe_load(args.spec.read_text())
    for key in ("dates_csv", "gt_dir", "valid_mask", "expected_valid_pixels", "entries"):
        require(key in spec, f"spec must set {key}")
    spec.setdefault("repo", "/home/noel/segwater_v2")
    spec.setdefault("pair_root", "outputs/inference")
    spec.setdefault("variant_column", False)
    spec.setdefault("lineage_slug", "pairbased")
    spec.setdefault("training_data", "pair-based")

    sweep_dest = Path(f"{args.out_prefix}_per_scene_sweep.csv")
    scene_dest = Path(f"{args.out_prefix}_per_scene.csv")
    for d in (sweep_dest, scene_dest):
        require(not d.exists(), f"refusing to clobber existing {d}")

    repo = Path(spec["repo"])
    sys.path.insert(0, str(repo / "scripts"))
    from inference_overlap_utils import (  # noqa: E402
        load_overlap_reference_and_score,
        threshold_probability,
    )

    pair_root = repo / spec["pair_root"]
    gt_dir = Path(spec["gt_dir"])
    mask = Path(spec["valid_mask"])
    expected_valid = int(spec["expected_valid_pixels"])
    require(mask.exists(), f"valid mask not found: {mask}")

    pairing = pd.read_csv(spec["dates_csv"])
    require(len(pairing) == 24, f"expected 24 rows in {spec['dates_csv']}, got {len(pairing)}")
    scenes = {}
    for sid in pairing["scene_id"]:
        date = str(sid)[17:25]
        require(date not in scenes, f"duplicate date {date} in dates_csv")
        scenes[date] = f"{sid}_Clipped"
    dates = sorted(scenes)

    entries = audit(spec, repo, pair_root)

    sweep_rows = []
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

            # Raw probabilities, via the same _load_overlap_reference_and_probability
            # path score_hamp24.py uses -- so masking, overlap and pixel order are
            # identical to the campaign scorer. y_pred is reconstructed at 0.5 with
            # the same helper that scorer applies, purely for the digest guard.
            y_true, prob, diag = load_overlap_reference_and_score(
                reference_path=str(ref),
                prediction_path=str(pred),
                reference_water_values=[1],
                reference_nodata_values=None,
                resolution_atol=1e-6,
                valid_mask_path=str(mask),
                valid_mask_value=1,
            )
            prob = np.asarray(prob, dtype=np.float64).ravel()
            y_pred = threshold_probability(prob, 0.5, "greater_equal")

            n_valid = diag["valid_pixels_after_all_masks"]
            require(n_valid == expected_valid,
                    f"{date} {tag}: valid px {n_valid} != {expected_valid}")
            require(prob.size == y_true.size,
                    f"{date} {tag}: prob size {prob.size} != y_true {y_true.size}")

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

            sw = sweep_counts(y_true, prob)
            sw.insert(0, "seed", e["seed"])
            sw.insert(0, "arch", e["label"])
            sw.insert(0, "date", int(date))
            sweep_rows.append(sw)

        print(f"  {date}: {len(entries)} arms swept "
              f"(water frac {float(y_true_reference.mean()):.4f})"
              f"{'  [EXCLUDED from aggregates]' if int(date) in EXCLUDED_DATES else ''}")

    sweep = pd.concat(sweep_rows, ignore_index=True)
    sweep.to_csv(sweep_dest, index=False)
    print(f"\nwrote {sweep_dest} ({len(sweep)} rows)")

    # ---- per-scene AP, and the IoU-vs-threshold profile tau* is selected from ----
    per_scene = (
        sweep.groupby(["arch", "seed", "date"])
        .apply(
            lambda g: pd.Series(
                {"ap": average_precision(g["precision"].to_numpy(), g["recall"].to_numpy())}
            ),
            include_groups=False,
        )
        .reset_index()
    )
    per_scene["excluded"] = per_scene["date"].isin(EXCLUDED_DATES)

    # ---- leave-one-scene-out tau*, over RETAINED scenes only ----
    iou_by = {
        (a, s, dt): g.set_index("threshold")["iou"]
        for (a, s, dt), g in sweep.groupby(["arch", "seed", "date"])
    }
    retained = sorted(set(per_scene.loc[~per_scene["excluded"], "date"]))
    print(f"\ntau* selection uses {len(retained)} retained scenes "
          f"(excluded: {sorted(EXCLUDED_DATES)})")

    tau_rows = []
    for (arch, seed), _ in per_scene.groupby(["arch", "seed"]):
        for held in retained:
            others = [d for d in retained if d != held]
            total = sum(iou_by[(arch, seed, d)] for d in others)
            tau = float(total.idxmax())
            held_profile = iou_by[(arch, seed, held)]
            tau_rows.append(
                {
                    "arch": arch,
                    "seed": seed,
                    "date": held,
                    "tau_star": tau,
                    "iou_at_tau": float(held_profile.loc[tau]),
                    "iou_at_0.5": float(held_profile.loc[0.5]),
                    "n_selection_scenes": len(others),
                }
            )
    tau = pd.DataFrame(tau_rows)

    out = per_scene.merge(tau, on=["arch", "seed", "date"], how="left")
    out.to_csv(scene_dest, index=False)
    print(f"wrote {scene_dest} ({len(out)} rows)")

    kept = out[~out["excluded"]]
    summary = (
        kept.groupby(["arch", "seed"], as_index=False)
        .agg(ap=("ap", "mean"), tau_star=("tau_star", "mean"),
             iou_at_tau=("iou_at_tau", "mean"), n_scenes=("date", "nunique"))
        .groupby("arch", as_index=False)
        .agg(ap_mean=("ap", "mean"), ap_sd=("ap", "std"),
             tau_mean=("tau_star", "mean"), tau_sd=("tau_star", "std"),
             iou_tau_mean=("iou_at_tau", "mean"), n_scenes=("n_scenes", "max"))
        .sort_values("ap_mean", ascending=False)
    )
    print()
    print(summary.round(4).to_string(index=False))
    print(f"\nAP spread across architectures: {summary.ap_mean.min():.3f} - "
          f"{summary.ap_mean.max():.3f} "
          f"(spread {summary.ap_mean.max() - summary.ap_mean.min():.3f})")
    print(f"tau* spread: {summary.tau_mean.min():.3f} - {summary.tau_mean.max():.3f}")


if __name__ == "__main__":
    main()
