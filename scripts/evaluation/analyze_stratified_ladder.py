"""Analyze the checkpoint-ladder stratified accumulators -> CSVs + H1/H2/H3.

Consumes the per-checkpoint artifacts written by eval_stratified_ladder.py
(raw/<seed>/<ckpt>.npz + <ckpt>_chipcm.npy) and produces:

  ladder_metrics.csv   one row/ckpt: every selection-candidate metric.
  ranking_compare.csv  per-metric rank of each ckpt + dynamic range (max-min).
  last_vs_best.csv     H2: last/late vs early-best, per seed, per metric.
  tau_drift.csv        H3: tau*(global) and tau*(mixed) per ckpt.

CM convention (as written by the eval): a length-4 vector is
  [TN, FP, FN, TP] = [t0p0, t0p1, t1p0, t1p1], class-1 = water.

Metric definitions (all computed, never estimated):
  pooled mIoU              macro IoU (2 classes, union>0) from chip_cm.sum(0)
                           == the incumbent selection metric (anchor-checked).
  mixed-only pooled mIoU   same macro IoU but from strat_cm[stratum=mixed] @ tau=0.5.
  mixed-only water IoU     class-1 IoU from strat_cm[mixed] @ tau=0.5.
  pair-macro water IoU     per-pair pooled water IoU @ tau=0.5 over pairs with
                           >=20 mixed chips, unweighted mean across pairs.
  boundary IoU +-1 / +-2   class-1 IoU in the +-1px core / +-2px band @ tau=0.5.
  tau* global              argmax over tau of pooled water IoU (all valid px).
  tau* mixed               argmax over tau of mixed-only water IoU.
  mixed share of FP+FN     (FP+FN in mixed) / (FP+FN over all strata) @ tau=0.5.

Usage (Mac eda env):
    python scripts/evaluation/analyze_stratified_ladder.py \
        --raw-root experiments/CHECKPOINT_LADDER/raw \
        --strata-index experiments/CHECKPOINT_LADDER/strata_index.npz \
        --out-dir experiments/CHECKPOINT_LADDER/results
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

MIN_MIXED = 20  # pair gate for pair-macro


def macro_miou(cm4: np.ndarray) -> float:
    """[TN,FP,FN,TP] -> macro IoU over classes with union>0 (float64)."""
    tn, fp, fn, tp = [float(x) for x in cm4]
    cm = np.array([[tn, fp], [fn, tp]], dtype=np.float64)
    diag = np.diag(cm)
    union = cm.sum(0) + cm.sum(1) - diag
    m = union > 0
    if not m.any():
        return float("nan")
    return float((diag[m] / union[m]).mean())


def water_iou(cm4: np.ndarray) -> float:
    """class-1 IoU = TP / (TP + FP + FN). NaN if union==0."""
    tn, fp, fn, tp = [float(x) for x in cm4]
    union = tp + fp + fn
    return float(tp / union) if union > 0 else float("nan")


def tau_star(cm_TxT4: np.ndarray, tau_grid: np.ndarray):
    """cm over tau [T,4] -> (tau*, best water IoU). Argmax of water IoU;
    ties resolved to the lowest tau (first argmax)."""
    ious = np.array([water_iou(cm_TxT4[t]) for t in range(cm_TxT4.shape[0])])
    ious_f = np.where(np.isnan(ious), -np.inf, ious)
    ti = int(np.argmax(ious_f))
    return float(tau_grid[ti]), float(ious[ti]), ti


def load_ckpts(raw_root: Path):
    rows = []
    for seed_dir in sorted(raw_root.iterdir()):
        if not seed_dir.is_dir():
            continue
        seed = seed_dir.name
        for npz_path in sorted(seed_dir.glob("*.npz")):
            d = np.load(npz_path, allow_pickle=True)
            chip_cm = np.load(seed_dir / f"{npz_path.stem}_chipcm.npy")  # [N,4]
            rows.append({
                "seed": seed,
                "step": int(d["step"]),
                "role": str(d["role"]),
                "ckpt": str(d["ckpt_name"]),
                "val_miou": float(d["val_miou"]),
                "filename_miou": float(d["filename_miou"]),
                "pooled_anchor": float(d["pooled_miou_fp16argmax"]),
                "anchor_delta": float(d["anchor_delta"]),
                "strat_cm": d["strat_cm"],   # [3,T,4]
                "pair_cm": d["pair_cm"],     # [P,T,4]
                "band_cm": d["band_cm"],     # [2,T,4]
                "tau_grid": d["tau_grid"],   # [T]
                "chip_cm_sum": chip_cm.sum(0).astype(np.int64),  # [4]
            })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-root", type=Path, required=True)
    ap.add_argument("--strata-index", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    idx = np.load(args.strata_index, allow_pickle=True)
    pair_mixed_count = idx["pair_mixed_count"].astype(np.int64)  # [P]
    ge20 = pair_mixed_count >= MIN_MIXED
    n_ge20 = int(ge20.sum())

    rows = load_ckpts(args.raw_root)
    if not rows:
        raise SystemExit(f"No checkpoints found under {args.raw_root}")
    tau_grid = rows[0]["tau_grid"]
    # index of tau=0.5
    t50 = int(np.argmin(np.abs(tau_grid - 0.5)))
    assert abs(tau_grid[t50] - 0.5) < 1e-9, f"tau grid missing 0.5: {tau_grid}"

    MIXED = 2  # stratum id

    out = []
    for r in rows:
        strat_cm = r["strat_cm"]      # [3,T,4]
        pair_cm = r["pair_cm"]        # [P,T,4]
        band_cm = r["band_cm"]        # [2,T,4]

        # --- pooled (incumbent) ---
        pooled = macro_miou(r["chip_cm_sum"])

        # --- mixed-only @ tau=0.5 ---
        mixed50 = strat_cm[MIXED, t50]           # [4]
        mixed_miou = macro_miou(mixed50)
        mixed_wiou = water_iou(mixed50)

        # --- pair-macro water IoU @ tau=0.5 (pairs >=20 mixed) ---
        pair50 = pair_cm[:, t50, :]              # [P,4]
        pair_wious = np.array([water_iou(pair50[p]) for p in range(pair50.shape[0])])
        sel = ge20 & ~np.isnan(pair_wious)
        pair_macro_wiou = float(np.mean(pair_wious[sel])) if sel.any() else float("nan")
        n_pairs_used = int(sel.sum())

        # --- boundary IoU @ tau=0.5 ---
        band1_50 = band_cm[0, t50]               # +-1px core [4]
        band2_50 = band_cm[0, t50] + band_cm[1, t50]  # +-2px = core + ring
        b1_wiou = water_iou(band1_50)
        b2_wiou = water_iou(band2_50)

        # --- tau* global (all valid px = sum over strata) ---
        global_cm_T = strat_cm.sum(0)            # [T,4]
        tau_g, wiou_g, _ = tau_star(global_cm_T, tau_grid)
        # tau* mixed
        tau_m, wiou_m, _ = tau_star(strat_cm[MIXED], tau_grid)

        # --- mixed share of FP+FN @ tau=0.5 ---
        fpfn_all = float(strat_cm[:, t50, 1].sum() + strat_cm[:, t50, 2].sum())
        fpfn_mixed = float(mixed50[1] + mixed50[2])
        mixed_err_share = fpfn_mixed / fpfn_all if fpfn_all > 0 else float("nan")
        # mixed share of chips (context)
        mixed_valid = float(strat_cm[MIXED, t50].sum())
        all_valid = float(strat_cm[:, t50].sum())

        out.append({
            "seed": r["seed"], "step": r["step"], "role": r["role"],
            "val_miou": r["val_miou"], "anchor_delta": r["anchor_delta"],
            "pooled_miou": pooled,
            "mixed_miou": mixed_miou,
            "mixed_water_iou": mixed_wiou,
            "pair_macro_water_iou": pair_macro_wiou,
            "n_pairs_used": n_pairs_used,
            "boundary_iou_pm1": b1_wiou,
            "boundary_iou_pm2": b2_wiou,
            "tau_star_global": tau_g, "water_iou_at_tau_g": wiou_g,
            "tau_star_mixed": tau_m, "water_iou_at_tau_m": wiou_m,
            "mixed_share_fp_fn": mixed_err_share,
            "mixed_valid_px_frac": mixed_valid / all_valid if all_valid > 0 else float("nan"),
        })

    df = pd.DataFrame(out).sort_values(["seed", "step"]).reset_index(drop=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_dir / "ladder_metrics.csv", index=False)

    # ---- ranking_compare.csv : per seed, per metric ranks + dynamic range ----
    RANK_METRICS = [
        "pooled_miou", "mixed_miou", "mixed_water_iou",
        "pair_macro_water_iou", "boundary_iou_pm1", "boundary_iou_pm2",
    ]
    rank_rows = []
    for seed, g in df.groupby("seed"):
        g = g.copy()
        for mcol in RANK_METRICS:
            # rank 1 = best (highest). ties -> min rank.
            g[f"rank_{mcol}"] = g[mcol].rank(ascending=False, method="min").astype(int)
        for _, rr in g.iterrows():
            row = {"seed": seed, "step": rr["step"], "role": rr["role"]}
            for mcol in RANK_METRICS:
                row[mcol] = rr[mcol]
                row[f"rank_{mcol}"] = int(rr[f"rank_{mcol}"])
            rank_rows.append(row)
    rank_df = pd.DataFrame(rank_rows)
    rank_df.to_csv(args.out_dir / "ranking_compare.csv", index=False)

    # dynamic range table (per seed, per metric)
    dyn_rows = []
    for seed, g in df.groupby("seed"):
        for mcol in RANK_METRICS + ["tau_star_global", "tau_star_mixed"]:
            vals = g[mcol].to_numpy(dtype=float)
            vals = vals[~np.isnan(vals)]
            dyn_rows.append({
                "seed": seed, "metric": mcol,
                "min": float(vals.min()), "max": float(vals.max()),
                "range": float(vals.max() - vals.min()),
            })
    pd.DataFrame(dyn_rows).to_csv(args.out_dir / "dynamic_range.csv", index=False)

    # ---- last_vs_best.csv : H2 ----
    # early-"best" = the retained top-k ckpts (role=best); the incumbent pick is
    # argmax val_miou among them. Compare that to last (role=last) and to the
    # latest snapshot, on each candidate metric.
    lvb_rows = []
    for seed, g in df.groupby("seed"):
        best_pool = g[g["role"] == "best"]
        incumbent = best_pool.loc[best_pool["val_miou"].idxmax()] if len(best_pool) else None
        last = g[g["role"] == "last"]
        last = last.iloc[0] if len(last) else None
        late = g.loc[g["step"].idxmax()]  # latest step available (last, or latest snap)
        for mcol in RANK_METRICS + ["tau_star_global", "tau_star_mixed"]:
            row = {"seed": seed, "metric": mcol}
            row["incumbent_step"] = int(incumbent["step"]) if incumbent is not None else None
            row["incumbent_val"] = float(incumbent[mcol]) if incumbent is not None else None
            row["last_step"] = int(last["step"]) if last is not None else None
            row["last_val"] = float(last[mcol]) if last is not None else None
            row["latest_step"] = int(late["step"])
            row["latest_val"] = float(late[mcol])
            if incumbent is not None:
                cmp_val = last[mcol] if last is not None else late[mcol]
                row["late_minus_incumbent"] = float(cmp_val - incumbent[mcol])
            lvb_rows.append(row)
    pd.DataFrame(lvb_rows).to_csv(args.out_dir / "last_vs_best.csv", index=False)

    # ---- tau_drift.csv : H3 ----
    df[["seed", "step", "role", "tau_star_global", "tau_star_mixed",
        "water_iou_at_tau_g", "water_iou_at_tau_m"]].to_csv(
        args.out_dir / "tau_drift.csv", index=False)

    # ---- console summary ----
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 40)
    show = ["seed", "step", "role", "val_miou", "pooled_miou", "mixed_miou",
            "mixed_water_iou", "pair_macro_water_iou", "boundary_iou_pm1",
            "boundary_iou_pm2", "tau_star_global", "tau_star_mixed",
            "mixed_share_fp_fn"]
    print(f"pairs used for pair-macro (>= {MIN_MIXED} mixed): {n_ge20}")
    print(df[show].to_string(index=False))
    print(f"\nWrote CSVs to {args.out_dir}")


if __name__ == "__main__":
    main()
