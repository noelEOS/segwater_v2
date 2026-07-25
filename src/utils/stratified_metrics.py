"""Stratified / pair-macro water-IoU accumulation for the HPO objective.

Ports the checkpoint-ladder eval conventions verbatim so the HPO objective is
the SAME metric the ladder work established as the only re-ranking signal:
pair-macro water IoU on mixed chips.

Confusion-matrix convention (eval_stratified_ladder.py): code = 2*target + pred,
i.e. a length-4 vector is [TN, FP, FN, TP] = [t0p0, t0p1, t1p0, t1p1], class-1 =
water. Predictions come from a fp32-softmax water-probability threshold at
tau=0.5 (NOT fp16 argmax) — matching the ladder's tau sweep at tau=0.5.

The accumulator consumes raw model `logits` (fp32 softmax over dim 1) plus the
integer label map, per validation batch, in val-loader (shuffle=False) order,
and keeps per-pair and per-stratum confusion matrices on-device.
"""
from typing import Dict

import torch


def _water_iou(cm4) -> float:
    """[TN,FP,FN,TP] -> class-1 IoU = TP/(TP+FP+FN). NaN if union==0.

    Mirrors analyze_stratified_ladder.py:53-57 exactly (NaN, not 0.0, on empty
    union) so the pair-macro mean excludes empty pairs the same way.
    """
    tn, fp, fn, tp = (float(x) for x in cm4)
    union = tp + fp + fn
    return float(tp / union) if union > 0 else float("nan")


def _safe_div(numerator: float, denominator: float, zero_value: float = 0.0) -> float:
    """0.0 on zero denominator (matches scripts/evaluation/metrics.py._safe_div)."""
    return float(numerator / denominator) if denominator else float(zero_value)


class StratifiedWaterAccumulator:
    """GPU accumulator for pair-macro water IoU + stratified guardrails.

    Parameters
    ----------
    stratum_id : int array [N]   pure-land=0, pure-water=1, mixed=2, all-ignore=3
    pair_id    : int array [N]   dense 0..P-1
    eligible   : bool array [P]  pairs to average over (>= min_mixed mixed chips)
    device     : torch device the val forward runs on
    tau        : water-probability threshold (default 0.5)
    ignore_index : label value to drop (default 255)
    """

    # stratum ids (match build_hpo_strata_index.py)
    PURE_LAND, PURE_WATER, MIXED, ALL_IGNORE = 0, 1, 2, 3

    def __init__(self, stratum_id, pair_id, eligible, device, tau: float = 0.5, ignore_index: int = 255):
        self.device = device
        self.tau = float(tau)
        self.ignore_index = int(ignore_index)

        self.stratum_id = torch.as_tensor(stratum_id, dtype=torch.int64, device=device)  # [N]
        self.pair_id = torch.as_tensor(pair_id, dtype=torch.int64, device=device)        # [N]
        self.eligible = torch.as_tensor(eligible, dtype=torch.bool)                       # [P] (cpu ok)
        self.N = int(self.stratum_id.numel())
        self.P = int(self.eligible.numel())

        # strat_cm[s, code]: row 3 (all-ignore) stays all-zero.
        self.strat_cm = torch.zeros((4, 4), dtype=torch.int64, device=device)
        # pair_cm[p, code]: mixed valid px only.
        self.pair_cm = torch.zeros((self.P, 4), dtype=torch.int64, device=device)
        self.base = 0

    @classmethod
    def from_npz(cls, path, device, expected_n=None, default_min_mixed: int = 20):
        """Build an accumulator from a strata-index npz (HPO subset or ladder).

        Handles both npz variants written across this project:
        - HPO subset npz (`data/hpo_strata/hpo_val_pair_strata.npz`) stores an
          `eligible` bool mask (and `min_mixed`) baked at build time -> used
          verbatim.
        - Ladder npz (`experiments/CHECKPOINT_LADDER/strata_index.npz`) has no
          `eligible`/`min_mixed` keys -> eligibility is derived as
          `pair_mixed_count >= min_mixed`, where `min_mixed` is `npz["min_mixed"]`
          if stored else `default_min_mixed` (20, ladder-consistent).

        When `expected_n` is not None, raise ValueError unless the index length
        matches (using `npz["N"]` when present, else `len(stratum_id)`). Same
        fail-fast role as the N-assert the HPO objective used inline.
        """
        import numpy as np

        idx = np.load(path, allow_pickle=True)
        keys = set(idx.keys())

        stratum_id = idx["stratum_id"].astype(np.int64)
        pair_id = idx["pair_id"].astype(np.int64)

        n = int(idx["N"]) if "N" in keys else len(stratum_id)
        if expected_n is not None and n != expected_n:
            raise ValueError(
                f"strata index N={n} != expected_n {expected_n} (wrong memmap for {path}?)"
            )

        if "eligible" in keys:
            eligible = idx["eligible"].astype(bool)
        else:
            min_mixed = int(idx["min_mixed"]) if "min_mixed" in keys else int(default_min_mixed)
            eligible = (idx["pair_mixed_count"] >= min_mixed).astype(bool)

        return cls(stratum_id=stratum_id, pair_id=pair_id, eligible=eligible, device=device)

    def reset(self):
        self.strat_cm.zero_()
        self.pair_cm.zero_()
        self.base = 0

    @torch.no_grad()
    def update(self, logits: torch.Tensor, y: torch.Tensor):
        """Accumulate one batch. `logits` [B,2,H,W], `y` [B,H,W] int64.

        Positional: chip global index = base + arange(B), relying on the val
        loader running shuffle=False in memmap/strata order.
        """
        B, H, W = y.shape
        chip_idx = self.base + torch.arange(B, device=self.device)  # [B]

        # fp32 softmax water prob >= tau  (ladder convention, NOT fp16 argmax)
        pt = (torch.softmax(logits.float(), dim=1)[:, 1] >= self.tau).to(torch.int64)  # [B,H,W]

        valid = y != self.ignore_index                       # [B,H,W]
        yb = torch.clamp(y, 0, 1)                             # ignore rows dropped by valid

        strat_px = self.stratum_id[chip_idx].view(B, 1, 1).expand(B, H, W)  # [B,H,W]
        pair_px = self.pair_id[chip_idx].view(B, 1, 1).expand(B, H, W)

        # ---- strat_cm[4,4]: index = stratum*4 + (2*y+pred), valid px ----
        yb_v = yb[valid]
        strat_v = strat_px[valid]
        code_v = 2 * yb_v + pt[valid]
        s_idx = strat_v * 4 + code_v
        self.strat_cm += torch.bincount(s_idx, minlength=4 * 4).view(4, 4)

        # ---- pair_cm[P,4]: mixed valid px only ----
        mixed_px = (strat_px == self.MIXED) & valid
        pair_m = pair_px[mixed_px]
        yb_m = yb[mixed_px]
        code_m = 2 * yb_m + pt[mixed_px]
        p_idx = pair_m * 4 + code_m
        self.pair_cm += torch.bincount(p_idx, minlength=self.P * 4).view(self.P, 4)

        self.base += B

    def compute(self) -> Dict[str, float]:
        """Reduce CMs (moved to CPU once) to a dict of float metrics.

        Objective: pair_macro_water_iou = unweighted mean of per-pair water IoU
        over eligible pairs whose IoU is finite (union>0). Guardrails computed
        per stratum. All bare divisions use _safe_div (0.0 on empty denom).
        """
        strat_cm = self.strat_cm.cpu().numpy()   # [4,4] -> rows=stratum, cols=[TN,FP,FN,TP]
        pair_cm = self.pair_cm.cpu().numpy()      # [P,4]
        eligible = self.eligible.cpu().numpy()    # [P]

        # --- pair-macro water IoU over eligible, finite pairs ---
        import numpy as np

        pair_wious = np.array([_water_iou(pair_cm[p]) for p in range(self.P)])
        sel = eligible & ~np.isnan(pair_wious)
        pair_macro = float(np.mean(pair_wious[sel])) if sel.any() else float("nan")
        n_pairs_used = int(sel.sum())

        # --- mixed-only diagnostic water IoU (all mixed px, aggregated) ---
        mixed_cm = strat_cm[self.MIXED]           # [4]
        mixed_wiou = _water_iou(mixed_cm)          # NaN if union==0

        # --- guardrails ---
        # pure-land FPR = FP/(FP+TN) over stratum 0
        land = strat_cm[self.PURE_LAND]
        pure_land_fpr = _safe_div(float(land[1]), float(land[1] + land[0]))
        # pure-water FNR = FN/(FN+TP) over stratum 1
        water = strat_cm[self.PURE_WATER]
        pure_water_fnr = _safe_div(float(water[2]), float(water[2] + water[3]))
        # mixed precision/recall (stratum 2)
        mixed_precision = _safe_div(float(mixed_cm[3]), float(mixed_cm[3] + mixed_cm[1]))
        mixed_recall = _safe_div(float(mixed_cm[3]), float(mixed_cm[3] + mixed_cm[2]))
        # overall precision/recall (all strata)
        tot = strat_cm.sum(axis=0)                 # [4] TN,FP,FN,TP over all valid px
        overall_precision = _safe_div(float(tot[3]), float(tot[3] + tot[1]))
        overall_recall = _safe_div(float(tot[3]), float(tot[3] + tot[2]))

        return {
            "pair_macro_water_iou": pair_macro,
            "n_pairs_used": float(n_pairs_used),
            "mixed_water_iou": mixed_wiou,
            "pure_land_fpr": pure_land_fpr,
            "pure_water_fnr": pure_water_fnr,
            "mixed_precision": mixed_precision,
            "mixed_recall": mixed_recall,
            "overall_precision": overall_precision,
            "overall_recall": overall_recall,
        }
