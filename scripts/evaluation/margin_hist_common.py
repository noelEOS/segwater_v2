"""Shared helpers for the val-split margin histograms (threshold-calibration plans).

A histogram artifact (written by val_split_probability_histograms.py) is a pair
of counts over N_BINS margin bins, GT-water and GT-land, with bin edges in
margin space m = z_water - z_land, so p_water = sigmoid(m) and a probability
threshold tau corresponds to the margin cut logit(tau).

Confusion at a cut placed on bin edge k (predict water <=> bin index >= k):
    TP(k) = hist_water[k:].sum()   FP(k) = hist_land[k:].sum()
    FN(k) = W - TP(k)              TN(k) = L - FP(k)
computed for every k in one cumulative sum.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


def sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=np.float64)))


def logit(p: np.ndarray | float) -> np.ndarray | float:
    p = np.asarray(p, dtype=np.float64)
    return np.log(p) - np.log1p(-p)


@dataclass
class MarginHist:
    name: str            # directory name, e.g. unet_resnet50_s19
    hist_water: np.ndarray
    hist_land: np.ndarray
    bin_edges: np.ndarray  # (N_BINS + 1,) margin values
    n_ignored: int
    n_pixels_total: int
    meta: dict

    @property
    def n_bins(self) -> int:
        return self.hist_water.size

    @property
    def bin_centers(self) -> np.ndarray:
        return 0.5 * (self.bin_edges[:-1] + self.bin_edges[1:])

    def halved(self) -> "MarginHist":
        """Pairwise-summed copy with half the bins (fit-sensitivity check)."""
        assert self.n_bins % 2 == 0
        return MarginHist(
            name=self.name + "_halfbins",
            hist_water=self.hist_water.reshape(-1, 2).sum(1),
            hist_land=self.hist_land.reshape(-1, 2).sum(1),
            bin_edges=self.bin_edges[::2],
            n_ignored=self.n_ignored,
            n_pixels_total=self.n_pixels_total,
            meta=self.meta,
        )


def load_margin_hist(hist_dir: Path) -> MarginHist:
    z = np.load(hist_dir / "hist_margin.npz")
    meta = json.loads((hist_dir / "meta.json").read_text())
    h = MarginHist(
        name=hist_dir.name,
        hist_water=z["hist_water"].astype(np.int64),
        hist_land=z["hist_land"].astype(np.int64),
        bin_edges=z["bin_edges"].astype(np.float64),
        n_ignored=int(z["n_ignored"]),
        n_pixels_total=int(z["n_pixels_total"]),
        meta=meta,
    )
    assert int(h.hist_water.sum()) + int(h.hist_land.sum()) + h.n_ignored == h.n_pixels_total, \
        f"{hist_dir}: count invariant violated"
    return h


def discover_hists(calib_root: Path) -> list[Path]:
    """All histogram dirs under a calibration root, seeds before ensembles."""
    dirs = sorted(d for d in calib_root.iterdir()
                  if d.is_dir() and (d / "hist_margin.npz").exists())
    return sorted(dirs, key=lambda d: (d.name.endswith("_ensemble"), d.name))


def confusion_curves(h: MarginHist) -> dict[str, np.ndarray]:
    """Metrics at every bin-edge cut k = 0..N_BINS (predict water <=> bin >= k).

    Returns arrays of length N_BINS + 1 keyed by:
    edge_margin, tau, TP, FP, FN, TN, iou_water, iou_land, miou, f1_water, mcc.
    """
    W = int(h.hist_water.sum())
    L = int(h.hist_land.sum())
    # suffix sums: TP[k] = sum_{i >= k} hist_water[i]
    tp = np.concatenate([np.cumsum(h.hist_water[::-1])[::-1], [0]]).astype(np.float64)
    fp = np.concatenate([np.cumsum(h.hist_land[::-1])[::-1], [0]]).astype(np.float64)
    fn = W - tp
    tn = L - fp
    with np.errstate(divide="ignore", invalid="ignore"):
        iou_w = tp / np.maximum(tp + fp + fn, 1)
        iou_l = tn / np.maximum(tn + fp + fn, 1)
        f1_w = 2 * tp / np.maximum(2 * tp + fp + fn, 1)
        denom = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        mcc = np.where(denom > 0, (tp * tn - fp * fn) / np.maximum(denom, 1), 0.0)
    return {
        "edge_margin": h.bin_edges,
        "tau": sigmoid(h.bin_edges),
        "TP": tp, "FP": fp, "FN": fn, "TN": tn,
        "iou_water": iou_w, "iou_land": iou_l, "miou": (iou_w + iou_l) / 2.0,
        "f1_water": f1_w, "mcc": mcc,
    }


def edge_index_for_tau(h: MarginHist, tau: float) -> int:
    """Nearest bin edge to the margin cut logit(tau)."""
    m = float(logit(tau))
    k = int(round((m - h.bin_edges[0]) / (h.bin_edges[1] - h.bin_edges[0])))
    return int(np.clip(k, 0, h.n_bins))
