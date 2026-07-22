"""Pooled AP + IoU@0.50 + F1@0.50 with spatial-block-bootstrap CIs at Hampyeong Bay.

Purpose
-------
Produce, for Hampyeong, the SAME accuracy quantities reported for Demak/Semarang so
the two sites are analysed by one consistent method:
  - Average Precision (AP)   -- threshold-free
  - IoU@0.50, F1@0.50        -- fixed deployment operating point (threshold 0.5)
each with a POOLED (across-date) spatial block-bootstrap 95% percentile CI.

Method = Option B, but with ZERO definitional drift: the AP / IoU@0.50 / F1@0.50
computation is the *imported* engine function
`scripts/analysis/spatial_block_bootstrap.metrics_from_hist`, and the pooled block
resampling (shared multiplicity draw per iteration, seed 42, 2000 iterations,
percentile CI) replicates `cmd_bootstrap`'s POOLED path exactly. Validated: pooling
the engine's own stored Demak histograms through this same code reproduces
Demak Swin-B pooled iou@0.50 = 0.706552, f1@0.50 = 0.828046, ap = 0.915288 to <1e-4
(see the validate_engine_against_demak() check that runs at startup).

Pixel access reuses the verified Hampyeong loaders: valid mask, GT, block ids, and
the shared overlap/masking of inference_overlap_utils, so the valid-pixel set and
its row-major order are identical to hampyeong_model_comparison.load_pair and to
hampyeong_block_bootstrap.block_ids_for_valid_pixels.

Design (matched to Demak engine):
  - blocks: 200 px (2 km @ 10 m), row-major over the valid-mask reference grid.
  - fine histogram: 1000 bins per block (sufficient statistic for AP + F1/IoU@t).
  - pooled: collapse all dates within a block, then resample blocks (a resampled
    block carries all its dates together), so temporal correlation is preserved.
  - 2000 iterations, one shared multiplicity vector per iteration reused for every
    model (paired), percentile 95% CI, RNG seed 42.
  - two scopes: 3-date (legacy-comparable) and 5-date (clean dates; 20211019 excluded).

Models: 8 new architectures as the 3-seed MEAN of their probability rasters, plus
the 2 legacy single ResNet50 runs (binary {0,1}; AP is degenerate for a binary map,
so legacy reports only IoU@0.50 / F1@0.50).

Run: /opt/homebrew/Caskroom/miniforge/base/envs/eda/bin/python scripts/evaluation/hampyeong_pooled_ap_iou.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import yaml

_SCRIPTS = Path(__file__).resolve().parent.parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
_ANALYSIS = _SCRIPTS / "analysis"
if str(_ANALYSIS) not in sys.path:
    sys.path.insert(0, str(_ANALYSIS))

from evaluation.hampyeong_model_comparison import (  # noqa: E402
    CKPT_KEY_BY_ARCH,
    EXPECTED_VALID_PIXELS,
    NEW_RUNS,
    OLD_RUNS,
    SCENE_IDS as SCENE_IDS_3,
    gt_path,
    old_pred_path,
    valid_mask_path,
)
from inference_overlap_utils import _load_overlap_reference_and_probability  # noqa: E402
from spatial_block_bootstrap import metrics_from_hist, _percentile_ci  # noqa: E402

DEFAULT_NAS_ROOT = "/Volumes/WD_8tb_RedPlus_NAS_A/MACKBOOK_AIR_M2_BACKUP/Documents/EOS/ACDC"
DEFAULT_RUNS_ROOT = Path("experiments/hampyeong/runs")
DEFAULT_TOPLEVEL_ROOT = Path("experiments/hampyeong")
DEFAULT_OUT_DIR = Path("experiments/hampyeong/evaluation")

BLOCK_PX = 200
FINE_BINS = 1000
COARSE_BINS = 15
N_BOOT = 2000
BOOT_SEED = 42
THRESHOLD = 0.50
RESOLUTION_ATOL = 1e-6

# All SAR dates with GT + full-clip inputs. 20211019 excluded (pre-clipped input).
SCENE_IDS = dict(SCENE_IDS_3)
SCENE_IDS["20210504"] = "S1B_IW_GRDH_1SDV_20210504T213225_20210504T213250_026760_033253_DB62_Clipped"
SCENE_IDS["20210913"] = "S1B_IW_GRDH_1SDV_20210913T213233_20210913T213258_028685_036C5F_7EDB_Clipped"

DATES_3 = ["20210305", "20210422", "20210621"]
DATES_5 = ["20210305", "20210422", "20210504", "20210621", "20210913"]

# ---- New architecture run-dirs -------------------------------------------------
# The 7 architectures in hampyeong_model_comparison live under runs/. Swin-Large
# lives at the top level of experiments/hampyeong/. All are s32 stride, 3 seeds.
SWIN_LARGE_DIRS = {
    19: "dev_sweep_all_hampyeong_s19_20260711T091955Z_upernet_swin_large_224_native224_weighted_224_b0_s32",
    42: "dev_sweep_all_hampyeong_s42_20260711T091848Z_upernet_swin_large_224_native224_weighted_224_b0_s32",
    58: "dev_sweep_all_hampyeong_s58_20260711T091910Z_upernet_swin_large_224_native224_weighted_224_b0_s32",
}
CKPT_KEY_BY_ARCH_EXT = dict(CKPT_KEY_BY_ARCH)
CKPT_KEY_BY_ARCH_EXT["Swin-Large"] = "upernet_tu-swin_large_patch4_window7_224"

SEEDS = (19, 42, 58)


def build_new_model_specs():
    """(arch_label, {seed: (root, run_dir)}). root distinguishes runs/ vs top-level."""
    specs: dict[str, dict[int, tuple[str, str]]] = {}
    for _model, arch, seed, run_dir in NEW_RUNS:
        specs.setdefault(arch, {})[seed] = ("runs", run_dir)
    specs["Swin-Large"] = {s: ("toplevel", d) for s, d in SWIN_LARGE_DIRS.items()}
    return specs


NEW_MODEL_SPECS = build_new_model_specs()
# Report label order (as requested): Swin-B, Swin-Large, ConvNeXtV2, DeepLabV3+,
# DPT-ViT-B, SegFormer-B4, UNet-R50, UNet++-R50
NEW_MODEL_ORDER = [
    "Swin-B", "Swin-Large", "ConvNeXtV2", "DeepLabV3+",
    "DPT-ViT-B", "SegFormer-B4", "UNet-R50", "UNet++-R50",
]
LEGACY_ORDER = ["ResNet50-baseline", "ResNet50-finetuned"]


def new_pred_path(runs_root: Path, toplevel_root: Path, root_kind: str, run_dir: str, date: str) -> Path:
    scene = SCENE_IDS[date]
    base = runs_root if root_kind == "runs" else toplevel_root
    return base / run_dir / scene / f"{scene}_probability_water.tif"


# --------------------------------------------------------------------------- #
# Provenance / distinctness audit (extends hampyeong_model_comparison's audit
# to Swin-Large; verifies the 3 seed rasters per model are distinct).
# --------------------------------------------------------------------------- #

def audit_provenance(runs_root: Path, toplevel_root: Path, dates: list[str]) -> None:
    print("=== Provenance + distinctness audit ===")
    seen_ckpt: dict[str, str] = {}
    for arch in NEW_MODEL_ORDER:
        for seed in SEEDS:
            root_kind, run_dir = NEW_MODEL_SPECS[arch][seed]
            base = (runs_root if root_kind == "runs" else toplevel_root) / run_dir
            cfg = yaml.safe_load((base / "run_config.yaml").read_text())
            summ = json.loads((base / "run_summary.json").read_text())
            cfg_ckpt = cfg["inference"]["checkpoint_path"]
            if cfg_ckpt != summ["checkpoint_path"]:
                raise AssertionError(f"{arch} s{seed}: config vs summary checkpoint mismatch")
            expected = f"outputs/stage2/{CKPT_KEY_BY_ARCH_EXT[arch]}/s{seed}/best.pth"
            if cfg_ckpt != expected:
                raise AssertionError(f"{arch} s{seed}: expected {expected}, got {cfg_ckpt}")
            m = re.search(r"/s(\d+)/", cfg_ckpt)
            if not m or int(m.group(1)) != seed:
                raise AssertionError(f"{arch}: dir seed {seed} != checkpoint {cfg_ckpt}")
            if cfg_ckpt in seen_ckpt:
                raise AssertionError(f"checkpoint collision {arch} s{seed} vs {seen_ckpt[cfg_ckpt]}")
            seen_ckpt[cfg_ckpt] = f"{arch} s{seed}"
    print(f"  checkpoint provenance OK: {len(seen_ckpt)} distinct checkpoints "
          f"across {len(NEW_MODEL_ORDER)}x{len(SEEDS)} runs")

    # raster distinctness: the 3 seed probability rasters per model must differ,
    # on every date, by sha1 of the valid-pixel probability vector.
    mask = valid_mask_path(_nas)
    for arch in NEW_MODEL_ORDER:
        for date in dates:
            digests = {}
            for seed in SEEDS:
                root_kind, run_dir = NEW_MODEL_SPECS[arch][seed]
                pth = new_pred_path(runs_root, toplevel_root, root_kind, run_dir, date)
                y_true, y_score, diag = _load_overlap_reference_and_probability(
                    reference_path=str(gt_path(_nas, date)), prediction_path=str(pth),
                    reference_water_values=[1], reference_nodata_values=None,
                    resolution_atol=RESOLUTION_ATOL, valid_mask_path=str(mask), valid_mask_value=1)
                assert diag["valid_pixels_after_all_masks"] == EXPECTED_VALID_PIXELS, \
                    f"{arch} s{seed} {date}: valid px {diag['valid_pixels_after_all_masks']}"
                dg = hashlib.sha1(y_score.tobytes()).hexdigest()
                if dg in digests:
                    raise AssertionError(f"{arch} {date}: seeds {seed} and {digests[dg]} identical raster")
                digests[dg] = seed
    print(f"  raster distinctness OK: 3 seeds distinct for every model x date\n")


# --------------------------------------------------------------------------- #
# Load a model's per-block fine histogram, pooled over the scope's dates.
# For new models: mean of the 3 seed probability rasters (matches the Hampyeong
# new-vs-legacy analysis). For legacy: the single binary raster.
# --------------------------------------------------------------------------- #

def _fine_hist_for_scores(y_true: np.ndarray, y_score: np.ndarray, block_index: np.ndarray,
                          n_blocks: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """[B, fine] count / pos / sum_p, identical binning to the engine's histogram pass."""
    fidx = np.minimum((y_score * FINE_BINS).astype(np.int64), FINE_BINS - 1)
    comp = block_index.astype(np.int64) * FINE_BINS + fidx
    size = n_blocks * FINE_BINS
    count = np.bincount(comp, minlength=size).reshape(n_blocks, FINE_BINS)
    pos = np.bincount(comp, weights=y_true, minlength=size).reshape(n_blocks, FINE_BINS)
    sump = np.bincount(comp, weights=y_score, minlength=size).reshape(n_blocks, FINE_BINS)
    return count.astype(np.int64), pos.astype(np.int64), sump.astype(np.float64)


def load_model_scores(kind: str, arch: str, date: str, runs_root: Path, toplevel_root: Path,
                      nas_root: Path, mask: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return (y_true, y_score) on the valid pixels. New = 3-seed mean probability."""
    ref = gt_path(nas_root, date)
    if kind == "new":
        y_true_ref = None
        acc = None
        for seed in SEEDS:
            root_kind, run_dir = NEW_MODEL_SPECS[arch][seed]
            pth = new_pred_path(runs_root, toplevel_root, root_kind, run_dir, date)
            yt, ys, diag = _load_overlap_reference_and_probability(
                reference_path=str(ref), prediction_path=str(pth),
                reference_water_values=[1], reference_nodata_values=None,
                resolution_atol=RESOLUTION_ATOL, valid_mask_path=str(mask), valid_mask_value=1)
            assert diag["valid_pixels_after_all_masks"] == EXPECTED_VALID_PIXELS
            if y_true_ref is None:
                y_true_ref = yt
                acc = ys.astype(np.float64)
            else:
                assert np.array_equal(yt, y_true_ref), f"{arch} {date}: y_true differs across seeds"
                acc += ys
        y_score = np.clip(acc / len(SEEDS), 0.0, 1.0).astype(np.float32)
        return y_true_ref.astype(np.float64), y_score
    else:  # legacy single run; subdir carried in arch tuple
        subdir = arch
        pth = old_pred_path(nas_root, subdir, date)
        yt, ys, diag = _load_overlap_reference_and_probability(
            reference_path=str(ref), prediction_path=str(pth),
            reference_water_values=[1], reference_nodata_values=None,
            resolution_atol=RESOLUTION_ATOL, valid_mask_path=str(mask), valid_mask_value=1)
        assert diag["valid_pixels_after_all_masks"] == EXPECTED_VALID_PIXELS
        return yt.astype(np.float64), np.clip(ys, 0.0, 1.0).astype(np.float32)


def block_ids_for_valid_pixels(mask_path: Path) -> tuple[np.ndarray, int]:
    with rasterio.open(mask_path) as src:
        valid = src.read(1) == 1
    if int(valid.sum()) != EXPECTED_VALID_PIXELS:
        raise AssertionError(f"valid mask has {valid.sum()} px, expected {EXPECTED_VALID_PIXELS}")
    rows, cols = np.nonzero(valid)  # row-major, matches boolean indexing in the loader
    raw = (rows // BLOCK_PX).astype(np.int64) * 100_000 + (cols // BLOCK_PX)
    _uniq, dense = np.unique(raw, return_inverse=True)
    return dense.astype(np.int64), int(_uniq.size)


# --------------------------------------------------------------------------- #
# Pooled block bootstrap (shared draws, percentile CI) -- mirrors cmd_bootstrap
# POOLED path. Metrics via the imported engine metrics_from_hist.
# --------------------------------------------------------------------------- #

def pooled_metrics(pooled_hist: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
                   legacy_models: set[str], n_blocks: int,
                   thr_by_model: dict[str, float] | None = None,
                   calibrated: bool = False) -> pd.DataFrame:
    """Point estimates + pooled bootstrap. thr_by_model gives each model its own
    decision threshold (calibrated mode); default is THRESHOLD for everyone and
    reproduces the canonical output bit-for-bit (labels iou@0.50 / f1@0.50).

    In calibrated mode the metric labels are iou@tau / f1@tau (uniform across
    models so the paired contrasts stay well-defined) and each row carries its
    model's threshold. metrics_from_hist snaps thresholds to the fine-histogram
    edges (0.001 in probability): tau values from the 0.005 val grid are exact.
    """
    def thr_of(m: str) -> float:
        if thr_by_model is None or m in legacy_models:
            return THRESHOLD
        return thr_by_model[m]

    iou_key = "iou@tau" if calibrated else f"iou@{THRESHOLD:.2f}"
    f1_key = "f1@tau" if calibrated else f"f1@{THRESHOLD:.2f}"
    metric_keys = ["ap", iou_key, f1_key]
    rng = np.random.default_rng(BOOT_SEED)
    models = list(pooled_hist.keys())

    def extract(res: dict, t: float) -> dict:
        return {"ap": res["ap"], iou_key: res[f"iou@{t:.2f}"], f1_key: res[f"f1@{t:.2f}"]}

    # point estimates (all blocks)
    point = {}
    for m in models:
        c, p, s = pooled_hist[m]
        t = thr_of(m)
        res = metrics_from_hist(c.sum(0), p.sum(0), s.sum(0), FINE_BINS, COARSE_BINS, [t])
        point[m] = extract(res, t)

    # bootstrap replicates, shared multiplicity per iteration
    rep = {m: {k: np.full(N_BOOT, np.nan) for k in metric_keys} for m in models}
    for r in range(N_BOOT):
        draw = rng.integers(0, n_blocks, size=n_blocks)
        m_vec = np.bincount(draw, minlength=n_blocks).astype(np.float64)
        for m in models:
            c, p, s = pooled_hist[m]
            t = thr_of(m)
            res = extract(metrics_from_hist(m_vec @ c, m_vec @ p, m_vec @ s,
                                            FINE_BINS, COARSE_BINS, [t]), t)
            for k in metric_keys:
                rep[m][k][r] = res[k]

    rows = []
    for m in models:
        is_legacy = m in legacy_models
        for k in metric_keys:
            if is_legacy and k == "ap":
                continue  # binary map -> degenerate PR curve; AP not meaningful
            lo, hi = _percentile_ci(rep[m][k])
            row = {
                "model": m, "metric": k, "point": point[m][k],
                "boot_mean": float(np.nanmean(rep[m][k])),
                "ci_lo": lo, "ci_hi": hi,
                "n_blocks": n_blocks, "n_boot": N_BOOT,
            }
            if calibrated:
                row["threshold"] = thr_of(m)
            rows.append(row)
    return pd.DataFrame(rows), rep, point


def oracle_rows(scope_name: str, pooled_hist: dict, legacy_models: set[str],
                thr_by_model: dict[str, float] | None) -> list[dict]:
    """DIAGNOSTIC CEILING, TUNED ON TEST: per model, the pooled IoU at the
    test-optimal threshold (argmax over the 1000 fine-histogram edges).
    Never quote as a performance number (plan Option 1, Phase 4)."""
    rows = []
    for m, (c, p, s) in pooled_hist.items():
        if m in legacy_models:
            continue  # binary raster: threshold sweep is degenerate
        count, pos = c.sum(0).astype(np.float64), p.sum(0).astype(np.float64)
        neg = count - pos
        P = pos.sum()
        tp = np.cumsum(pos[::-1])[::-1]   # predict water = bins >= k
        fp = np.cumsum(neg[::-1])[::-1]
        with np.errstate(divide="ignore", invalid="ignore"):
            iou = tp / np.maximum(tp + fp + (P - tp), 1)
        k = int(np.argmax(iou))
        edges = np.linspace(0.0, 1.0, FINE_BINS + 1)
        k05 = int(round(THRESHOLD * FINE_BINS))
        row = {
            "scope": scope_name, "model": m,
            "tau_oracle_test": float(edges[k]),
            "iou_at_tau_oracle": float(iou[k]),
            "iou_at_0.5": float(iou[k05]),
            "note": "tuned-on-test upper bound; diagnostic only",
        }
        if thr_by_model is not None and m in thr_by_model:
            kv = int(round(thr_by_model[m] * FINE_BINS))
            row["tau_val"] = thr_by_model[m]
            row["iou_at_tau_val"] = float(iou[min(kv, FINE_BINS - 1)])
        rows.append(row)
    return rows


def run_scope(scope_name: str, dates: list[str], runs_root: Path, toplevel_root: Path,
              nas_root: Path, mask: Path, block_index: np.ndarray, n_blocks: int,
              thr_by_model: dict[str, float] | None = None, calibrated: bool = False,
              oracle: bool = False) -> pd.DataFrame:
    print(f"=== Scope {scope_name}: dates {dates} ===")
    # pooled histogram per model = sum of per-date per-block fine histograms
    pooled_hist: dict[str, tuple] = {}
    legacy_models: set[str] = set()

    def accumulate(model_key: str, kind: str, arch_or_subdir: str):
        acc = None
        for date in dates:
            y_true, y_score = load_model_scores(kind, arch_or_subdir, date, runs_root,
                                                toplevel_root, nas_root, mask)
            c, p, s = _fine_hist_for_scores(y_true, y_score, block_index, n_blocks)
            if acc is None:
                acc = [c, p, s]
            else:
                acc[0] += c; acc[1] += p; acc[2] += s
        pooled_hist[model_key] = tuple(acc)

    for arch in NEW_MODEL_ORDER:
        accumulate(arch, "new", arch)
        print(f"  loaded new  {arch}")
    for _mdl, label, subdir in OLD_RUNS:
        accumulate(label, "legacy", subdir)
        legacy_models.add(label)
        print(f"  loaded legacy {label}")

    df, rep, _point = pooled_metrics(pooled_hist, legacy_models, n_blocks,
                                     thr_by_model=thr_by_model, calibrated=calibrated)
    df.insert(0, "scope", scope_name)

    # paired pooled contrasts on IoU at the operating point (shared draws already
    # used above, so the per-iteration difference cancels shared spatial
    # variance). All model pairs. In calibrated mode each model sits at its own
    # threshold, so the contrast compares deployed operating points.
    iou_key = "iou@tau" if calibrated else f"iou@{THRESHOLD:.2f}"
    models = list(pooled_hist.keys())
    contr_rows = []
    for i in range(len(models)):
        for j in range(i + 1, len(models)):
            A, Bm = models[i], models[j]
            diff = rep[A][iou_key] - rep[Bm][iou_key]
            lo, hi = _percentile_ci(diff)
            crow = {
                "scope": scope_name, "metric": iou_key, "model_A": A, "model_B": Bm,
                "diff_mean": float(np.nanmean(diff)), "ci_lo": lo, "ci_hi": hi,
                "excludes_zero": bool((lo > 0) or (hi < 0)),
                "n_blocks": n_blocks, "n_boot": N_BOOT,
            }
            if calibrated:
                crow["thr_A"] = thr_by_model.get(A, THRESHOLD) if A not in legacy_models else THRESHOLD
                crow["thr_B"] = thr_by_model.get(Bm, THRESHOLD) if Bm not in legacy_models else THRESHOLD
            contr_rows.append(crow)

    odf = pd.DataFrame(oracle_rows(scope_name, pooled_hist, legacy_models,
                                   thr_by_model)) if oracle else None
    return df, pd.DataFrame(contr_rows), odf


def validate_engine_against_demak() -> None:
    """Reproduce Demak Swin-B pooled numbers from the engine's stored histograms."""
    hdir = (Path("experiments/demak_semarang/s1_s2_concurrent_accuracy_multiseed/evaluation/"
                 "semarang_probability_aucroc_s42_s32/spatial_block_bootstrap/histograms_block200px"))
    if not hdir.exists():
        print("  (Demak histograms not found; skipping engine cross-check)\n")
        return
    model = "Upernet_Swin_Base_224_native224_weighted"
    ddates = ['2019-06-22', '2019-10-20', '2020-06-16', '2021-08-10', '2023-07-31', '2024-07-25']
    c = p = s = None
    for d in ddates:
        z = np.load(hdir / f"{model}__{d}.npz")
        if c is None:
            c, p, s = z["count"].copy(), z["pos"].copy(), z["sum_p"].copy()
        else:
            c += z["count"]; p += z["pos"]; s += z["sum_p"]
    res = metrics_from_hist(c.sum(0), p.sum(0), s.sum(0), FINE_BINS, COARSE_BINS, [THRESHOLD])
    for k, tgt in [("ap", 0.915288), ("iou@0.50", 0.706552), ("f1@0.50", 0.828046)]:
        assert abs(res[k] - tgt) < 1e-4, f"engine cross-check {k}: {res[k]} vs {tgt}"
    print(f"  engine cross-check OK: Demak Swin-B pooled ap={res['ap']:.6f} "
          f"iou@0.50={res['iou@0.50']:.6f} f1@0.50={res['f1@0.50']:.6f}\n")


_nas: Path  # set in main for the audit's convenience


def main() -> None:
    global _nas
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nas-root", type=Path, default=Path(DEFAULT_NAS_ROOT))
    ap.add_argument("--runs-root", type=Path, default=DEFAULT_RUNS_ROOT)
    ap.add_argument("--toplevel-root", type=Path, default=DEFAULT_TOPLEVEL_ROOT)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--threshold", type=float, default=None,
                    help="Single decision threshold for all new models "
                         "(default: 0.5, canonical output). Legacy binary "
                         "rasters always stay at 0.5.")
    ap.add_argument("--threshold-file", type=Path, default=None,
                    help="val_thresholds.csv from select_val_thresholds.py; each "
                         "model scored at its ENSEMBLE tau*_iou (this script's "
                         "rasters are 3-seed mean probabilities)")
    ap.add_argument("--out-suffix", default=None,
                    help="Suffix for calibrated output CSVs (default: _tauval "
                         "for --threshold-file, _thr<t> for --threshold)")
    ap.add_argument("--oracle", action="store_true",
                    help="Also write the tuned-on-test oracle-threshold table "
                         "(diagnostic ceiling, plan Option 1 Phase 4)")
    args = ap.parse_args()
    if not args.nas_root.exists():
        raise SystemExit(f"NAS root not found: {args.nas_root}")
    _nas = args.nas_root

    if args.threshold is not None and args.threshold_file is not None:
        raise SystemExit("--threshold and --threshold-file are mutually exclusive")
    calibrated = args.threshold is not None or args.threshold_file is not None
    thr_by_model = None
    if args.threshold_file is not None:
        table = pd.read_csv(args.threshold_file).set_index("name")["tau_star_iou"]
        thr_by_model = {}
        for arch in NEW_MODEL_ORDER:
            name = f"{CKPT_KEY_BY_ARCH_EXT[arch]}_ensemble"
            if name not in table.index:
                raise SystemExit(f"No row named {name} in {args.threshold_file}")
            thr_by_model[arch] = float(table.loc[name])
        print("Per-model ensemble tau* (val-selected):")
        for arch in NEW_MODEL_ORDER:
            print(f"  {arch:12s} tau* = {thr_by_model[arch]:.3f}")
        print()
    elif args.threshold is not None:
        thr_by_model = {arch: args.threshold for arch in NEW_MODEL_ORDER}

    print("=== Validation: engine metrics reproduce Demak pooled numbers ===")
    validate_engine_against_demak()

    mask = valid_mask_path(args.nas_root)
    block_index, n_blocks = block_ids_for_valid_pixels(mask)
    print(f"AOI -> {n_blocks} blocks of {BLOCK_PX*10/1000:.0f} km ({EXPECTED_VALID_PIXELS} valid px)\n")

    audit_provenance(args.runs_root, args.toplevel_root, DATES_5)

    frames, contrasts, oracles = [], [], []
    for scope, dates in [("3date", DATES_3), ("5date", DATES_5)]:
        df, cdf, odf = run_scope(scope, dates, args.runs_root, args.toplevel_root,
                                 args.nas_root, mask, block_index, n_blocks,
                                 thr_by_model=thr_by_model, calibrated=calibrated,
                                 oracle=args.oracle)
        frames.append(df)
        contrasts.append(cdf)
        if odf is not None:
            oracles.append(odf)
    out = pd.concat(frames, ignore_index=True)
    contr = pd.concat(contrasts, ignore_index=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    if calibrated:
        suffix = args.out_suffix or (
            "_tauval" if args.threshold_file is not None else f"_thr{args.threshold:g}")
    else:
        suffix = ""
    out_csv = args.out_dir / f"hampyeong_pooled_ap_iou_at05{suffix}.csv"
    contr_csv = args.out_dir / f"hampyeong_pooled_iou_paired_contrasts{suffix}.csv"
    out.to_csv(out_csv, index=False)
    contr.to_csv(contr_csv, index=False)
    print(f"\nWrote {out_csv}  ({len(out)} rows, n_blocks={n_blocks})")
    print(f"Wrote {contr_csv}  ({len(contr)} paired IoU contrasts)")
    if oracles:
        oracle_csv = args.out_dir / f"hampyeong_oracle_thresholds{suffix}.csv"
        pd.concat(oracles, ignore_index=True).to_csv(oracle_csv, index=False)
        print(f"Wrote {oracle_csv}  (TUNED-ON-TEST diagnostic ceiling — never a headline)")


if __name__ == "__main__":
    main()
