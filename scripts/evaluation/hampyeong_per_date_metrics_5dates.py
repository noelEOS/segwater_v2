"""Per-date, per-run metrics at Hampyeong Bay over the five clean SAR dates.

Extends hampyeong_model_comparison's per_date_metrics table (three legacy-study
dates) to the full clean-date set (adds 20210504 and 20210913; 20211019 stays
excluded because it was inferred on a pre-clipped input), and adds Swin-Large,
whose three seed runs live at the top level of experiments/hampyeong/.

Everything is reused from the audited scripts: metrics, loader, run specs and
checkpoint provenance come from hampyeong_model_comparison; the two extra scene
IDs and the Swin-Large run dirs come from hampyeong_pooled_ap_iou (which already
audited these runs on all five dates). Rows are per seed, same schema as
per_date_metrics_3_dates.csv.

Writes experiments/hampyeong/evaluation/per_date_metrics_5_dates.csv.

Run with an interpreter that has rasterio, e.g.
    /opt/homebrew/Caskroom/miniforge/base/envs/eda/bin/python
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_SCRIPTS = Path(__file__).resolve().parent.parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from evaluation.metrics import METRIC_NAMES, compute_binary_metrics  # noqa: E402
from evaluation.hampyeong_model_comparison import (  # noqa: E402
    DEFAULT_NAS_ROOT,
    EXPECTED_VALID_PIXELS,
    NEW_RUNS,
    OLD_RUNS,
    audit_checkpoint_provenance,
    gt_path,
    load_pair,
    old_pred_path,
    valid_mask_path,
)
from evaluation.hampyeong_pooled_ap_iou import (  # noqa: E402
    CKPT_KEY_BY_ARCH_EXT,
    DATES_5,
    SCENE_IDS,
    SWIN_LARGE_DIRS,
)

DEFAULT_RUNS_ROOT = Path("experiments/hampyeong/runs")
DEFAULT_TOPLEVEL_ROOT = Path("experiments/hampyeong")
DEFAULT_OUT_DIR = Path("experiments/hampyeong/evaluation")


def _require(path: Path, what: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"{what} not found: {path}")
    return path


def pred_path(base: Path, run_dir: str, date: str) -> Path:
    scene = SCENE_IDS[date]
    return base / run_dir / scene / f"{scene}_probability_water.tif"


def make_threshold_resolver(threshold: float | None, threshold_file: Path | None):
    """Return (resolver(arch, seed) -> float, calibrated: bool).

    Legacy runs (seed is None; binary rasters) always stay at 0.5. With
    --threshold-file, each new run gets its own per-seed tau*_iou selected on
    the training validation split (val_thresholds.csv from
    scripts/evaluation/select_val_thresholds.py).
    """
    if threshold is not None and threshold_file is not None:
        raise SystemExit("--threshold and --threshold-file are mutually exclusive")
    if threshold_file is not None:
        table = pd.read_csv(threshold_file).set_index("name")["tau_star_iou"]

        def resolver(arch: str, seed: int | None) -> float:
            if seed is None:
                return 0.5
            name = f"{CKPT_KEY_BY_ARCH_EXT[arch]}_s{seed}"
            if name not in table.index:
                raise SystemExit(f"No row named {name} in {threshold_file}")
            return float(table.loc[name])

        return resolver, True
    if threshold is not None:
        return (lambda arch, seed: 0.5 if seed is None else float(threshold)), True
    return (lambda arch, seed: 0.5), False


def evaluate(nas_root: Path, runs_root: Path, toplevel_root: Path,
             threshold_for=None, calibrated: bool = False) -> pd.DataFrame:
    if threshold_for is None:
        threshold_for = lambda arch, seed: 0.5  # noqa: E731
    mask = _require(valid_mask_path(nas_root), "valid mask")
    rows: list[dict] = []

    for date in DATES_5:
        reference = _require(gt_path(nas_root, date), f"ground truth {date}")
        y_true_reference: np.ndarray | None = None
        prediction_digests: dict[str, str] = {}

        specs: list[tuple[str, str, int | None, Path]] = []
        for model, arch, subdir in OLD_RUNS:
            specs.append((model, arch, None, _require(old_pred_path(nas_root, subdir, date), f"{model} {date}")))
        for model, arch, seed, run_dir in NEW_RUNS:
            specs.append((model, arch, seed, _require(pred_path(runs_root, run_dir, date), f"{model} {date}")))
        for seed, run_dir in SWIN_LARGE_DIRS.items():
            specs.append((f"swin_large_s{seed}", "Swin-Large", seed,
                          _require(pred_path(toplevel_root, run_dir, date), f"swin_large_s{seed} {date}")))

        for model, arch, seed, prediction in specs:
            thr = threshold_for(arch, seed)
            y_true, y_pred = load_pair(reference, prediction, mask, threshold=thr)

            if seed is not None:
                digest = hashlib.sha1(y_pred.tobytes()).hexdigest()
                if digest in prediction_digests:
                    raise AssertionError(
                        f"{model} and {prediction_digests[digest]} produced identical predictions on "
                        f"{date}. Two runs loaded the same weights — suspect seed miswiring."
                    )
                prediction_digests[digest] = model

            if y_true_reference is None:
                y_true_reference = y_true
            elif not np.array_equal(y_true, y_true_reference):
                raise AssertionError(f"y_true differs across models on {date} (model={model})")

            metrics = compute_binary_metrics(y_true, y_pred, include_counts=True)
            counts_total = metrics["tn"] + metrics["fp"] + metrics["fn"] + metrics["tp"]
            if counts_total != EXPECTED_VALID_PIXELS:
                raise AssertionError(f"confusion counts sum to {counts_total}, expected {EXPECTED_VALID_PIXELS}")

            row = {
                "date": date,
                "model": model,
                "arch": arch,
                "seed": seed,
                **{m: metrics[m] for m in METRIC_NAMES},
                **{c: metrics[c] for c in ("tn", "fp", "fn", "tp")},
                "n_valid": counts_total,
                "water_fraction_gt": float(y_true.mean()),
                "prediction_path": str(prediction),
            }
            if calibrated:
                # extra column only in calibrated mode: the default output must
                # stay byte-identical to the canonical table
                row["threshold"] = thr
            rows.append(row)
            print(f"  {date}  {model:20s} IoU={metrics['iou']:.4f}  F1={metrics['f1']:.4f}  MCC={metrics['mcc']:.4f}")

    return pd.DataFrame(rows)


def check_against_3date_table(df: pd.DataFrame, out_dir: Path) -> None:
    """The 3-date rows must reproduce per_date_metrics_3_dates.csv exactly."""
    ref_csv = out_dir / "per_date_metrics_3_dates.csv"
    if not ref_csv.exists():
        print("(per_date_metrics_3_dates.csv not found; skipping consistency check)")
        return
    ref = pd.read_csv(ref_csv, dtype={"date": str})
    got = df[df["date"].isin(ref["date"].unique()) & (df["arch"] != "Swin-Large")].reset_index(drop=True)
    key = ["date", "model"]
    merged = ref.merge(got, on=key, suffixes=("_ref", "_new"))
    if len(merged) != len(ref):
        raise AssertionError(f"row mismatch vs 3-date table: {len(merged)} joined, {len(ref)} expected")
    for m in list(METRIC_NAMES) + ["tn", "fp", "fn", "tp"]:
        if not np.allclose(merged[f"{m}_ref"], merged[f"{m}_new"], rtol=0, atol=1e-12):
            raise AssertionError(f"metric {m} differs from per_date_metrics_3_dates.csv")
    print(f"Consistency check passed: {len(ref)} rows reproduce per_date_metrics_3_dates.csv exactly")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--nas-root", type=Path, default=Path(DEFAULT_NAS_ROOT))
    parser.add_argument("--runs-root", type=Path, default=DEFAULT_RUNS_ROOT)
    parser.add_argument("--toplevel-root", type=Path, default=DEFAULT_TOPLEVEL_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--threshold", type=float, default=None,
                        help="Single decision threshold for all new runs "
                             "(default: 0.5, canonical output)")
    parser.add_argument("--threshold-file", type=Path, default=None,
                        help="val_thresholds.csv from select_val_thresholds.py; "
                             "each new run scored at its per-seed tau*_iou")
    parser.add_argument("--out-suffix", default=None,
                        help="Suffix for the calibrated output CSV "
                             "(default: _tauval for --threshold-file, "
                             "_thr<t> for --threshold)")
    args = parser.parse_args()

    threshold_for, calibrated = make_threshold_resolver(args.threshold, args.threshold_file)

    if not args.nas_root.exists():
        raise SystemExit(f"NAS root not found: {args.nas_root}\nIs the external volume mounted?")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    audit_checkpoint_provenance(args.runs_root)

    n_runs = len(OLD_RUNS) + len(NEW_RUNS) + len(SWIN_LARGE_DIRS)
    print(f"Evaluating {n_runs} model-runs x {len(DATES_5)} dates\n")
    per_date = evaluate(args.nas_root, args.runs_root, args.toplevel_root,
                        threshold_for=threshold_for, calibrated=calibrated)
    if calibrated:
        suffix = args.out_suffix or (
            "_tauval" if args.threshold_file is not None
            else f"_thr{args.threshold:g}")
        out_csv = args.out_dir / f"per_date_metrics_5_dates{suffix}.csv"
        print("(calibrated mode: 3-date consistency check skipped; "
              "thresholds differ from the canonical table)")
    else:
        check_against_3date_table(per_date, args.out_dir)
        out_csv = args.out_dir / "per_date_metrics_5_dates.csv"
    per_date.to_csv(out_csv, index=False)
    print(f"\nWrote {out_csv} ({len(per_date)} rows)")


if __name__ == "__main__":
    main()
