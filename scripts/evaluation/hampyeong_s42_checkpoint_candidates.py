"""Per-date Hampyeong metrics for alternative Swin-Base seed-42 checkpoints.

Seed 42's shipped `best.pth` was picked by the historical filename rule (now
preserved in scripts/evaluation/ensure_best_ckpts.py): the 4-decimal mIoU in the
checkpoint FILENAME, breaking ties by max step. For Swin-Base s42 two
checkpoints tie at that precision (step38400 and step39600, both miou0.9506), so the
shipped pick is decided by the tiebreak rather than by the full-float validation mIoU.
This script scores the alternative checkpoints' inference runs on the same pixels and
through the same code path as the audited comparison, so the shipped s42 run and the
candidates can be compared directly.

Only the three legacy dates are scored: the candidate runs contain those scenes only.

The audited `audit_checkpoint_provenance` is deliberately not called on the candidate
runs -- it asserts every checkpoint path is `.../s{seed}/best.pth`, which is exactly
what these runs are NOT. A candidate-specific provenance check is done here instead:
architecture, encoder and seed must match, the checkpoint must not be best.pth, and
config/summary/manifest must agree.

Run with an interpreter that has rasterio, e.g.
    /opt/homebrew/Caskroom/miniforge/base/envs/eda/bin/python
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
import yaml

_SCRIPTS = Path(__file__).resolve().parent.parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from evaluation.metrics import METRIC_NAMES, compute_binary_metrics  # noqa: E402
from evaluation.hampyeong_model_comparison import (  # noqa: E402
    CKPT_KEY_BY_ARCH,
    DATES,
    DEFAULT_NAS_ROOT,
    DEFAULT_RUNS_ROOT,
    EXPECTED_VALID_PIXELS,
    NEW_RUNS,
    gt_path,
    load_pair,
    new_pred_path,
    valid_mask_path,
)

DEFAULT_CANDIDATE_ROOT = Path("/Users/noel/code/tmp/hampyeong_s42_other_checkpoints")
DEFAULT_OUT_DIR = Path("experiments/hampyeong/evaluation")

ARCH = "Swin-B"
SEED = 42
# The shipped s42 run, as registered in the audited comparison script.
SHIPPED_RUN_DIR = dict((seed, run_dir) for _m, arch, seed, run_dir in NEW_RUNS if arch == ARCH)[SEED]


def _require(path: Path, what: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"{what} not found: {path}")
    return path


def discover_candidates(candidate_root: Path) -> list[tuple[str, Path, str]]:
    """Return (label, run_dir, checkpoint_path) for each candidate run directory."""
    out: list[tuple[str, Path, str]] = []
    for run_dir in sorted(p for p in candidate_root.iterdir() if p.is_dir()):
        summary = json.loads((run_dir / "run_summary.json").read_text())
        ckpt = summary["checkpoint_path"]
        step = re.search(r"_step(\d+)_", ckpt)
        if not step:
            raise AssertionError(f"cannot parse step from checkpoint {ckpt}")
        out.append((f"swin_b_s{SEED}_step{step.group(1)}", run_dir, ckpt))
    if not out:
        raise SystemExit(f"no candidate run directories under {candidate_root}")
    return out


def audit_candidate(run_dir: Path, checkpoint: str) -> None:
    """Provenance check tailored to non-best.pth candidate runs."""
    config = yaml.safe_load((run_dir / "run_config.yaml").read_text())
    summary = json.loads((run_dir / "run_summary.json").read_text())

    cfg_ckpt = config["inference"]["checkpoint_path"]
    if cfg_ckpt != summary["checkpoint_path"] or cfg_ckpt != checkpoint:
        raise AssertionError(f"{run_dir.name}: config/summary checkpoints disagree")

    manifest = pd.read_csv(run_dir / "run_manifest.csv")
    manifest_ckpts = set(manifest["checkpoint_path"].unique())
    if manifest_ckpts != {cfg_ckpt}:
        raise AssertionError(f"{run_dir.name}: manifest checkpoints {manifest_ckpts} != config {cfg_ckpt}")

    key = CKPT_KEY_BY_ARCH[ARCH]
    if summary["model_encoder"] != key.split("upernet_", 1)[-1] or summary["model_architecture"] != "upernet":
        raise AssertionError(f"{run_dir.name}: not a {ARCH} run (encoder={summary['model_encoder']})")
    if f"outputs/stage2/{key}/s{SEED}/" not in cfg_ckpt:
        raise AssertionError(f"{run_dir.name}: checkpoint {cfg_ckpt} is not {key} seed {SEED}")
    if cfg_ckpt.endswith("/best.pth"):
        raise AssertionError(f"{run_dir.name}: this is the shipped best.pth, not an alternative candidate")


def evaluate(nas_root: Path, runs_root: Path, candidates: list[tuple[str, Path, str]]) -> pd.DataFrame:
    mask = _require(valid_mask_path(nas_root), "valid mask")
    rows: list[dict] = []

    for date in DATES:
        reference = _require(gt_path(nas_root, date), f"ground truth {date}")
        y_true_reference: np.ndarray | None = None
        digests: dict[str, str] = {}

        specs: list[tuple[str, str, Path]] = [
            (f"swin_b_s{SEED}_best", f"outputs/stage2/{CKPT_KEY_BY_ARCH[ARCH]}/s{SEED}/best.pth",
             _require(new_pred_path(runs_root, SHIPPED_RUN_DIR, date), f"shipped s{SEED} {date}")),
        ]
        for label, run_dir, ckpt in candidates:
            scene = run_dir / _scene_stem(run_dir, date)
            specs.append((label, ckpt, _require(scene, f"{label} {date}")))

        for label, ckpt, prediction in specs:
            y_true, y_pred = load_pair(reference, prediction, mask)

            digest = hashlib.sha1(y_pred.tobytes()).hexdigest()
            if digest in digests:
                print(f"  NOTE {date}: {label} is pixel-identical to {digests[digest]} at threshold 0.5")
            else:
                digests[digest] = label

            if y_true_reference is None:
                y_true_reference = y_true
            elif not np.array_equal(y_true, y_true_reference):
                raise AssertionError(f"y_true differs across runs on {date} (run={label})")

            metrics = compute_binary_metrics(y_true, y_pred, include_counts=True)
            counts_total = metrics["tn"] + metrics["fp"] + metrics["fn"] + metrics["tp"]
            if counts_total != EXPECTED_VALID_PIXELS:
                raise AssertionError(f"confusion counts sum to {counts_total}, expected {EXPECTED_VALID_PIXELS}")

            rows.append({
                "date": date,
                "run": label,
                "arch": ARCH,
                "seed": SEED,
                "checkpoint": ckpt,
                **{m: metrics[m] for m in METRIC_NAMES},
                **{c: metrics[c] for c in ("tn", "fp", "fn", "tp")},
                "n_valid": counts_total,
                "water_fraction_gt": float(y_true.mean()),
                "prediction_path": str(prediction),
            })
            print(f"  {date}  {label:26s} IoU={metrics['iou']:.4f}  F1={metrics['f1']:.4f}  MCC={metrics['mcc']:.4f}")

    return pd.DataFrame(rows)


def _scene_stem(run_dir: Path, date: str) -> str:
    from evaluation.hampyeong_model_comparison import SCENE_IDS
    scene = SCENE_IDS[date]
    return f"{scene}/{scene}_probability_water.tif"


def check_shipped_against_table(df: pd.DataFrame, out_dir: Path) -> None:
    """The shipped-s42 rows must reproduce per_date_metrics_3_dates.csv exactly."""
    ref_csv = out_dir / "per_date_metrics_3_dates.csv"
    if not ref_csv.exists():
        print("(per_date_metrics_3_dates.csv not found; skipping consistency check)")
        return
    ref = pd.read_csv(ref_csv, dtype={"date": str})
    ref = ref[ref["model"] == f"swin_b_s{SEED}"]
    got = df[df["run"] == f"swin_b_s{SEED}_best"]
    merged = ref.merge(got, on="date", suffixes=("_ref", "_new"))
    if len(merged) != len(ref):
        raise AssertionError(f"row mismatch vs 3-date table: {len(merged)} joined, {len(ref)} expected")
    for m in list(METRIC_NAMES) + ["tn", "fp", "fn", "tp"]:
        if not np.allclose(merged[f"{m}_ref"], merged[f"{m}_new"], rtol=0, atol=1e-12):
            raise AssertionError(f"shipped-s42 metric {m} differs from per_date_metrics_3_dates.csv")
    print(f"\nConsistency check passed: shipped s{SEED} reproduces per_date_metrics_3_dates.csv exactly "
          f"({len(ref)} rows)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--nas-root", type=Path, default=Path(DEFAULT_NAS_ROOT))
    parser.add_argument("--runs-root", type=Path, default=Path(DEFAULT_RUNS_ROOT))
    parser.add_argument("--candidate-root", type=Path, default=DEFAULT_CANDIDATE_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    if not args.nas_root.exists():
        raise SystemExit(f"NAS root not found: {args.nas_root}\nIs the external volume mounted?")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    candidates = discover_candidates(args.candidate_root)
    for label, run_dir, ckpt in candidates:
        audit_candidate(run_dir, ckpt)
        print(f"Candidate {label}: {ckpt}")
    print(f"Shipped   swin_b_s{SEED}_best: {SHIPPED_RUN_DIR}\n")

    per_date = evaluate(args.nas_root, args.runs_root, candidates)
    check_shipped_against_table(per_date, args.out_dir)

    out_csv = args.out_dir / "swin_b_s42_checkpoint_candidates_3_dates.csv"
    per_date.to_csv(out_csv, index=False)
    print(f"\nWrote {out_csv} ({len(per_date)} rows)")

    print("\nMean over the three dates:")
    summary = per_date.groupby("run")[list(METRIC_NAMES)].mean()
    print(summary.to_string())


if __name__ == "__main__":
    main()
