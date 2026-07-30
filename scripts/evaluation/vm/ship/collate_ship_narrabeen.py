"""Collate the per-arm Narrabeen SDS sweep outputs into one long CSV.

One row per (arm x stride x threshold) = 6 x 3 x 9 = 162 for a 6-arm campaign.
Column schema matches the Swin-B `ship` campaign's `sds_narrabeen_ship_msl.csv`
exactly, so the two campaigns' SDS tables are directly comparable and poolable
*provided* `arch` and `lineage` are carried -- which is why they are columns here
and not left to the filename.

Reads only `sweep_metrics.csv` (produced by
`SDS_Benchmark_slim/scripts/sds/run_sds_from_rasters.py`); computes no shoreline
metrics of its own, so a discrepancy here means a discrepancy upstream.

WHY THIS IS TRACKED
-------------------
The Swin-B campaign's equivalent script lived only in its results directory,
which is gitignored -- so that leg was one `rm -rf` from unreproducible.
`RUNBOOK_multiagent_vm_campaign.md` §9: commit the toolchain, not just results.

GUARDS
  * run dirs resolved via `runsel.resolve_run_dir` (UTC-stamp anchored), never a
    prefix glob -- this campaign shares a runs root with ~200 other dirs;
  * `n_shorelines` must equal the staged scene count (87 at Narrabeen). SDS
    silently ignores scenes it cannot pair with a survey, so a mis-staged input
    dir never announces itself; `n_shorelines` counts rasters FED IN;
  * every (arm, stride) cell must be present and carry all 9 thresholds -- a
    short table that looks complete is the failure mode being prevented;
  * `--strict` additionally requires the full expected row count.

⚠️ `n` (matched transect points) is NOT the scorable scene count. At Narrabeen 87
scenes are staged and 82 are scorable at the default 10-day tolerance; `n` is a
third, larger quantity. Report the two scene counts, never conflate them with `n`.

Usage:
    python scripts/evaluation/vm/ship/collate_ship_narrabeen.py \\
        --raw-dir ~/workspace/results/ship_decision_cnxb_2026-07/narrabeen/raw \\
        --tag cnxb --arch cnxb --lineage mx630s2cnx \\
        --out .../narrabeen/sds_narrabeen_cnxb_msl.csv --strict
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from completion import expected_scenes  # noqa: E402
from naming import atomic_to_csv  # noqa: E402
from runsel import RunDirError, resolve_run_dir  # noqa: E402

SEEDS = ["s19", "s42", "s58"]
VARIANTS = ["best", "last"]
STRIDES = [112, 32, 8]
RUNS = Path.home() / "segwater_v2/outputs/inference/runs"
N_THRESHOLDS = 9


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw-dir", type=Path, required=True,
                    help="dir holding one <run_dir_name>_sweep subdir per cell")
    ap.add_argument("--tag", required=True, help="campaign tag, e.g. cnxb")
    ap.add_argument("--arch", required=True,
                    help="architecture slug for the `arch` column, e.g. cnxb")
    ap.add_argument("--lineage", required=True,
                    help="training-lineage slug for the `lineage` column. MUST "
                         "differ between lineages whose checkpoint filenames "
                         "collide (e.g. mx630s2 vs mx630s2cnx)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--variants", nargs="+", default=VARIANTS,
                    choices=["best", "last", "swa5"])
    ap.add_argument("--site", default="NARRABEEN")
    ap.add_argument("--reference", default="MSL")
    ap.add_argument("--strict", action="store_true",
                    help="also require the full expected row count (for the "
                         "FINAL collation pass)")
    a = ap.parse_args()

    n_staged = expected_scenes("narrabeen")
    rows, problems = [], []

    for seed in SEEDS:
        for variant in a.variants:
            sweep = "narrabeen_%s_%s_%s" % (a.tag, seed, variant)
            for stride in STRIDES:
                cell = "%s/%s/s%d" % (seed, variant, stride)
                # One sweep name -> three dirs (one per stride), so disambiguate
                # with runsel's own `suffix=`. Still stamp-anchored; still raises
                # rather than guessing when the match is not unique.
                try:
                    run_dir = resolve_run_dir(RUNS, sweep,
                                              suffix="_b0_s%d" % stride)
                except RunDirError as exc:
                    problems.append("%s: %s" % (cell, str(exc).splitlines()[0]))
                    continue
                m = a.raw_dir / (run_dir.name + "_sweep") / "sweep_metrics.csv"
                if not m.exists():
                    problems.append("%s: missing %s" % (cell, m))
                    continue
                df = pd.read_csv(m)
                if len(df) != N_THRESHOLDS:
                    problems.append("%s: %d thresholds, expected %d"
                                    % (cell, len(df), N_THRESHOLDS))
                bad = df[df.n_shorelines != n_staged]
                if len(bad):
                    problems.append(
                        "%s: n_shorelines %s != %d staged — a mis-staged input dir "
                        "does not announce itself"
                        % (cell, sorted(bad.n_shorelines.unique()), n_staged))
                df = df.assign(
                    arch=a.arch, lineage=a.lineage, seed=seed, variant=variant,
                    arm="%s_%s_%s_%s" % (a.arch, seed, a.lineage, variant),
                    stride=stride, site=a.site, reference=a.reference,
                    run_dir=run_dir.name)
                rows.append(df)

    if rows:
        out = pd.concat(rows, ignore_index=True)[[
            "arch", "lineage", "seed", "variant", "arm", "stride", "threshold",
            "n", "n_shorelines", "rmse", "bias", "std", "q90", "R2",
            "site", "reference", "run_dir"]].sort_values(
                ["seed", "variant", "stride", "threshold"])
        a.out.parent.mkdir(parents=True, exist_ok=True)
        atomic_to_csv(out, a.out, index=False)
        n_cells = len(out.groupby(["seed", "variant", "stride"]))
        print("wrote %s (%d rows, %d arm x stride cells)"
              % (a.out, len(out), n_cells))
        print("scene counts: %d staged per run dir; `n` is matched transect "
              "points, NOT a scene count" % n_staged)

    n_expected = len(SEEDS) * len(a.variants) * len(STRIDES) * N_THRESHOLDS
    if a.strict and (not rows or len(out) != n_expected):
        problems.append("got %d rows, expected %d (--strict)"
                        % (len(out) if rows else 0, n_expected))
    if problems:
        print("\n=== %d PROBLEM(S) ===" % len(problems))
        for p in problems:
            print("  " + p)
        raise SystemExit(1)
    print("=== ALL NARRABEEN COLLATION CHECKS PASSED ===")


if __name__ == "__main__":
    main()
