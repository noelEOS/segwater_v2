"""HPO-subset census -> hpo_val_pair_strata.npz for the pair-macro HPO objective.

Distinct from build_strata_index.py: that consumes the full-val chip-stats
parquet, whose `stratum` column is precomputed. The HPO-subset census carries
raw pixel counts per chip (n_land, n_water, n_ignore, n_boundary_edges), so the
stratum must be DERIVED here with the exact census rule
(analyze_hpo.py:43-46, purity threshold 0.1%).

Row i of pair_val_census.npy / pair_val_meta.parquet is HPO val memmap row i
(both in memmap order), so the emitted per-chip arrays align to the val loader's
shuffle=False iteration order.

Encoding:
  stratum_id  int8[N]   pure-land=0, pure-water=1, mixed=2, all-ignore=3
  pair_id     int32[N]  dense 0..P-1, ordered by first appearance in row order
  pair_names  str[P]    pair_id -> original PAIR_* name
  pair_mixed_count int32[P]  #mixed chips per pair
  eligible    bool[P]   pair_mixed_count >= min_mixed (>=5 -> 554 pairs)
  min_mixed   int64
  N           int64     29177 (asserted)

Usage (Mac eda env):
    python scripts/evaluation/build_hpo_strata_index.py \
        --out data/hpo_strata/hpo_val_pair_strata.npz
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# Purity rule mirrors analyze_hpo.py exactly: minority class <= 0.1% of valid px.
PURE = 0.001
# Stratum ids. all-ignore is defensively handled (count is 0 today) so the
# accumulator's strat_cm has a dedicated, always-empty row 3.
PURE_LAND, PURE_WATER, MIXED, ALL_IGNORE = 0, 1, 2, 3

EXPECTED_N = 29177
EXPECTED_PAIRS = 686
EXPECTED_MIXED = 11027
EXPECTED_ELIGIBLE = 554  # at min_mixed=5

_HERE = Path(__file__).resolve()
_CENSUS_DIR = (
    _HERE.parents[2]
    / "docs/roadmap_for_publication/dataset_build_evidence/hpo_subset_census/data"
)
DEFAULT_CENSUS = _CENSUS_DIR / "pair_val_census.npy"
DEFAULT_META = _CENSUS_DIR / "pair_val_meta.parquet"
DEFAULT_OUT = _HERE.parents[2] / "data/hpo_strata/hpo_val_pair_strata.npz"


def build_index(census_path, meta_path, min_mixed=5, expected_n=EXPECTED_N):
    """Derive per-chip strata + dense pair ids from the HPO-subset census.

    Returns (arrays, stats): `arrays` is the dict of npz payload arrays,
    `stats` a small dict of the reported invariants. Raises on N mismatch.
    """
    census = np.load(census_path).astype(np.int64)  # [N,4] n_land,n_water,n_ignore,n_bedge
    n = len(census)
    if n != expected_n:
        raise ValueError(f"Census row count {n} != expected {expected_n}")

    meta = pd.read_parquet(meta_path)
    if len(meta) != n:
        raise ValueError(f"Meta rows {len(meta)} != census rows {n}")

    n_land = census[:, 0]
    n_water = census[:, 1]
    valid = n_land + n_water
    # wf = water fraction of valid px; NaN where all-ignore (valid==0).
    with np.errstate(invalid="ignore", divide="ignore"):
        wf = np.where(valid > 0, n_water / valid, np.nan)

    # np.select order matches analyze_hpo.py; default = mixed.
    stratum_id = np.select(
        [valid == 0, wf >= 1 - PURE, wf <= PURE],
        [ALL_IGNORE, PURE_WATER, PURE_LAND],
        default=MIXED,
    ).astype(np.int8)

    # Dense pair ids in first-appearance (row) order so pair_names[pair_id] is
    # stable and human-legible (copied from build_strata_index.py:50-62).
    pair_names_series = meta["pair_name"].astype("string")
    seen = {}
    order = []
    pair_id = np.empty(n, dtype=np.int32)
    for i, name in enumerate(pair_names_series.to_numpy()):
        pid = seen.get(name)
        if pid is None:
            pid = len(order)
            seen[name] = pid
            order.append(name)
        pair_id[i] = pid
    pair_names = np.array(order, dtype=object)
    p = len(order)

    mixed_mask = stratum_id == MIXED
    pair_mixed_count = np.zeros(p, dtype=np.int32)
    np.add.at(pair_mixed_count, pair_id[mixed_mask], 1)
    eligible = pair_mixed_count >= min_mixed

    n_all_ignore = int((stratum_id == ALL_IGNORE).sum())
    stats = {
        "N": n,
        "pairs": p,
        "pure_land": int((stratum_id == PURE_LAND).sum()),
        "pure_water": int((stratum_id == PURE_WATER).sum()),
        "mixed": int(mixed_mask.sum()),
        "all_ignore": n_all_ignore,
        "eligible": int(eligible.sum()),
        "min_mixed": int(min_mixed),
    }

    arrays = {
        "stratum_id": stratum_id,
        "pair_id": pair_id,
        "pair_names": pair_names,
        "pair_mixed_count": pair_mixed_count,
        "eligible": eligible,
        "min_mixed": np.int64(min_mixed),
        "N": np.int64(n),
    }
    return arrays, stats


def _assert_invariants(stats):
    problems = []
    if stats["N"] != EXPECTED_N:
        problems.append(f"N={stats['N']} != {EXPECTED_N}")
    if stats["pairs"] != EXPECTED_PAIRS:
        problems.append(f"pairs={stats['pairs']} != {EXPECTED_PAIRS}")
    if stats["mixed"] != EXPECTED_MIXED:
        problems.append(f"mixed={stats['mixed']} != {EXPECTED_MIXED}")
    if stats["min_mixed"] == 5 and stats["eligible"] != EXPECTED_ELIGIBLE:
        problems.append(f"eligible={stats['eligible']} != {EXPECTED_ELIGIBLE}")
    if problems:
        raise SystemExit("Invariant check FAILED: " + "; ".join(problems))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--census", type=Path, default=DEFAULT_CENSUS)
    ap.add_argument("--meta", type=Path, default=DEFAULT_META)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--min-mixed", type=int, default=5)
    ap.add_argument("--expected-n", type=int, default=EXPECTED_N)
    args = ap.parse_args()

    arrays, stats = build_index(
        args.census, args.meta, min_mixed=args.min_mixed, expected_n=args.expected_n
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, **arrays)

    print(f"Wrote {args.out}")
    print(f"  N={stats['N']}  pairs={stats['pairs']}")
    print(
        f"  strata: pure-land={stats['pure_land']} pure-water={stats['pure_water']} "
        f"mixed={stats['mixed']} all-ignore={stats['all_ignore']}"
    )
    print(
        f"  eligible pairs (>= {stats['min_mixed']} mixed): {stats['eligible']} "
        f"of {stats['pairs']}"
    )
    _assert_invariants(stats)
    print("  invariants OK: N=29177, pairs=686, mixed=11027, eligible=554")


if __name__ == "__main__":
    main()
