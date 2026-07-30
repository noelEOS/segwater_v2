"""Census parquet -> strata_index.npz for the checkpoint-ladder eval.

The per-chip census (val_chip_stats.parquet, one row per val-memmap chip in
memmap row order) carries each chip's stratum {pure-land, pure-water, mixed}
and its pair. The stratified ladder eval needs those two labels as compact
per-chip arrays aligned to the val loader's (shuffle=False) iteration order.

Encoding:
  stratum_id  int8[N]   pure-land=0, pure-water=1, mixed=2
  pair_id     int32[N]  dense 0..P-1, ordered by first appearance in row order
  pair_names  str[P]    pair_id -> original PAIR_* name
  pair_mixed_count int32[P]  #mixed chips per pair (for the >=20 gate downstream)
  N           int       291771 (asserted)

Usage (Mac eda env):
    python scripts/evaluation/build_strata_index.py \
        --parquet <scratchpad>/val_chip_stats.parquet \
        --out data/full_val_strata/strata_index.npz
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

STRATUM_MAP = {"pure-land": 0, "pure-water": 1, "mixed": 2}
EXPECTED_N = 291771


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    df = pd.read_parquet(args.parquet)
    n = len(df)
    if n != EXPECTED_N:
        raise SystemExit(f"Row count {n} != expected {EXPECTED_N}")

    # Census rows are already in memmap order; do NOT sort. Verify strata vocab.
    unknown = set(df["stratum"].unique()) - set(STRATUM_MAP)
    if unknown:
        raise SystemExit(f"Unknown stratum values: {unknown}")

    stratum_id = df["stratum"].map(STRATUM_MAP).to_numpy(np.int8)

    # Dense pair ids in first-appearance (row) order so pair_names[pair_id] is
    # stable and human-legible.
    pair_names_series = df["pair_name"].astype("string")
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

    # Per-pair mixed-chip counts.
    mixed_mask = stratum_id == 2
    pair_mixed_count = np.zeros(p, dtype=np.int32)
    np.add.at(pair_mixed_count, pair_id[mixed_mask], 1)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.out,
        stratum_id=stratum_id,
        pair_id=pair_id,
        pair_names=pair_names,
        pair_mixed_count=pair_mixed_count,
        N=np.int64(n),
    )

    # Report.
    counts = {k: int((stratum_id == v).sum()) for k, v in STRATUM_MAP.items()}
    n_ge20 = int((pair_mixed_count >= 20).sum())
    mixed_in_ge20 = int(pair_mixed_count[pair_mixed_count >= 20].sum())
    print(f"Wrote {args.out}")
    print(f"  N={n}  pairs={p}")
    print(f"  strata: {counts}")
    print(f"  pairs with >=20 mixed chips: {n_ge20}  "
          f"({mixed_in_ge20}/{int(mixed_mask.sum())} = "
          f"{100*mixed_in_ge20/int(mixed_mask.sum()):.1f}% of mixed chips)")


if __name__ == "__main__":
    main()
