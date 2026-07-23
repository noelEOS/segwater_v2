#!/usr/bin/env python3
"""
build_pair_based_memmap.py

Rebuild the train/val/test memmaps onto the PAIR-PURE split (`split_pair_based`)
by DETERMINISTIC REINDEX of the existing (leaky, chip-level) memmaps — no source
TIFFs, no recomputation. Chip values are copied byte-for-byte, so the output is
identical to what `create_memmap_multiprocessing_V2.py` would produce from the
same parquet on `split_pair_based`, but without re-reading a single raster.

WHY THIS IS CORRECT (see docs/roadmap_for_publication/DATASET_BUILD_LEDGER.md):
  The original build is order-deterministic:
      for split in [train,val,test]:
          sub = df[df['split'] == split].reset_index(drop=True)
          # memmap row i  <->  sub.iloc[i]
  So we can recover, for every SHIPPED memmap row, exactly which chip it is, by
  replaying that filter on the SAME parquet. Then we scatter each chip to its
  destination split according to `split_pair_based` (restricted to selected chips).

  Identity was verified four ways (VV histograms, clip co-movement, 16/16
  fingerprint discrimination, and 7 live EE S1_GRD fetches at VV corr >= 0.9992).
  As a belt-and-braces check, this script re-asserts the VV raw-value histogram
  of every copied row against the parquet's stored `raw_vv_counts`.

INPUTS
  --parquet  : the build parquet with BOTH `split` and `split_pair_based`
               (…-RESPLIT_PATHS_FIXED.parquet). Must also have
               `selected_for_memmap`, `raw_vv_bins`, `raw_vv_counts`.
  --src-root : dir with the SHIPPED train/val/test.memmap (chip-level `split`)
  --dst-root : output dir for the pair-pure memmaps

EXPECTED COUNTS (sanity — will assert):
  src (chip `split`, selected):  train 1,032,526 / val 294,643 / test 146,878
  dst (`split_pair_based`, sel): train 1,032,004 / val 291,771 / test 150,272
  both total 1,474,047.

USAGE
  python build_pair_based_memmap.py \
      --parquet .../…-RESPLIT_PATHS_FIXED.parquet \
      --src-root  .../memmap_datasets_v2/memmap_datasets_v2 \
      --dst-root  .../memmap_datasets_v2_pairbased \
      [--verify-every 1]   # 1 = check every row's VV histogram (default); N = every Nth
      [--limit-verify 0]   # 0 = no cap; else stop histogram checks after N rows/split
"""
import os
import argparse
import logging

import numpy as np
import pandas as pd

# S1 normalization constants — MUST match create_memmap_multiprocessing_V2.py
VV_STATS = (-15.510920639155525, 6.564080466988128)  # (mean, std), dB
CHIP_PIXELS = 224 * 224
CHANNELS = 3  # [VV_norm, VH_norm, water_mask]
H = W = 224

SPLITS = ["train", "val", "test"]

# Expected per-split counts (assert to catch a wrong parquet immediately).
EXPECT_SRC = {"train": 1032526, "val": 294643, "test": 146878}   # chip-level `split`, selected
EXPECT_DST = {"train": 1032004, "val": 291771, "test": 150272}   # split_pair_based, selected


def parse_hist(s):
    """The raw_* histogram columns are stored as bracketed strings."""
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return None
    return np.array([float(x) for x in str(s).replace("[", "").replace("]", "").replace(",", " ").split()])


def vv_hist_from_memmap(vv_norm, bins):
    """Invert z-score to dB and histogram over the parquet's bin edges."""
    db = vv_norm.astype(np.float64) * VV_STATS[1] + VV_STATS[0]
    counts, _ = np.histogram(db.ravel(), bins=bins)
    return counts.astype(np.int64)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--src-root", required=True, help="dir with shipped {train,val,test}.memmap")
    ap.add_argument("--dst-root", required=True)
    ap.add_argument("--verify-every", type=int, default=1,
                    help="assert VV histogram on every Nth copied row (1=all, 0=never)")
    ap.add_argument("--limit-verify", type=int, default=0,
                    help="stop histogram checks after this many rows per split (0=no cap)")
    ap.add_argument("--hist-tol", type=int, default=250,
                    help="max allowed L1 between memmap and parquet VV histograms "
                         "(edge-binning + float round-trip noise; empirically <150)")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    os.makedirs(args.dst_root, exist_ok=True)

    logging.info(f"Loading parquet: {args.parquet}")
    need = ["pair_name", "chip_id", "split", "split_pair_based",
            "selected_for_memmap", "raw_vv_bins", "raw_vv_counts"]
    df = pd.read_parquet(args.parquet, columns=need)

    # ---- 1. Reconstruct the SHIPPED (source) row order, per split ----
    # Source rows = df[df['split']==s].reset_index(drop=True); row i <-> src.iloc[i].
    # We give every source row a global position so we can open the src memmap by index.
    src_frames = {}
    for s in SPLITS:
        sub = df[df["split"] == s].reset_index(drop=True)
        assert len(sub) == EXPECT_SRC[s], \
            f"SRC {s}: got {len(sub)} rows, expected {EXPECT_SRC[s]} — wrong parquet or wrong `split` column"
        sub = sub.copy()
        sub["src_row"] = np.arange(len(sub), dtype=np.int64)
        src_frames[s] = sub
        logging.info(f"src '{s}': {len(sub):,} rows (matches expected)")

    # A single table of every selected chip with (src_split, src_row) and dst split.
    # Every selected chip appears exactly once across the src frames (chip-level `split`
    # partitions the selected set). We key on the unique (pair_name, chip_id).
    src_all = pd.concat(src_frames.values(), ignore_index=True)
    # dst assignment: split_pair_based, restricted to selected chips.
    sel = df[df["selected_for_memmap"] == True].copy()  # noqa: E712
    for s in SPLITS:
        assert int((sel["split_pair_based"] == s).sum()) == EXPECT_DST[s], \
            f"DST {s}: split_pair_based selected count mismatch"

    # Map (pair_name, chip_id) -> (src_split, src_row). chip_id is only unique WITHIN a pair,
    # so the key MUST be the pair+chip composite, never chip_id alone.
    src_all["key"] = list(zip(src_all["pair_name"], src_all["chip_id"]))
    key_to_src = dict(zip(src_all["key"], zip(src_all["split"], src_all["src_row"])))
    assert len(key_to_src) == len(src_all), "duplicate (pair_name, chip_id) keys in source — cannot reindex"

    # ---- 2. Open source memmaps read-only ----
    src_mm = {}
    for s in SPLITS:
        p = os.path.join(args.src_root, f"{s}.memmap")
        n = EXPECT_SRC[s]
        assert os.path.getsize(p) == n * CHANNELS * H * W * 4, f"{p}: size != {n} chips"
        src_mm[s] = np.memmap(p, dtype=np.float32, mode="r", shape=(n, CHANNELS, H, W))
        logging.info(f"opened src memmap '{s}' ({n:,} chips)")

    # ---- 3. Build each destination split ----
    for dst_split in SPLITS:
        dst_sub = sel[sel["split_pair_based"] == dst_split].reset_index(drop=True)
        n = len(dst_sub)
        out_path = os.path.join(args.dst_root, f"{dst_split}.memmap")
        dst = np.memmap(out_path, dtype=np.float32, mode="w+", shape=(n, CHANNELS, H, W))
        logging.info(f"writing dst '{dst_split}': {n:,} chips -> {out_path}")

        verified = zeros = 0
        for i in range(n):
            row = dst_sub.iloc[i]
            key = (row["pair_name"], row["chip_id"])
            ssplit, srow = key_to_src[key]
            chip = src_mm[ssplit][srow]                 # (3,224,224) float32, byte-identical payload
            dst[i] = chip                               # copy VV, VH, mask verbatim

            # zero-row detection (swallowed-error chips in the original build)
            if not np.any(chip[0]) and not np.any(chip[1]):
                zeros += 1
                logging.warning(f"dst '{dst_split}' row {i} (pair {key[0]} chip {key[1]}): ZERO-FILLED source chip")

            # per-row VV-histogram assertion
            do_check = args.verify_every and (i % args.verify_every == 0) \
                and (args.limit_verify == 0 or verified < args.limit_verify)
            if do_check:
                bins = parse_hist(row["raw_vv_bins"])
                exp = parse_hist(row["raw_vv_counts"])
                if bins is not None and exp is not None:
                    got = vv_hist_from_memmap(chip[0], bins)
                    l1 = int(np.abs(got - exp.astype(np.int64)).sum())
                    if l1 > args.hist_tol:
                        raise AssertionError(
                            f"dst '{dst_split}' row {i} (pair {key[0]} chip {key[1]}): "
                            f"VV histogram L1={l1} > tol {args.hist_tol} — identity check FAILED")
                    verified += 1

            if (i + 1) % 100000 == 0:
                logging.info(f"  {dst_split}: {i+1:,}/{n:,} copied, {verified:,} hist-verified")

        dst.flush()
        assert n == EXPECT_DST[dst_split], f"dst '{dst_split}': wrote {n}, expected {EXPECT_DST[dst_split]}"
        logging.info(f"finished '{dst_split}': {n:,} chips, {verified:,} VV-hist-verified, {zeros} zero-rows")

    # ---- 4. Pair-purity assertion on the output partition ----
    chk = sel.groupby("pair_name")["split_pair_based"].nunique()
    spanning = int((chk > 1).sum())
    assert spanning == 0, f"OUTPUT NOT PAIR-PURE: {spanning} pairs span splits"
    logging.info(f"PAIR-PURE confirmed: 0 of {chk.size:,} pairs span splits.")
    logging.info("Done. Point configs/data/memmap.yaml::memmap_root at --dst-root to train on the clean split.")


if __name__ == "__main__":
    main()
