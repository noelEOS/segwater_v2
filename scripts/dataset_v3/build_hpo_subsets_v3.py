#!/usr/bin/env python3
"""HPO subsets and strata indices for dataset_v3.

Decision record: ``docs/dataset_v3/HPO_SUBSETS_V3.md``.

Two subcommands, split so that the part needing pandas runs anywhere the
memmap manifest is, and the part touching the 280 GB arrays needs numpy only.

``draw``   -- from ``dataset_v3_memmap_manifest.parquet``: a stratified-within-
              pair draw of ``--ratio`` of each (pair, composition class) group of
              the train and val splits, and the strata indices the training stack
              reads (``StratifiedWaterAccumulator.from_npz`` contract). Writes to
              ``--out-dir`` (git-tracked ``data/v3_strata/``):

              subset_indices.npz   train_idx / val_idx (sorted source rows),
                                   per-row expected label counts, seed, ratio, t
              hpo_val_strata.npz   N = len(val_idx); ``eligible`` baked at ≥5
              full_val_strata.npz  N = full val; ``min_mixed`` = 20, no ``eligible``
              subset_rows.parquet  pair_name, chip_id, split, src_row, dst_row, cls
              build_summary.json

``gather`` -- on the VM: ``{dst-root}/{train,val}.memmap`` as row gathers of the
              full memmaps, then a full verification pass: every destination row
              is compared bit-for-bit against its source row, and the label
              counts recomputed from channel 2 must equal the manifest's
              (carried inside the npz, so no parquet reader is needed there).
              Writes ``build_manifest.json`` beside the arrays with md5/sha256.

Composition class (``t`` = 0.01, the dataset_v3 docs rule): with
``ws = n_water / (n_water + n_land)``, pure land is ``ws <= t``, pure water is
``ws >= 1 - t``, mixed is strictly between. Stratum ids follow the accumulator:
pure-land 0, pure-water 1, mixed 2, all-ignore 3.

Draw rule: for every (pair, class) group of size n, ``q = min(n, max(1,
floor(ratio * n + 0.5)))`` rows are drawn without replacement with
``numpy.random.default_rng(seed)``; groups are visited in sorted key order, so
the draw is reproducible from (manifest, seed, ratio, t) alone. Because the
quota is fixed per group, a pair's mixed count in the subset is a property of
the rule, not of the seed, and a pair clearing ≥5 in a 10% subset holds ≥45
mixed chips on full val -- inside the 20-chip ladder gate by construction.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np

PURE_LAND, PURE_WATER, MIXED, ALL_IGNORE = 0, 1, 2, 3
CLS_NAME = {PURE_LAND: "pure_land", PURE_WATER: "pure_water", MIXED: "mixed", ALL_IGNORE: "all_ignore"}
CHANNELS = 3
H = W = 224
HPO_MIN_MIXED = 5
FULL_MIN_MIXED = 20


def stratum_ids(n_water: np.ndarray, n_land: np.ndarray, t: float) -> np.ndarray:
    valid = n_water.astype(np.int64) + n_land.astype(np.int64)
    out = np.full(len(valid), ALL_IGNORE, dtype=np.int8)
    ok = valid > 0
    ws = np.zeros(len(valid), dtype=np.float64)
    ws[ok] = n_water[ok] / valid[ok]
    out[ok & (ws <= t)] = PURE_LAND
    out[ok & (ws >= 1.0 - t)] = PURE_WATER
    out[ok & (ws > t) & (ws < 1.0 - t)] = MIXED
    return out


def strata_npz(pair_names_rows: np.ndarray, strata: np.ndarray, min_mixed: int, bake_eligible: bool) -> dict:
    """Dense pair ids in order of first appearance (memmap order)."""
    names, first_pos, pair_id = np.unique(pair_names_rows, return_index=True, return_inverse=True)
    order = np.argsort(first_pos)  # unique() sorts names; re-rank by first appearance
    rank = np.empty(len(order), dtype=np.int64)
    rank[order] = np.arange(len(order))
    pair_id = rank[pair_id].astype(np.int32)
    pair_names = names[order].astype(object)
    mixed_count = np.bincount(pair_id, weights=(strata == MIXED), minlength=len(pair_names)).astype(np.int32)
    out = {
        "stratum_id": strata.astype(np.int8),
        "pair_id": pair_id,
        "pair_names": pair_names,
        "pair_mixed_count": mixed_count,
        "N": np.int64(len(strata)),
        "min_mixed": np.int64(min_mixed),
    }
    if bake_eligible:
        out["eligible"] = mixed_count >= min_mixed
    return out


def draw(args: argparse.Namespace) -> int:
    import pandas as pd  # only this subcommand needs it

    t, ratio, seed = args.t, args.ratio, args.seed
    m = pd.read_parquet(args.manifest, columns=["pair_name", "chip_id", "split_v3", "row", "n_water", "n_land", "n_invalid"])
    m["cls"] = stratum_ids(m.n_water.to_numpy(), m.n_land.to_numpy(), t)
    rng = np.random.default_rng(seed)

    payload: dict = {"seed": np.int64(seed), "ratio": np.float64(ratio), "t": np.float64(t),
                     "mixed_rule": np.array(f"pure_land: ws<={t}; pure_water: ws>={1 - t}; mixed: between (dataset_v3 docs rule)")}
    rows_out = []
    summary: dict = {"seed": seed, "ratio": ratio, "t": t, "dtype": "float16",
                     "draw_rule": "per (pair, class): q = min(n, max(1, floor(ratio*n+0.5))), default_rng(seed), sorted group order",
                     "manifest": os.path.basename(args.manifest), "manifest_sha256": hashlib.sha256(Path(args.manifest).read_bytes()).hexdigest(),
                     "built_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")}

    for split in ("train", "val"):
        d = m[m.split_v3 == split].sort_values("row").reset_index(drop=True)
        assert (d.row.to_numpy() == np.arange(len(d))).all(), f"{split}: manifest rows are not 0..N-1"
        groups = d.groupby(["pair_name", "cls"]).indices
        keep = []
        for key in sorted(groups):
            idx = groups[key]
            n = len(idx)
            q = min(n, max(1, int(np.floor(ratio * n + 0.5))))
            keep.append(rng.choice(idx, size=q, replace=False))
        sel = np.sort(np.concatenate(keep)).astype(np.int64)
        assert len(np.unique(sel)) == len(sel)
        sub = d.iloc[sel]
        payload[f"{split}_idx"] = sel
        payload[f"{split}_counts"] = np.stack([sub.n_water.to_numpy(), sub.n_land.to_numpy(), sub.n_invalid.to_numpy()]).astype(np.int32)
        rows_out.append(pd.DataFrame({"pair_name": sub.pair_name.to_numpy(), "chip_id": sub.chip_id.to_numpy(), "split": split,
                                      "src_row": sel, "dst_row": np.arange(len(sel)), "cls": sub.cls.to_numpy()}))
        comp = {CLS_NAME[k]: int((sub.cls == k).sum()) for k in (PURE_LAND, MIXED, PURE_WATER, ALL_IGNORE)}
        comp_full = {CLS_NAME[k]: int((d.cls == k).sum()) for k in (PURE_LAND, MIXED, PURE_WATER, ALL_IGNORE)}
        summary[f"{split}_10pct"] = {"N": int(len(sel)), "pairs": int(sub.pair_name.nunique()), "composition": comp,
                                     "composition_share": {k: round(v / len(sel), 4) for k, v in comp.items()}}
        summary[f"{split}_full"] = {"N": int(len(d)), "pairs": int(d.pair_name.nunique()), "composition": comp_full,
                                    "composition_share": {k: round(v / len(d), 4) for k, v in comp_full.items()}}
        print(f"{split}: full N={len(d)} pairs={d.pair_name.nunique()} -> subset N={len(sel)} pairs={sub.pair_name.nunique()} {comp}")

    # strata indices (val only: the objective and the stage-2 monitor both read val)
    dval = m[m.split_v3 == "val"].sort_values("row").reset_index(drop=True)
    full = strata_npz(dval.pair_name.to_numpy(), dval.cls.to_numpy(), FULL_MIN_MIXED, bake_eligible=False)
    vsub = dval.iloc[payload["val_idx"]]
    hpo = strata_npz(vsub.pair_name.to_numpy(), vsub.cls.to_numpy(), HPO_MIN_MIXED, bake_eligible=True)

    # containment + tracking, asserted rather than reported
    full_elig = set(full["pair_names"][full["pair_mixed_count"] >= FULL_MIN_MIXED])
    hpo_elig = set(hpo["pair_names"][hpo["eligible"]])
    extra = sorted(hpo_elig - full_elig)
    assert not extra, f"subset-eligible pairs outside the full-val >={FULL_MIN_MIXED} set: {extra}"
    fm = dict(zip(full["pair_names"], full["pair_mixed_count"]))
    hm = dict(zip(hpo["pair_names"], hpo["pair_mixed_count"]))
    common = sorted(set(fm) & set(hm))
    r = float(np.corrcoef([fm[p] for p in common], [hm[p] for p in common])[0, 1])
    summary["full_val"] = {"N": int(full["N"]), "P": int(len(full["pair_names"])), "min_mixed": FULL_MIN_MIXED,
                           "eligible": int((full["pair_mixed_count"] >= FULL_MIN_MIXED).sum()),
                           "mixed_chips": int((dval.cls == MIXED).sum()),
                           "mixed_per_pair_median": float(np.median(full["pair_mixed_count"]))}
    summary["hpo_val"] = {"N": int(hpo["N"]), "P": int(len(hpo["pair_names"])), "min_mixed": HPO_MIN_MIXED,
                          "eligible": int(hpo["eligible"].sum()), "mixed_chips": int((vsub.cls == MIXED).sum()),
                          "eligible_pairs_mixed_share_of_full_val": round(float(sum(fm[p] for p in hpo_elig) / sum(fm.values())), 4),
                          "contained_in_full_val_eligible": True, "pearson_r_subset_vs_full_mixed_counts": round(r, 4)}
    print(f"full_val: P={summary['full_val']['P']} eligible(>={FULL_MIN_MIXED})={summary['full_val']['eligible']}")
    print(f"hpo_val : N={hpo['N']} P={summary['hpo_val']['P']} eligible(>={HPO_MIN_MIXED})={summary['hpo_val']['eligible']} "
          f"contained=True r={r:.4f} mixed-share={summary['hpo_val']['eligible_pairs_mixed_share_of_full_val']}")

    if not args.write:
        print("dry run; pass --write to emit artifacts")
        return 0
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "subset_indices.npz", **payload)
    np.savez(out / "hpo_val_strata.npz", **hpo)
    np.savez(out / "full_val_strata.npz", **full)
    pd.concat(rows_out, ignore_index=True).to_parquet(out / "subset_rows.parquet", index=False)
    (out / "build_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"wrote {out}/: subset_indices.npz hpo_val_strata.npz full_val_strata.npz subset_rows.parquet build_summary.json")
    return 0


def _hashes(path: Path, block: int = 64 << 20) -> dict:
    md5, sha = hashlib.md5(), hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(block):
            md5.update(chunk)
            sha.update(chunk)
    return {"md5": md5.hexdigest(), "sha256": sha.hexdigest()}


def gather(args: argparse.Namespace) -> int:
    split = args.split
    src_root, dst_root = Path(args.src_root).expanduser(), Path(args.dst_root).expanduser()
    bm = json.loads((src_root / "build_manifest.json").read_text())
    dtype = np.dtype(bm["dtype"])
    n_src = int(bm["splits"][split]["n_chips"])
    assert tuple(bm["shape_per_chip"]) == (CHANNELS, H, W), bm["shape_per_chip"]
    per_chip = CHANNELS * H * W * dtype.itemsize

    z = np.load(args.indices, allow_pickle=True)
    idx = z[f"{split}_idx"].astype(np.int64)
    counts = z[f"{split}_counts"]
    assert idx.ndim == 1 and len(idx) and (np.diff(idx) > 0).all(), "indices must be sorted and unique"
    assert idx[0] >= 0 and idx[-1] < n_src, f"index out of range for {split} (N={n_src})"
    assert counts.shape == (3, len(idx))

    src_path, dst_path = src_root / f"{split}.memmap", dst_root / f"{split}.memmap"
    assert src_path.stat().st_size == n_src * per_chip, f"{src_path}: size != N*per_chip"
    n_dst = len(idx)
    need = n_dst * per_chip
    print(f"{split}: gather {n_dst} of {n_src} rows -> {dst_path} ({need / 1e9:.1f} GB)")
    if not args.write:
        print("dry run; pass --write to build")
        return 0
    dst_root.mkdir(parents=True, exist_ok=True)
    if dst_path.exists():
        print(f"refusing to overwrite existing {dst_path}", file=sys.stderr)
        return 2
    free = shutil.disk_usage(dst_root).free
    if free < need + (5 << 30):
        print(f"not enough space: need {need / 1e9:.1f} GB + 5 GB headroom, have {free / 1e9:.1f} GB", file=sys.stderr)
        return 2

    src = np.memmap(src_path, dtype=dtype, mode="r", shape=(n_src, CHANNELS, H, W))
    dst = np.memmap(dst_path, dtype=dtype, mode="w+", shape=(n_dst, CHANNELS, H, W))
    chunk = args.chunk
    t0 = dt.datetime.now()
    for s in range(0, n_dst, chunk):
        e = min(s + chunk, n_dst)
        dst[s:e] = src[idx[s:e]]
        if (s // chunk) % 20 == 0:
            print(f"  wrote {e}/{n_dst}  {(dt.datetime.now() - t0).total_seconds():.0f}s", flush=True)
    dst.flush()
    del dst

    # verification: bit-exact rows + label-count identity, over every row
    dst = np.memmap(dst_path, dtype=dtype, mode="r", shape=(n_dst, CHANNELS, H, W))
    mismatch_rows = 0
    count_mismatch = 0
    for s in range(0, n_dst, chunk):
        e = min(s + chunk, n_dst)
        a = np.asarray(dst[s:e]).view(np.uint16)
        b = np.asarray(src[idx[s:e]]).view(np.uint16)
        bad = ~(a == b).reshape(e - s, -1).all(axis=1)
        mismatch_rows += int(bad.sum())
        lab = np.asarray(dst[s:e, 2]).astype(np.int32)
        nw = (lab == 1).sum(axis=(1, 2)); nl = (lab == 0).sum(axis=(1, 2)); ni = (lab == 255).sum(axis=(1, 2))
        exp = counts[:, s:e]
        count_mismatch += int(((nw != exp[0]) | (nl != exp[1]) | (ni != exp[2])).sum())
    ok = mismatch_rows == 0 and count_mismatch == 0
    print(f"verify: rows bit-exact mismatches={mismatch_rows}, label-count mismatches={count_mismatch} -> {'OK' if ok else 'FAIL'}")
    hashes = _hashes(dst_path)
    manifest_path = dst_root / "build_manifest.json"
    man = json.loads(manifest_path.read_text()) if manifest_path.exists() else {
        "dtype": str(dtype), "channels": bm["channels"], "shape_per_chip": bm["shape_per_chip"],
        "source_root": str(src_root), "indices": os.path.basename(args.indices),
        "row_contract": "row j of {split}.memmap = source row {split}_idx[j] of the full v3 memmap (indices sorted ascending)",
        "splits": {}}
    man["splits"][split] = {"n_chips": n_dst, "bytes": need, "source_n_chips": n_src, "verified_all_rows": ok,
                            "mismatch_rows": mismatch_rows, "label_count_mismatches": count_mismatch, **hashes,
                            "built_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")}
    manifest_path.write_text(json.dumps(man, indent=2) + "\n")
    print(f"md5 {hashes['md5']}  sha256 {hashes['sha256']}")
    if not ok:
        print("verification FAILED; the array is left in place for inspection but must not be used", file=sys.stderr)
        return 1
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    d = sp.add_parser("draw")
    d.add_argument("--manifest", required=True, help="dataset_v3_memmap_manifest.parquet")
    d.add_argument("--out-dir", default=str(Path(__file__).resolve().parents[2] / "data" / "v3_strata"))
    d.add_argument("--ratio", type=float, default=0.10)
    d.add_argument("--seed", type=int, default=42)
    d.add_argument("--t", type=float, default=0.01)
    d.add_argument("--write", action="store_true")
    d.set_defaults(fn=draw)
    g = sp.add_parser("gather")
    g.add_argument("--split", choices=["train", "val"], required=True)
    g.add_argument("--src-root", default="~/memmaps_v3")
    g.add_argument("--dst-root", default="~/memmaps_v3/hpo_10pct")
    g.add_argument("--indices", required=True, help="subset_indices.npz")
    g.add_argument("--chunk", type=int, default=256)
    g.add_argument("--write", action="store_true")
    g.set_defaults(fn=gather)
    args = ap.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
