#!/usr/bin/env python3
"""Build the v3 fp16 memmaps by cutting every chip from the pair rasters.

Output: ``{train,val,test}.memmap`` under ``--dst-root``, each
``(N, 3, 224, 224)`` in ``--dtype`` (default float16), channels
``[VV_norm, VH_norm, label]``; plus ``build_manifest.json`` beside them and
``manifests/memmap_row_index_v3.parquet``.

**Row-order contract.** Row *i* of ``{split}.memmap`` is the *i*-th row of
``manifest[manifest.split_v3 == split]`` in manifest positional order. The
contract is materialized as ``memmap_row_index_v3.parquet`` (``pair_name,
chip_id, split_v3, row, chip_origin``); every consumer joins through that file
and none re-derives the ordering.

**One code path for both populations.** The legacy memmaps are not on this VM
and are not read: existing chips (1.26 M, already cut once into the old
arrays) and recovered chips (66 k, never cut) are both cut fresh from the
per-pair rasters, by identical code -- which is what makes the two populations
comparable. ``legacy_*_row`` on the manifest stays what it is: verification
metadata, indexed by nothing here.

**S1 decode.** The rasters are the re-quantized encoding
``dB = code * 0.05 - 55``, code 0 = nodata (``s1_q55s005_*``). The decode is
affine -- ``10*log10`` belongs to the legacy linear-x1e4 encoding and must
never appear here. Passing chips are ``s1_status == 'clean'``, so a zero code
anywhere is a contradiction between the gate and the raster and fails the
pair. Bands are z-scored in float32 with the constants from
``derive_norm_constants.py`` and cast to fp16 exactly once, at assignment;
channel 2 carries the raw label {0, 1, 255}, which fp16 represents exactly,
and is never rescaled (rescaling would silently break ``ignore_index=255``).

**Windows are geometry, never pixel indices**, and the S1 and label rasters
sit on different grid extents per pair: each raster gets its own
``from_bounds`` window, with the sub-pixel residual measured against
``invalid_mask.MAX_SUBPIXEL_RESIDUAL`` before rounding.

**The two identity checks that prove the build** (both per chip, both exact):

* the recomputed label counts ``(n_water, n_land, n_invalid)`` from the cut
  block must equal the manifest's values -- proving the right window of the
  right variant raster landed in channel 2;
* the recomputed 320-bin S1 histogram (via ``build_histograms.histogram_codes``,
  imported, not reimplemented) must equal ``chip_s1_histograms.parquet`` --
  proving the right S1 pixels landed in channels 0-1. Both sides are exact
  bincounts over the same uint16 codes, so equality is ``==``, not a
  tolerance.

Workers write disjoint row ranges of the shared memmap (``mode='r+'``);
disjointness holds because rows are assigned by the manifest ordering and
asserted to be a permutation before any write. Errors are returned, not
raised; any error fails the run after the sweep rather than shipping a
partially-zero memmap silently (the v1 builder's per-chip ``except: pass`` is
the defect this refuses to repeat).
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import shutil
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import rasterio
from rasterio.windows import from_bounds

import paths
from build_histograms import FLOOR_DB, STEP_DB, N_BINS, histogram_codes, s1_path
from count_label_classes import label_path
from invalid_mask import CHIP_PIXELS, MAX_SUBPIXEL_RESIDUAL, window_residual

H = W = 224
CHANNELS = 3
SPLITS = ("train", "val", "test")
LAND, WATER, INVALID = 0, 1, 255

EXPECTED_ASSIGNED = 1_322_788
# fp16 has an 11-bit significand: relative quantization error <= 2**-11.
FP16_REL_TOL = 2.0 ** -11


def cut_pair(pair: str, split: str, chips: pd.DataFrame, dst_path: str,
             n_split: int, dtype_name: str, norm: dict,
             check_hists: bool) -> dict:
    """Cut, normalize and write every chip of one pair. Errors returned, not raised."""
    try:
        dtype = np.dtype(dtype_name)
        with rasterio.open(s1_path(pair)) as src:
            vv_full = src.read(1)
            vh_full = src.read(2)
            s1_transform = src.transform

        dst = np.memmap(dst_path, dtype=dtype, mode="r+",
                        shape=(n_split, CHANNELS, H, W))
        stats = {"pair": pair, "error": "", "n": len(chips), "max_rel_err": 0.0,
                 "chip_id": chips["chip_id"].to_numpy(),
                 "vv_hist": np.zeros((len(chips), N_BINS), dtype=np.uint32) if check_hists else None,
                 "vh_hist": np.zeros((len(chips), N_BINS), dtype=np.uint32) if check_hists else None}
        mean = np.array([norm["vv"]["mean"], norm["vh"]["mean"]],
                        dtype=np.float32).reshape(2, 1, 1)
        std = np.array([norm["vv"]["std"], norm["vh"]["std"]],
                       dtype=np.float32).reshape(2, 1, 1)

        # A pair's chips can span both label variants; open each raster once.
        for variant, group in chips.groupby("v3_label_variant", sort=True):
            with rasterio.open(label_path(pair, variant)) as lsrc:
                label_transform = lsrc.transform
                label_full = lsrc.read(1)

            for pos, row in zip(group.index, group.itertuples()):
                box = (row.bbox_w, row.bbox_s, row.bbox_e, row.bbox_n)

                s1_win = from_bounds(*box, transform=s1_transform)
                if window_residual(s1_win) > MAX_SUBPIXEL_RESIDUAL:
                    raise ValueError("%s chip %d: S1 window off-lattice by %g"
                                     % (pair, row.chip_id, window_residual(s1_win)))
                s1_win = s1_win.round_offsets().round_lengths()
                r0, c0 = int(s1_win.row_off), int(s1_win.col_off)
                sl = (slice(r0, r0 + H), slice(c0, c0 + W))
                vv = vv_full[sl]
                vh = vh_full[sl]
                if vv.size != CHIP_PIXELS or vh.size != CHIP_PIXELS:
                    raise ValueError("%s chip %d: S1 read %d/%d px"
                                     % (pair, row.chip_id, vv.size, vh.size))
                if (vv == 0).any() or (vh == 0).any():
                    raise ValueError("%s chip %d: nodata code in a clean chip"
                                     % (pair, row.chip_id))

                lab_win = from_bounds(*box, transform=label_transform)
                if window_residual(lab_win) > MAX_SUBPIXEL_RESIDUAL:
                    raise ValueError("%s chip %d: label window off-lattice by %g"
                                     % (pair, row.chip_id, window_residual(lab_win)))
                lab_win = lab_win.round_offsets().round_lengths()
                lr, lc = int(lab_win.row_off), int(lab_win.col_off)
                label = label_full[lr:lr + H, lc:lc + W]
                if label.size != CHIP_PIXELS:
                    raise ValueError("%s chip %d: label read %d px"
                                     % (pair, row.chip_id, label.size))

                n_water = int((label == WATER).sum())
                n_land = int((label == LAND).sum())
                n_invalid = int((label == INVALID).sum())
                if (n_water, n_land, n_invalid) != (row.n_water, row.n_land, row.n_invalid):
                    raise ValueError(
                        "%s chip %d: label counts (%d,%d,%d) != manifest (%d,%d,%d)"
                        % (pair, row.chip_id, n_water, n_land, n_invalid,
                           row.n_water, row.n_land, row.n_invalid))

                if check_hists:
                    at = chips.index.get_loc(pos)
                    stats["vv_hist"][at], _, _ = histogram_codes(vv)
                    stats["vh_hist"][at], _, _ = histogram_codes(vh)

                db = np.stack([vv, vh]).astype(np.float32) * np.float32(STEP_DB) \
                    + np.float32(FLOOR_DB)
                normed = (db - mean) / std
                out = np.empty((CHANNELS, H, W), dtype=dtype)
                out[:2] = normed.astype(dtype, copy=False)   # quantize once, on write
                out[2] = label.astype(dtype, copy=False)     # {0,1,255}: exact in fp16

                back = out[:2].astype(np.float32)
                denom = np.maximum(np.abs(normed), 1e-3)
                rel = float(np.max(np.abs(back - normed) / denom))
                stats["max_rel_err"] = max(stats["max_rel_err"], rel)

                dst[row.dst_row] = out

        dst.flush()
        del dst
        return stats
    except Exception as error:  # noqa: BLE001 - one bad pair must not kill the run
        return {"pair": pair, "error": "%s: %s" % (type(error).__name__, str(error)[:300]),
                "n": len(chips)}


def git_sha() -> str:
    try:
        return subprocess.run(["git", "-C", str(paths.REPO), "rev-parse", "HEAD"],
                              capture_output=True, text=True, timeout=10,
                              check=True).stdout.strip()
    except Exception:  # noqa: BLE001 - the VM copy is not a git checkout
        return "unknown"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dst-root", required=True,
                        help="directory for {train,val,test}.memmap + build_manifest.json")
    parser.add_argument("--dtype", choices=["float16", "float32"], default="float16")
    parser.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    parser.add_argument("--limit", type=int, default=None,
                        help="first N pairs only (smoke test; memmaps are still "
                             "created full-size but sparse)")
    parser.add_argument("--skip-histogram-check", action="store_true",
                        help="skip the per-chip S1 histogram identity check "
                             "(ON by default; it is the S1 identity proof)")
    parser.add_argument("--dry-run", action="store_true",
                        help="validate, report sizes and space, write nothing")
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    dtype = np.dtype(args.dtype)

    manifest = pd.read_parquet(
        paths.require(paths.MANIFESTS / "dataset_v3_manifest.parquet", "v3 manifest"),
        columns=["pair_name", "chip_id", "split_v3", "chip_origin",
                 "bbox_w", "bbox_s", "bbox_e", "bbox_n", "v3_label_variant",
                 "n_water", "n_land", "n_invalid"])
    if "split_v3" not in manifest.columns or manifest["split_v3"].isna().all():
        raise AssertionError("split_v3 missing or empty; run apply_splits.py first")

    assigned = manifest[manifest["split_v3"].notna()].reset_index(drop=True)
    if args.limit is None and len(assigned) != EXPECTED_ASSIGNED:
        raise AssertionError("assigned chips %d, expected %d"
                             % (len(assigned), EXPECTED_ASSIGNED))

    # The row-order contract, materialized. Positional order within each split.
    assigned["dst_row"] = -1
    n_per_split = {}
    for split in SPLITS:
        mask = assigned["split_v3"] == split
        n = int(mask.sum())
        n_per_split[split] = n
        assigned.loc[mask, "dst_row"] = np.arange(n, dtype=np.int64)
        rows = assigned.loc[mask, "dst_row"].to_numpy()
        if not (np.sort(rows) == np.arange(n)).all():
            raise AssertionError("%s: dst_row is not a permutation of 0..%d" % (split, n - 1))
    print("splits: %s" % {s: "{:,}".format(n) for s, n in n_per_split.items()})

    norm = json.loads(
        paths.require(paths.MANIFESTS / "norm_constants_v3.json",
                      "norm constants; run derive_norm_constants.py first").read_text())

    check_hists = not args.skip_histogram_check
    hist_lookup = None
    if check_hists:
        table = pq.read_table(
            paths.require(paths.MANIFESTS / "chip_s1_histograms.parquet", "S1 histograms"),
            columns=["pair_name", "chip_id", "vv_counts", "vh_counts"])
        hist = table.to_pandas()
        hist_lookup = hist.set_index(["pair_name", "chip_id"])
        print("histogram identity check: ON (%s reference chips)" % "{:,}".format(len(hist)))

    total_bytes = sum(n * CHANNELS * H * W * dtype.itemsize for n in n_per_split.values())
    usage = shutil.disk_usage(os.path.dirname(os.path.abspath(args.dst_root)) or ".")
    print("destination %s: need %.1f GB, free %.1f GB"
          % (args.dst_root, total_bytes / 1e9, usage.free / 1e9))
    if usage.free < total_bytes + 10e9:
        raise AssertionError("insufficient free space: need %.1f GB + 10 GB margin, "
                             "have %.1f GB" % (total_bytes / 1e9, usage.free / 1e9))

    if args.dry_run:
        for split in SPLITS:
            print("  %s: %s chips -> %.1f GB %s"
                  % (split, "{:,}".format(n_per_split[split]),
                     n_per_split[split] * CHANNELS * H * W * dtype.itemsize / 1e9,
                     args.dtype))
        print("\n(dry run; nothing written)")
        return 0
    if not args.write:
        print("\n(pass --write to build, or --dry-run for sizes only)")
        return 0

    os.makedirs(args.dst_root, exist_ok=True)
    dst_paths = {}
    for split in SPLITS:
        path = os.path.join(args.dst_root, "%s.memmap" % split)
        mm = np.memmap(path, dtype=dtype, mode="w+",
                       shape=(n_per_split[split], CHANNELS, H, W))
        del mm
        dst_paths[split] = path

    groups = list(assigned.groupby("pair_name", sort=True))
    if args.limit:
        groups = groups[:args.limit]
    print("cutting %s chips across %d pairs with %d jobs"
          % ("{:,}".format(sum(len(g) for _, g in groups)), len(groups), args.jobs))

    errors, hist_mismatches = [], []
    max_rel_err = 0.0
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        futures = []
        for pair, group in groups:
            split = group["split_v3"].iloc[0]
            if group["split_v3"].nunique() != 1:
                raise AssertionError("%s spans splits; scene purity is broken" % pair)
            futures.append(pool.submit(
                cut_pair, pair, split, group.reset_index(drop=True),
                dst_paths[split], n_per_split[split], args.dtype, norm, check_hists))
        for done, future in enumerate(as_completed(futures), 1):
            result = future.result()
            if result["error"]:
                errors.append(result)
                continue
            max_rel_err = max(max_rel_err, result["max_rel_err"])
            if check_hists:
                pair = result["pair"]
                for i, chip_id in enumerate(result["chip_id"]):
                    ref = hist_lookup.loc[(pair, chip_id)]
                    if not (np.array_equal(result["vv_hist"][i], np.asarray(ref["vv_counts"]))
                            and np.array_equal(result["vh_hist"][i], np.asarray(ref["vh_counts"]))):
                        hist_mismatches.append((pair, int(chip_id)))
            if done % 250 == 0:
                print("  %d/%d pairs" % (done, len(groups)), flush=True)

    if errors:
        print("\nERRORS on %d pairs, e.g. %s" % (len(errors), errors[0]["error"]))
    if hist_mismatches:
        print("HISTOGRAM MISMATCHES on %d chips, e.g. %s"
              % (len(hist_mismatches), hist_mismatches[0]))
    print("max fp16 relative quantization error: %.2e (bound %.2e)"
          % (max_rel_err, FP16_REL_TOL))

    for split in SPLITS:
        want = n_per_split[split] * CHANNELS * H * W * dtype.itemsize
        got = os.path.getsize(dst_paths[split])
        if got != want:
            raise AssertionError("%s: %d bytes on disk, expected %d"
                                 % (dst_paths[split], got, want))

    failed = bool(errors or hist_mismatches) or \
        (args.limit is None and max_rel_err > 4 * FP16_REL_TOL)
    if failed:
        raise AssertionError("build FAILED: %d pair errors, %d histogram mismatches"
                             % (len(errors), len(hist_mismatches)))

    paths.ensure_out()
    index_target = paths.MANIFESTS / "memmap_row_index_v3.parquet"
    assigned[["pair_name", "chip_id", "split_v3", "dst_row", "chip_origin"]] \
        .rename(columns={"dst_row": "row"}) \
        .to_parquet(index_target, index=False, compression="zstd")

    build_manifest = {
        "dtype": args.dtype,
        "channels": ["vv_norm", "vh_norm", "label"],
        "shape_per_chip": [CHANNELS, H, W],
        "splits": {s: {"n_chips": n_per_split[s],
                       "bytes": n_per_split[s] * CHANNELS * H * W * dtype.itemsize,
                       "origin": assigned.loc[assigned["split_v3"] == s, "chip_origin"]
                                         .value_counts().to_dict()}
                   for s in SPLITS},
        "norm_constants": norm,
        "s1_encoding": "dB = code * %g + %g; code 0 = nodata (s1_q55s005 rasters)"
                       % (STEP_DB, FLOOR_DB),
        "label_encoding": "channel 2 raw {0,1,255}; never rescaled",
        "row_order": "row i of {split}.memmap = i-th row of "
                     "manifest[manifest.split_v3 == split] in manifest positional "
                     "order; materialized in memmap_row_index_v3.parquet",
        "label_variants": assigned["v3_label_variant"].value_counts().to_dict(),
        "max_fp16_rel_err": max_rel_err,
        "limit": args.limit,
        "git_sha": git_sha(),
        "built": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
    }
    manifest_path = os.path.join(args.dst_root, "build_manifest.json")
    with open(manifest_path, "w") as fh:
        json.dump(build_manifest, fh, indent=2)
    print("\nwrote %s and %s" % (manifest_path, index_target))
    print("Next: run verify_memmaps_v3.py before training on these.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
