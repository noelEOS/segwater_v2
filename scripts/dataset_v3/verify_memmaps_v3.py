#!/usr/bin/env python3
"""Verify the v3 memmaps against the rasters, the manifest and the row index.

Independent of ``build_memmaps_v3.py`` by construction: it re-reads the bytes
on disk and re-cuts sample chips from the rasters with its OWN window and
decode code, sharing only ``paths.py`` with the builder. Importing the
builder's cutting functions would make this a re-run of the same code, not a
verification -- the duplication below is deliberate.

Checks:

1. **sizes** -- each file is exactly ``n * 3 * 224 * 224 * itemsize``.
2. **round-trip identity** (sampled) -- z-score a fresh cut of the same chip
   from the S1 raster and compare against the stored fp16 values, in z-score
   space, against the fp16 rounding bound: |stored - fresh| must not exceed
   ``2**-11 * max(|fresh|, 1)`` (half the fp16 spacing at the value's own
   magnitude, with a 25% slack factor). The bound must scale with magnitude:
   fp16 error is relative, and comparing in dB against a flat tolerance
   mistakes ordinary quantization for corruption once multiplied by a ~9 dB
   std (the first run of this verifier made exactly that mistake).
3. **MATCH/DECOY** -- every sampled comparison is also run against a
   deliberately wrong chip (another chip of the same pair, or the next row)
   and must FAIL. A verifier whose decoys pass is comparing something to
   itself; a passing decoy is a hard failure of the verifier, not the data.
4. **mask plane** -- values are a subset of {0, 1, 255} and the per-chip
   counts equal the manifest's ``n_water/n_land/n_invalid`` exactly (fp16 is
   exact on these integers, so no tolerance).
5. **train moments** -- sampled band mean/std near 0/1 under the v3 constants.
   Exact 0/1 is NOT expected: the constants come from binned moments. Fail
   outside mean +-0.15 or std outside (0.85, 1.15).
6. **zero rows** -- no chip with both bands all zero.
7. **row-index integrity** -- the index is a permutation per split, splits are
   disjoint, the union is the assigned corpus, and scene purity holds --
   re-established here from the index alone, independently of apply_splits.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import from_bounds

import paths

H = W = 224
CHANNELS = 3
CHIP_PIXELS = H * W
SPLITS = ("train", "val", "test")
EXPECTED_ASSIGNED = 1_322_788

# Independent copies of the encoding facts (see the docstring on why).
FLOOR_DB = -55.0
STEP_DB = 0.05
# fp16 half-spacing is 2**-11 relative to the value's magnitude; 1.25 is slack
# for the float64 round-trip arithmetic. Applied as
# tol = FP16_HALF_ULP_REL * max(|z|, 1), elementwise.
FP16_HALF_ULP_REL = 1.25 * 2.0 ** -11


def s1_raster(pair: str):
    matches = sorted(paths.RAW.glob("sentinel1_GRD/s1_q55s005_%s_*.tif" % pair))
    if len(matches) != 1:
        raise FileNotFoundError("expected one S1 raster for %s, found %d"
                                % (pair, len(matches)))
    return matches[0]


def cut_db(src, box) -> np.ndarray:
    """(2, 224, 224) float64 dB for one chip, windowed read, own code path."""
    window = from_bounds(*box, transform=src.transform).round_offsets().round_lengths()
    codes = src.read((1, 2), window=window)
    if codes.shape != (2, H, W):
        raise ValueError("window read shape %r" % (codes.shape,))
    if (codes == 0).any():
        raise ValueError("nodata code in a supposedly clean chip")
    return codes.astype(np.float64) * STEP_DB + FLOOR_DB


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dst-root", required=True)
    parser.add_argument("--dtype", choices=["float16", "float32"], default="float16")
    parser.add_argument("--sample", type=int, default=20_000,
                        help="chips per split for the raster round-trip")
    parser.add_argument("--full", action="store_true", help="round-trip every chip (slow)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--write", action="store_true",
                        help="write verify_report_v3.json next to the memmaps")
    args = parser.parse_args()
    dtype = np.dtype(args.dtype)
    rng = np.random.default_rng(args.seed)

    index = pd.read_parquet(
        paths.require(paths.MANIFESTS / "memmap_row_index_v3.parquet",
                      "row index; run build_memmaps_v3.py first"))
    manifest = pd.read_parquet(
        paths.require(paths.MANIFESTS / "dataset_v3_manifest.parquet", "v3 manifest"),
        columns=["pair_name", "chip_id", "bbox_w", "bbox_s", "bbox_e", "bbox_n",
                 "n_water", "n_land", "n_invalid"])
    norm = json.loads(paths.require(paths.MANIFESTS / "norm_constants_v3.json",
                                    "norm constants").read_text())
    mean = np.array([norm["vv"]["mean"], norm["vh"]["mean"]]).reshape(2, 1, 1)
    std = np.array([norm["vv"]["std"], norm["vh"]["std"]]).reshape(2, 1, 1)

    # --- 7. row-index integrity, before any byte is read ---
    if len(index) != EXPECTED_ASSIGNED:
        raise AssertionError("row index has %d chips, expected %d"
                             % (len(index), EXPECTED_ASSIGNED))
    if index.duplicated(["pair_name", "chip_id"]).any():
        raise AssertionError("row index repeats a (pair_name, chip_id)")
    spans = index.groupby("pair_name")["split_v3"].nunique()
    if int(spans.max()) != 1:
        raise AssertionError("scene purity broken in the row index")
    for split in SPLITS:
        rows = index.loc[index["split_v3"] == split, "row"].to_numpy()
        if not (np.sort(rows) == np.arange(len(rows))).all():
            raise AssertionError("%s: rows are not a permutation of 0..%d"
                                 % (split, len(rows) - 1))
    print("row index OK: %s chips, splits disjoint, scene-pure, permutation rows"
          % "{:,}".format(len(index)))

    joined = index.merge(manifest, on=["pair_name", "chip_id"], validate="1:1")
    if len(joined) != len(index):
        raise AssertionError("manifest join changed the row count")

    failures: list[tuple] = []
    report = {"checks": {}, "dtype": args.dtype,
              "sampled_per_split": None if args.full else args.sample}

    for split in SPLITS:
        sub = joined[joined["split_v3"] == split].sort_values("row").reset_index(drop=True)
        n = len(sub)
        path = os.path.join(args.dst_root, "%s.memmap" % split)
        want = n * CHANNELS * H * W * dtype.itemsize
        got = os.path.getsize(paths.require(Path(path), "%s memmap" % split))
        if got != want:
            raise AssertionError("%s: %d bytes, expected %d" % (path, got, want))
        mm = np.memmap(path, dtype=dtype, mode="r", shape=(n, CHANNELS, H, W))
        print("=== %s: %s chips, %.1f GB ===" % (split, "{:,}".format(n), got / 1e9))

        take = n if args.full else min(args.sample, n)
        picked = np.sort(rng.choice(n, size=take, replace=False))
        sample = sub.iloc[picked]

        worst = 0.0
        match_n = decoy_n = decoy_passed = 0
        zero_rows = 0
        mask_bad = 0
        band_sum = np.zeros(2)
        band_sq = np.zeros(2)
        band_px = 0

        for pair, group in sample.groupby("pair_name", sort=True):
            with rasterio.open(s1_raster(pair)) as src:
                boxes = group[["bbox_w", "bbox_s", "bbox_e", "bbox_n"]].to_numpy()
                for k, (_, row) in enumerate(group.iterrows()):
                    stored = np.asarray(mm[int(row["row"])], dtype=np.float64)

                    if not stored[:2].any():
                        zero_rows += 1
                    # 4. mask counts, exact.
                    lab = stored[2]
                    vals = np.unique(lab)
                    if not np.isin(vals, (0.0, 1.0, 255.0)).all():
                        mask_bad += 1
                    counts = (int((lab == 1.0).sum()), int((lab == 0.0).sum()),
                              int((lab == 255.0).sum()))
                    if counts != (row["n_water"], row["n_land"], row["n_invalid"]):
                        failures.append((split, int(row["row"]), pair,
                                         int(row["chip_id"]), "mask_counts"))

                    # 2. round-trip against a fresh, independent cut, compared
                    # in z-score space against the magnitude-scaled fp16 bound.
                    # norm_err <= 1 means "within fp16 rounding of the truth".
                    fresh_z = (cut_db(src, tuple(boxes[k])) - mean) / std
                    bound = FP16_HALF_ULP_REL * np.maximum(np.abs(fresh_z), 1.0)
                    norm_err = float(np.max(np.abs(stored[:2] - fresh_z) / bound))
                    worst = max(worst, norm_err)
                    match_n += 1
                    if norm_err > 1.0:
                        failures.append((split, int(row["row"]), pair,
                                         int(row["chip_id"]),
                                         "roundtrip %.2f x fp16 bound" % norm_err))

                    # 3. decoy: the same stored bytes vs a WRONG chip's cut.
                    others = [j for j in range(len(group)) if j != k]
                    if others:
                        wrong_z = (cut_db(src, tuple(boxes[others[0]])) - mean) / std
                        wbound = FP16_HALF_ULP_REL * np.maximum(np.abs(wrong_z), 1.0)
                        decoy_err = float(np.max(np.abs(stored[:2] - wrong_z) / wbound))
                        decoy_n += 1
                        if decoy_err <= 1.0:
                            decoy_passed += 1
                            failures.append((split, int(row["row"]), pair,
                                             int(row["chip_id"]), "DECOY PASSED"))

                    band_sum += stored[:2].sum(axis=(1, 2))
                    band_sq += (stored[:2] ** 2).sum(axis=(1, 2))
                    band_px += CHIP_PIXELS

        print("  round-trip max normalized err %.3f x fp16 bound (must be <= 1) "
              "over %s chips" % (worst, "{:,}".format(match_n)))
        print("  MATCH %d / DECOY %d (passed decoys: %d -- must be 0)"
              % (match_n, decoy_n, decoy_passed))
        if mask_bad:
            failures.append((split, -1, "mask_values", "", mask_bad))
        print("  mask values OK on sample: %s | zero rows: %d"
              % ("yes" if not mask_bad else "NO", zero_rows))
        if zero_rows:
            failures.append((split, -1, "zero_rows", "", zero_rows))

        if split == "train":
            m_ = band_sum / band_px
            s_ = np.sqrt(band_sq / band_px - m_ ** 2)
            print("  sampled train moments: VV %+0.4f/%.4f  VH %+0.4f/%.4f"
                  % (m_[0], s_[0], m_[1], s_[1]))
            print("    (expect ~0/~1 within the binned-moment margin; exact 0/1 is "
                  "NOT expected)")
            for i, band in enumerate(("VV", "VH")):
                if abs(m_[i]) > 0.15 or not (0.85 < s_[i] < 1.15):
                    failures.append((split, -1, "%s_moments" % band, "", float(m_[i])))

        report["checks"][split] = {
            "n_chips": n, "sampled": int(take),
            "roundtrip_max_err_vs_fp16_bound": worst,
            "match": match_n, "decoy": decoy_n, "decoy_passed": decoy_passed,
            "zero_rows": zero_rows,
        }
        del mm

    if failures:
        print("\n%d FAILURES; first 10:" % len(failures))
        for f in failures[:10]:
            print("  %r" % (f,))
        return 1

    print("\nALL CHECKS PASSED")
    if args.write:
        report["ok"] = True
        report["verified_at"] = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
        out = os.path.join(args.dst_root, "verify_report_v3.json")
        with open(out, "w") as fh:
            json.dump(report, fh, indent=2)
        print("wrote %s" % out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
