#!/usr/bin/env python3
"""Per-chip S1 backscatter histograms, on the re-quantized dB encoding.

One histogram per band per chip, over chips whose S1 is intact
(``s1_status == 'clean'``). Damaged and fill-bearing chips are excluded: a
histogram of pixels the export lost would be a histogram of the defect.

**Bins are dB, computed from codes.** The encoding is ``dB = code*0.05 - 55``,
which is affine, so a 0.25 dB bin is exactly 5 codes. The counts are therefore
made with ``np.bincount`` over the uint16 codes and summed in groups of five --
exact, deterministic, and with no float bin edges to fall between. The schema is
labelled in dB rather than codes because dB survives a re-encoding: if the
offset or step ever change, the same columns keep the same meaning, whereas
code-indexed bins would be orphaned.

Edges are ``-55 + 0.25*k`` for ``k`` in 0..320, stored once in the parquet
metadata rather than per row.

Two rules this follows that the previous generation's histograms did not:

* **Nothing is dropped silently.** The old script used
  ``np.histogram(range=...)``, which discards out-of-range pixels without
  recording it -- the mechanism behind the "6.6% of VH lost at the -60 bound"
  finding. Here every pixel is accounted for and
  ``sum(counts) + n_nodata == 50,176`` is asserted per chip per band.
* **Nodata is excluded, label-255 is not.** Code 0 is absence of observation and
  must not be binned as a value (it would compute to exactly -55.0 dB and pile
  into the floor bin). But label-invalid pixels ARE fed to the network -- label
  invalidity masks the loss, not the input -- so excluding them would bias the
  normalization constants, and bias them precisely in the intertidal regime
  where SCL false cloud concentrates.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import rasterio
from rasterio.windows import from_bounds

import paths

CHIP_PIXELS = 224 * 224

# The re-quantized encoding: dB = code * STEP_DB + FLOOR_DB, code 0 = nodata.
FLOOR_DB = -55.0
STEP_DB = 0.05

# Histogram grid, in dB. 0.25 dB is exactly CODES_PER_BIN codes.
BIN_WIDTH_DB = 0.25
TOP_DB = 25.0
N_BINS = int(round((TOP_DB - FLOOR_DB) / BIN_WIDTH_DB))      # 320
CODES_PER_BIN = int(round(BIN_WIDTH_DB / STEP_DB))           # 5
MAX_CODE = N_BINS * CODES_PER_BIN                            # 1600 -> +25 dB

EXPECTED_CLEAN = 1_355_702


def histogram_codes(block: np.ndarray) -> tuple[np.ndarray, int, int]:
    """(counts over N_BINS dB bins, n_nodata, n_above_top) for one chip band."""
    flat = block.ravel()
    n_nodata = int((flat == 0).sum())

    # Codes 1..MAX_CODE map into the bins; anything above folds into the top bin
    # and is counted separately so the fold is visible rather than silent.
    above = flat > MAX_CODE
    n_above = int(above.sum())

    counted = np.where(above, MAX_CODE, flat)
    # bincount over codes, then sum groups of CODES_PER_BIN. Index 0 is nodata
    # and is dropped here, having been counted above.
    per_code = np.bincount(counted, minlength=MAX_CODE + 1)[1:MAX_CODE + 1]
    counts = per_code.reshape(N_BINS, CODES_PER_BIN).sum(axis=1).astype(np.uint32)
    return counts, n_nodata, n_above


def s1_path(pair: str):
    matches = sorted(paths.RAW.glob("sentinel1_GRD/s1_q55s005_%s_*.tif" % pair))
    if len(matches) != 1:
        raise FileNotFoundError("expected one S1 raster for %s, found %d"
                                % (pair, len(matches)))
    return matches[0]


def histogram_pair(pair: str, chip_ids: np.ndarray, boxes: np.ndarray) -> dict:
    """Histogram every clean chip of one pair. Errors are returned, not raised."""
    try:
        with rasterio.open(s1_path(pair)) as src:
            vv_full = src.read(1)
            vh_full = src.read(2)
            transform = src.transform

        n = len(chip_ids)
        vv_counts = np.zeros((n, N_BINS), dtype=np.uint32)
        vh_counts = np.zeros((n, N_BINS), dtype=np.uint32)
        meta = {k: np.zeros(n, dtype=np.int32)
                for k in ("n_nodata_vv", "n_nodata_vh", "n_above_vv", "n_above_vh")}

        for i, (west, south, east, north) in enumerate(boxes):
            window = from_bounds(west, south, east, north,
                                 transform=transform).round_offsets().round_lengths()
            r0, c0 = int(window.row_off), int(window.col_off)
            sl = (slice(r0, r0 + int(window.height)), slice(c0, c0 + int(window.width)))
            for band, counts, tag in ((vv_full, vv_counts, "vv"), (vh_full, vh_counts, "vh")):
                block = band[sl]
                if block.size != CHIP_PIXELS:
                    raise ValueError("%s chip %d read %d px, expected %d"
                                     % (pair, chip_ids[i], block.size, CHIP_PIXELS))
                c, nod, above = histogram_codes(block)
                counts[i] = c
                meta["n_nodata_" + tag][i] = nod
                meta["n_above_" + tag][i] = above

        # Every pixel accounted for, per chip per band.
        for tag, counts in (("vv", vv_counts), ("vh", vh_counts)):
            total = counts.sum(axis=1).astype(np.int64) + meta["n_nodata_" + tag]
            if not (total == CHIP_PIXELS).all():
                raise AssertionError("%s %s: counts + nodata != %d" % (pair, tag, CHIP_PIXELS))

        return {"pair": pair, "chip_id": chip_ids, "vv": vv_counts, "vh": vh_counts,
                "error": "", **meta}
    except Exception as error:  # noqa: BLE001 - one bad pair must not kill the run
        return {"pair": pair, "error": "%s: %s" % (type(error).__name__, str(error)[:200])}


def summarise(frame: pd.DataFrame, vv: np.ndarray, vh: np.ndarray) -> None:
    edges = FLOOR_DB + BIN_WIDTH_DB * np.arange(N_BINS + 1)
    centres = (edges[:-1] + edges[1:]) / 2

    print("HISTOGRAMS  %d chips x %d bins x 2 bands" % (len(frame), N_BINS))
    print("  bins: %.2f .. %.2f dB, width %.2f (= %d codes)"
          % (edges[0], edges[-1], BIN_WIDTH_DB, CODES_PER_BIN))
    print()
    for tag, counts in (("VV", vv), ("VH", vh)):
        total = counts.sum(axis=0).astype(np.float64)
        share = total / total.sum()
        mean = float((centres * share).sum())
        var = float((share * (centres - mean) ** 2).sum())
        cum = np.cumsum(share)
        pct = lambda q: float(centres[np.searchsorted(cum, q)])  # noqa: E731
        print("  %s  pixels %s" % (tag, f"{int(total.sum()):,}"))
        print("     mean %.3f dB   sd %.3f dB" % (mean, var ** 0.5))
        print("     p1 %.2f   p50 %.2f   p99 %.2f dB" % (pct(0.01), pct(0.50), pct(0.99)))
        occupied = int((total > 0).sum())
        print("     occupied bins %d / %d" % (occupied, N_BINS))
    print()
    print("  nodata pixels  VV %s   VH %s"
          % (f"{int(frame.n_nodata_vv.sum()):,}", f"{int(frame.n_nodata_vh.sum()):,}"))
    print("  above +25 dB   VV %s   VH %s"
          % (f"{int(frame.n_above_vv.sum()):,}", f"{int(frame.n_above_vh.sum()):,}"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    parser.add_argument("--limit", type=int, default=None, help="first N pairs only")
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()

    manifest = pd.read_parquet(
        paths.require(paths.MANIFESTS / "dataset_v3_manifest.parquet", "v3 manifest"),
        columns=["pair_name", "chip_id", "bbox_w", "bbox_s", "bbox_e", "bbox_n",
                 "s1_status"])
    clean = manifest[manifest["s1_status"] == "clean"]
    if not args.limit and len(clean) != EXPECTED_CLEAN:
        raise AssertionError("expected %d clean chips, found %d"
                             % (EXPECTED_CLEAN, len(clean)))

    groups = list(clean.groupby("pair_name", sort=True))
    if args.limit:
        groups = groups[:args.limit]

    rows, vv_blocks, vh_blocks, errors = [], [], [], []
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        futures = [
            pool.submit(histogram_pair, pair, group["chip_id"].to_numpy(),
                        group[["bbox_w", "bbox_s", "bbox_e", "bbox_n"]].to_numpy())
            for pair, group in groups
        ]
        for done, future in enumerate(as_completed(futures), 1):
            result = future.result()
            if result["error"]:
                errors.append(result)
                continue
            rows.append(pd.DataFrame({
                "pair_name": result["pair"], "chip_id": result["chip_id"],
                "n_nodata_vv": result["n_nodata_vv"], "n_nodata_vh": result["n_nodata_vh"],
                "n_above_vv": result["n_above_vv"], "n_above_vh": result["n_above_vh"],
            }))
            vv_blocks.append(result["vv"])
            vh_blocks.append(result["vh"])
            if done % 250 == 0:
                print("  %d/%d pairs" % (done, len(groups)), flush=True)

    if errors:
        print("\nERRORS on %d pairs, e.g. %s" % (len(errors), errors[0]["error"]))
    frame = pd.concat(rows, ignore_index=True)
    vv = np.vstack(vv_blocks)
    vh = np.vstack(vh_blocks)

    # Clean chips have no nodata by construction -- that is what clean means.
    if int(frame.n_nodata_vv.sum()) or int(frame.n_nodata_vh.sum()):
        raise AssertionError("a clean chip contains nodata; s1_status is inconsistent")

    summarise(frame, vv, vh)

    if not args.write:
        print("\n(dry run; pass --write to save)")
        return 0

    edges = FLOOR_DB + BIN_WIDTH_DB * np.arange(N_BINS + 1)
    table = pa.Table.from_pydict({
        "pair_name": pa.array(frame["pair_name"]),
        "chip_id": pa.array(frame["chip_id"]),
        "vv_counts": pa.array(list(vv), type=pa.list_(pa.uint32(), N_BINS)),
        "vh_counts": pa.array(list(vh), type=pa.list_(pa.uint32(), N_BINS)),
        "n_nodata_vv": pa.array(frame["n_nodata_vv"]),
        "n_nodata_vh": pa.array(frame["n_nodata_vh"]),
        "n_above_vv": pa.array(frame["n_above_vv"]),
        "n_above_vh": pa.array(frame["n_above_vh"]),
    })
    # Edges once, in metadata -- never a copy per row.
    table = table.replace_schema_metadata({
        "segwater.histogram": json.dumps({
            "space": "dB",
            "encoding": "dB = code * %g + %g; code 0 = nodata" % (STEP_DB, FLOOR_DB),
            "edges_db": [float(e) for e in edges],
            "n_bins": N_BINS,
            "bin_width_db": BIN_WIDTH_DB,
            "codes_per_bin": CODES_PER_BIN,
            "nodata": "code 0 excluded, counted in n_nodata_*",
            "label_mask": "none -- label-255 pixels ARE binned; they are model input",
            "chips": "s1_status == 'clean' only",
            "built": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        }),
    })
    paths.ensure_out()
    target = paths.MANIFESTS / "chip_s1_histograms.parquet"
    pq.write_table(table, target, compression="zstd")
    print("\nwrote %s  (%d rows, %.1f MiB)"
          % (target, len(frame), target.stat().st_size / 2**20))
    return 0


if __name__ == "__main__":
    sys.exit(main())
