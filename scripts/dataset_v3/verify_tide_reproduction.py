#!/usr/bin/env python3
"""Reproduce stored tide values for chips that already have them.

Before computing tide for the 93,121 recovered chips, confirm the pipeline
reproduces what the original produced. Sampling chips that already carry stored
values and regenerating them from FES2022b is the only way to know the chain --
centroid, nearest ocean-edge point, FES prediction, window snapping, phase
labelling -- is faithful rather than merely plausible.

The chain being reproduced, per chip:

1. chip bbox centroid -> BallTree(haversine) over the 489,673 ocean-edge points
2. FES2022b at that point, 15-min grid snapped to the 2014-01-01 epoch
3. find_peaks over a +/-25 h window -> phase label
4. nearest 15-min sample to the satellite time -> level

Phase is labelled two ways and both are reported. The original used
``.loc[pos_i:pos_j]`` with *positional* indices from ``find_peaks`` against a
*label*-indexed frame, which mislabels rows near turning points. Stored phases
match that replica, not the corrected positional logic, so the replica is what
must agree -- reproducing the bug is the requirement, not a concession.

Reports level agreement in cm and phase agreement as a match rate, for S1 and
S2 separately. Agreement is judged against the July 2026 reproduction, which reached max
|delta| 0.089 cm over 399 chip x sensor comparisons. Sub-millimetre level
agreement is therefore the bar; a systematically larger error means the chain
has drifted. Phase is the stricter test, since it depends on the window
snapping and the extremum search as well as the prediction.
"""

from __future__ import annotations

import argparse
import os
import pathlib
import sqlite3
import struct
import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.signal import find_peaks
from sklearn.neighbors import BallTree

import paths

FES_ROOT = pathlib.Path(os.environ.get("SEGWATER_FES_ROOT", "/mnt/local_ssd/fes/fes2022b"))
CENTROIDS = pathlib.Path(os.environ.get(
    "SEGWATER_EDGE_POINTS", "/mnt/local_ssd/fes/ancillary/Global_Ocean_Edge_centroids.gpkg"))

SOURCE = ("Global_Sen12_Coast_2017_2024_CHIPS_DATABASE_with_splits_w_metadata"
          "_w_tides_w_path_passedQC_memmap_selected-V2-w_histograms-CONFIRMED"
          "-RESPLIT_PATHS_FIXED.parquet")

HOURS = 25
STEP_MIN = 15
GRID_EPOCH = pd.Timestamp("2014-01-01T00:00:00", tz="UTC")
EXPECTED_EDGE_POINTS = 489_673


def edge_points() -> np.ndarray:
    """(N,2) lat/lon of the ocean-edge centroids, from the GeoPackage header."""
    with sqlite3.connect("file:%s?mode=ro" % CENTROIDS, uri=True) as connection:
        layer = connection.execute(
            "SELECT table_name FROM gpkg_contents").fetchone()[0]
        blobs = connection.execute('SELECT geom FROM "%s"' % layer).fetchall()
    points = np.empty((len(blobs), 2), dtype=np.float64)
    for i, (blob,) in enumerate(blobs):
        flags = blob[3]
        envelope = (flags >> 1) & 0x07
        # Point geometries carry no envelope (indicator 0) -- unlike the polygon
        # layers elsewhere in this project -- so the coordinates come from the
        # WKB body: header, then byte order, then a uint32 type, then x and y.
        offset = 8 + {0: 0, 1: 32, 2: 48, 3: 48, 4: 64}[envelope]
        wkb_order = "<" if blob[offset] == 1 else ">"
        x, y = struct.unpack_from("%s2d" % wkb_order, blob, offset + 5)
        points[i] = (y, x)
    if len(points) != EXPECTED_EDGE_POINTS:
        raise AssertionError("expected %d edge points, read %d"
                             % (EXPECTED_EDGE_POINTS, len(points)))
    return points


def snap_window(sat: pd.Timestamp) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Window bounds on the canonical 2014-epoch grid.

    The original per-point CSVs were one continuous 15-min series anchored at
    2014-01-01, and the window was a slice of it. Regenerating from an arbitrary
    floor of the satellite time would offset the grid by (sat mod 15 min),
    shifting find_peaks' extrema and flipping labels near turning points.
    """
    step = pd.Timedelta(minutes=STEP_MIN)
    lo = np.floor((sat - pd.Timedelta(hours=HOURS) - GRID_EPOCH) / step)
    hi = np.ceil((sat + pd.Timedelta(hours=HOURS) - GRID_EPOCH) / step)
    return GRID_EPOCH + int(lo) * step, GRID_EPOCH + int(hi + 1) * step


def label_phases(pure: np.ndarray, index: pd.Index):
    """(positional, .loc-replica) phase labels. The replica is what must match."""
    peaks, _ = find_peaks(pure)
    bottoms, _ = find_peaks(-pure)
    extrema = sorted(peaks.tolist() + bottoms.tolist())
    if not extrema:
        blank = pd.Series([None] * len(pure), index=index, dtype=object)
        return blank, blank.copy()

    seq = (["high", "ebb", "low", "flood"] if extrema[0] in peaks
           else ["low", "flood", "high", "ebb"])

    positional = pd.Series([None] * len(pure), index=index, dtype=object)
    for i in range(len(extrema) - 1):
        positional.iloc[extrema[i]:extrema[i + 1]] = seq[i % 4]
    positional.iloc[extrema[-1]:] = seq[(len(extrema) - 1) % 4]

    replica = pd.Series([None] * len(pure), index=index, dtype=object)
    for i in range(len(extrema) - 1):
        try:
            replica.loc[extrema[i]:extrema[i + 1]] = seq[i % 4]
        except Exception:  # noqa: BLE001 - the original swallowed these too
            pass
    try:
        replica.loc[extrema[-1]:] = seq[(len(extrema) - 1) % 4]
    except Exception:  # noqa: BLE001
        pass
    return positional, replica


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chips", type=int, default=200, help="how many to sample")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    import pyfes

    manifest = pd.read_parquet(
        paths.require(paths.MANIFESTS / "dataset_v3_manifest.parquet", "v3 manifest"),
        columns=["pair_name", "chip_id", "chip_origin", "bbox_w", "bbox_s",
                 "bbox_e", "bbox_n", "system:time_start_s1",
                 "s1_tide_level_sat", "s1_tide_level_sat_phase",
                 "s2_tide_level_sat", "s2_tide_level_sat_phase"])
    stored = manifest[(manifest["chip_origin"] == "existing")
                      & manifest["s1_tide_level_sat"].notna()]
    sample = stored.sample(n=min(args.chips, len(stored)), random_state=args.seed)

    # S2 acquisition time is per pair and not derivable from the scene id, so it
    # is joined rather than parsed. The id's leading datetime is the datastrip
    # sensing start, up to 49 minutes before the granule time.
    source = pq.read_table(paths.DB_SLIM / SOURCE,
                           columns=["pair_name", "system:time_start_s2"]).to_pandas()
    s2_time = (source.dropna(subset=["system:time_start_s2"])
               .drop_duplicates("pair_name").set_index("pair_name")["system:time_start_s2"])
    sample = sample.assign(s2_time=sample["pair_name"].map(s2_time))

    print("sampled %d chips with stored tide" % len(sample))
    points = edge_points()
    tree = BallTree(np.deg2rad(points), metric="haversine")

    # The YAML uses paths relative to the atlas root, so it must be loaded with
    # that root as CWD. pyfes >= 2025 exposes this as pyfes.config.load.
    cwd = os.getcwd()
    os.chdir(FES_ROOT)
    try:
        # Configuration is a NamedTuple(models, settings); models holds the
        # {"tide": ocean, "radial": load} handlers.
        handlers = pyfes.config.load(FES_ROOT / "fes2022.yaml").models
    finally:
        os.chdir(cwd)

    rows = []
    for n, (_, chip) in enumerate(sample.iterrows(), 1):
        lat = (chip["bbox_s"] + chip["bbox_n"]) / 2
        lon = (chip["bbox_w"] + chip["bbox_e"]) / 2
        _, idx = tree.query(np.deg2rad([[lat, lon]]), k=1)
        plat, plon = points[idx[0][0]]

        for tag, sat_raw in (("s1", chip["system:time_start_s1"]), ("s2", chip["s2_time"])):
            sat = pd.to_datetime(sat_raw, utc=True, errors="coerce")
            if pd.isna(sat):
                continue
            # Generate on the snapped grid, then filter to the true +/-25 h as
            # the original did. The snapped bounds are deliberately wider, so
            # taking them as the window would hand find_peaks extra extrema and
            # shift every phase label by a segment.
            lo, hi = snap_window(sat)
            grid = pd.date_range(lo, hi, freq="%dmin" % STEP_MIN, inclusive="left", tz="UTC")
            keep = ((grid >= sat - pd.Timedelta(hours=HOURS))
                    & (grid <= sat + pd.Timedelta(hours=HOURS)))
            times = grid[keep]
            dates = times.tz_localize(None).to_numpy(dtype="datetime64[us]")
            lons = np.full(len(dates), plon)
            lats = np.full(len(dates), plat)
            tide, lp, _ = pyfes.evaluate_tide(handlers["tide"], dates, lons, lats)
            pure = np.asarray(tide) + np.asarray(lp)

            # Row labels as they would have been inside the full 2014-2025 file,
            # which is what makes the .loc replica faithful.
            labels = pd.Index(((times - GRID_EPOCH) // pd.Timedelta(minutes=STEP_MIN)).astype(int))
            _, replica = label_phases(pure, labels)

            k = int(np.argmin(np.abs(times - sat)))
            rows.append({
                "pair_name": chip["pair_name"], "chip_id": chip["chip_id"],
                "sensor": tag,
                "stored_level": chip["%s_tide_level_sat" % tag],
                "regen_level": float(pure[k]),
                "stored_phase": chip["%s_tide_level_sat_phase" % tag],
                "regen_phase": replica.iloc[k],
            })
        if n % 25 == 0:
            print("  %d/%d chips" % (n, len(sample)), flush=True)

    out = pd.DataFrame(rows)
    out["level_diff"] = (out["stored_level"] - out["regen_level"]).abs()

    print()
    print("LEVEL agreement (cm)")
    for sensor, group in out.groupby("sensor"):
        d = group["level_diff"].dropna()
        print("  %s  n=%-5d  max %.6f  mean %.6f  p99 %.6f"
              % (sensor.upper(), len(d), d.max(), d.mean(), d.quantile(0.99)))
    print()
    print("PHASE agreement, against the .loc replica")
    for sensor, group in out.groupby("sensor"):
        ok = group.dropna(subset=["stored_phase", "regen_phase"])
        match = (ok["stored_phase"] == ok["regen_phase"]).sum()
        print("  %s  %d/%d  (%.2f%%)"
              % (sensor.upper(), match, len(ok), 100 * match / max(len(ok), 1)))
    LEVEL_TOLERANCE_CM = 0.1  # the July reproduction reached 0.089
    mism = out[out["stored_phase"].notna() & out["regen_phase"].notna()
               & (out["stored_phase"] != out["regen_phase"])]
    if len(mism):
        print()
        print("  PHASE mismatches (%d) -- level at each, to see if it sits near a turn:" % len(mism))
        print(mism[["pair_name", "chip_id", "sensor", "stored_level",
                    "stored_phase", "regen_phase"]].head(12).to_string(index=False))

    bad = out[out["level_diff"] > LEVEL_TOLERANCE_CM]
    if len(bad):
        print()
        print("  %d rows exceed %.2f cm:" % (len(bad), LEVEL_TOLERANCE_CM))
        print(bad.head(10).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
