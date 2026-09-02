#!/usr/bin/env python3
"""Compute FES2022b tide for every chip that has no stored value.

Two populations, same treatment:

* the **93,166 recovered windows**, which were never chipped, so nothing ever
  computed tide for them;
* **2,995 existing chips** whose tide is null in *every* parquet of the lineage
  -- all ten that carry the column. 2,108 of them are missing both sensors;
  a further 887 across 59 pairs have a stored S1 value but no S2 one, so the two
  sensors are independently absent and selecting on S1 alone misses them. It was never computed for
  them rather than lost in a join. Distance does not explain it: they sit a
  median 2.91 km from the nearest ocean-edge point against 3.97 km for chips
  that do have tide. None intersects the GCL coastline, so the tide gate never
  needed them, which is presumably why the gap went unnoticed.

Everything else carries tide from the original build; ``apply_tide.py`` joins
those rather than recomputing them.

The chain is the one verified in ``verify_tide_reproduction.py``, which
reproduced stored values on 600 chips x 2 sensors at max 0.097 cm level error
and **600/600 exact phase**. Nothing here deviates from it -- the shared logic
lives in that module and is imported rather than reimplemented, so the two
cannot drift apart.

Per chip:

1. bbox centroid -> BallTree(haversine) over the 489,673 ocean-edge points
2. FES2022b on the 15-min grid snapped to the 2014-01-01 epoch, then filtered to
   the true +/-25 h window
3. ``find_peaks`` -> phase, via the ``.loc`` replica of the original's indexing
4. nearest sample to the satellite time -> level

Two details that the verification proved load-bearing:

* The window must be **generated wide and then filtered**, not generated from
  the snapped bounds. Taking the snapped bounds directly hands ``find_peaks``
  extra extrema and shifts every phase label by one segment -- it scored 91.5%
  against stored values instead of 100%.
* Phase uses the ``.loc`` replica, reproducing the original's use of positional
  indices as labels. The corrected positional logic disagrees with the stored
  corpus on ~13.5% of rows, so reproducing the bug is what keeps the recovered
  chips consistent with the chips beside them.

Satellite times: S1 from ``system:time_start_s1`` on the manifest (100%
populated). S2 from the source parquet, joined **per pair** -- it is constant
within a pair and is *not* derivable from the scene id, whose leading datetime
is the datastrip sensing start, up to 49 minutes before the granule time.

Work is grouped by pair, and each pair loads FES over a **bbox around its own
chips** rather than the global atlas. That is not an optimisation but a
requirement: the global atlas needs 32.1 GB resident, so two workers would
exhaust the machine. A bbox load is seconds and a fraction of the memory.

⚠️ The FES2022b longitude axis runs 0..360, so a bbox spanning the 0/360 seam
yields a non-monotonic axis and pyfes raises "the axis values must be evenly
spaced from each other". Handled with the ladder the July reproduction
established: raw bbox, then the same bbox expressed in 0..360, then the global
atlas as a correct-but-slow last resort.
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.neighbors import BallTree

import paths
from verify_tide_reproduction import (FES_ROOT, GRID_EPOCH, HOURS, SOURCE,
                                      STEP_MIN, edge_points, label_phases,
                                      snap_window)

# Chips that have never had tide computed, by origin, on a manifest where none
# has been filled in yet. The existing ones are a genuine gap in the source
# lineage, not a join failure -- see the module docstring.
#
# This is a CEILING, not an equality: the script is resumable, so a rerun after
# a partial pass legitimately sees fewer. Only an origin appearing that is not
# in this dict, or a count above it, means something changed.
MAX_MISSING = {"recovered": 93_166, "existing": 2_995}
RECOVERED_ID_OFFSET = 100_000

# Padding around a pair's chips for the FES bbox, in degrees. Generous: the
# nearest ocean-edge point can sit some way from a chip, and an under-sized box
# would silently fall back to the 32 GB global atlas.
BBOX_PAD_DEG = 2.0

_STATE: dict = {}


def _init() -> None:
    """Per-worker state that does not depend on the pair: the edge-point tree."""
    import pyfes

    points = edge_points()
    _STATE["points"] = points
    _STATE["tree"] = BallTree(np.deg2rad(points), metric="haversine")
    _STATE["pyfes"] = pyfes


def load_handlers(bbox: tuple[float, float, float, float]):
    """FES handlers over ``bbox``, with the 0/360-seam ladder.

    The global atlas is correct everywhere but costs 32.1 GB and ~375 s, so it
    is the last resort rather than the default.
    """
    pyfes = _STATE["pyfes"]
    west, south, east, north = bbox
    attempts = [(west, south, east, north)]

    # The atlas longitude axis is 0..360, monotonic, step 1/30 deg. A box wholly
    # in negative longitude re-expresses cleanly by adding 360. A box STRADDLING
    # lon 0 cannot: it is not a contiguous slice of that axis in any shift, so
    # no bbox exists and the global atlas is the only correct answer. That is a
    # property of the grid, not a limitation worth coding around -- 63 of the
    # 1,915 recovered pairs straddle the seam and are handled by scheduling them
    # with reduced concurrency rather than by contorting the box.
    if west < 0 and east < 0:
        attempts.append((west + 360, south, east + 360, north))
    attempts.append(None)
    cwd = os.getcwd()
    os.chdir(FES_ROOT)
    try:
        for attempt in attempts:
            try:
                config = pyfes.config.load(FES_ROOT / "fes2022.yaml", attempt)
                return config.models, attempt is None
            except Exception:  # noqa: BLE001 - fall through to the next rung
                continue
        raise RuntimeError("could not load FES handlers for bbox %r" % (bbox,))
    finally:
        os.chdir(cwd)


def tide_at(plat: float, plon: float, sat: pd.Timestamp) -> tuple[float, object]:
    """(level_cm, phase) at one point and time, by the verified chain."""
    pyfes = _STATE["pyfes"]
    lo, hi = snap_window(sat)
    grid = pd.date_range(lo, hi, freq="%dmin" % STEP_MIN, inclusive="left", tz="UTC")
    keep = ((grid >= sat - pd.Timedelta(hours=HOURS))
            & (grid <= sat + pd.Timedelta(hours=HOURS)))
    times = grid[keep]
    if not len(times):
        return float("nan"), None

    dates = times.tz_localize(None).to_numpy(dtype="datetime64[us]")
    tide, lp, _ = pyfes.evaluate_tide(
        _STATE["handlers"]["tide"], dates,
        np.full(len(dates), plon), np.full(len(dates), plat))
    pure = np.asarray(tide) + np.asarray(lp)

    # Row labels as they would have been in the full 2014-2025 series, which is
    # what makes the .loc replica faithful.
    labels = pd.Index(((times - GRID_EPOCH) // pd.Timedelta(minutes=STEP_MIN)).astype(int))
    _, replica = label_phases(pure, labels)

    k = int(np.argmin(np.abs(times - sat)))
    return float(pure[k]), replica.iloc[k]


def compute_pair(pair: str, chips: pd.DataFrame) -> dict:
    """Tide for every recovered chip of one pair. Errors are returned, not raised."""
    try:
        if not _STATE:
            _init()
        tree, points = _STATE["tree"], _STATE["points"]

        lat = ((chips["bbox_s"] + chips["bbox_n"]) / 2).to_numpy()
        lon = ((chips["bbox_w"] + chips["bbox_e"]) / 2).to_numpy()
        _, idx = tree.query(np.deg2rad(np.column_stack([lat, lon])), k=1)
        nearest = points[idx[:, 0]]

        # Box the matched tide points, not just the chips: the nearest edge
        # point can lie outside the chips' own extent.
        box = (min(lon.min(), nearest[:, 1].min()) - BBOX_PAD_DEG,
               min(lat.min(), nearest[:, 0].min()) - BBOX_PAD_DEG,
               max(lon.max(), nearest[:, 1].max()) + BBOX_PAD_DEG,
               max(lat.max(), nearest[:, 0].max()) + BBOX_PAD_DEG)

        # A seam-straddling pair needs the global atlas, which costs ~375 s to
        # load. Reuse it across every such pair in this worker rather than
        # paying that per pair: 62 pairs x 375 s would be 6.5 hours serially,
        # against one load for all of them.
        if box[0] < 0 <= box[2] and _STATE.get("global_handlers") is not None:
            _STATE["handlers"], used_global = _STATE["global_handlers"], True
        else:
            _STATE["handlers"], used_global = load_handlers(box)
            if used_global:
                _STATE["global_handlers"] = _STATE["handlers"]

        out = {k: [] for k in ("s1_level", "s1_phase", "s2_level", "s2_phase")}
        for i in range(len(chips)):
            plat, plon = nearest[i]
            for tag, raw in (("s1", chips["sat_s1"].iat[i]), ("s2", chips["sat_s2"].iat[i])):
                sat = pd.to_datetime(raw, utc=True, errors="coerce")
                if pd.isna(sat):
                    out["%s_level" % tag].append(float("nan"))
                    out["%s_phase" % tag].append(None)
                    continue
                level, phase = tide_at(plat, plon, sat)
                out["%s_level" % tag].append(level)
                out["%s_phase" % tag].append(phase)

        return {"pair": pair, "chip_id": chips["chip_id"].to_numpy(),
                "nearest_lat": nearest[:, 0], "nearest_lon": nearest[:, 1],
                "used_global": used_global, "error": "", **out}
    except Exception as error:  # noqa: BLE001 - one bad pair must not kill the run
        return {"pair": pair, "error": "%s: %s" % (type(error).__name__, str(error)[:200])}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    parser.add_argument("--global-jobs", type=int, default=1,
                        help="workers for seam-straddling pairs. Each loads the "
                             "global atlas, measured at 32 GB at load and up to "
                             "45 GB RSS in use, so 1 is what fits in 86 GB -- 2 "
                             "was OOM-killed")
    parser.add_argument("--limit", type=int, default=None, help="first N pairs only")
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()

    manifest = pd.read_parquet(
        paths.require(paths.MANIFESTS / "dataset_v3_manifest.parquet", "v3 manifest"),
        columns=["pair_name", "chip_id", "chip_origin", "bbox_w", "bbox_s",
                 "bbox_e", "bbox_n", "system:time_start_s1", "s1_tide_level_sat",
                 "s2_tide_level_sat"])
    # Everything missing a tide value for EITHER sensor, whatever its origin.
    # Keying on s1 alone missed 887 chips whose s1 is stored but whose s2 is
    # null in the source -- the two sensors' values are independently absent.
    recovered = manifest[manifest["s1_tide_level_sat"].isna()
                         | manifest["s2_tide_level_sat"].isna()].copy()
    by_origin = recovered["chip_origin"].value_counts().to_dict()
    unknown = set(by_origin) - set(MAX_MISSING)
    if unknown:
        raise AssertionError("chips missing tide from an unexpected origin: %s"
                             % sorted(unknown))
    for origin, n in by_origin.items():
        if n > MAX_MISSING[origin]:
            raise AssertionError("%s chips missing tide rose to %d, above the "
                                 "known %d" % (origin, n, MAX_MISSING[origin]))
    if recovered.empty:
        print("every chip already has tide; nothing to compute")
        return 0
    is_recovered = recovered["chip_origin"] == "recovered"
    if (recovered.loc[is_recovered, "chip_id"] < RECOVERED_ID_OFFSET).any():
        raise AssertionError("a recovered chip_id is below the offset")

    source = pq.read_table(paths.DB_SLIM / SOURCE,
                           columns=["pair_name", "system:time_start_s2"]).to_pandas()
    s2_time = (source.dropna(subset=["system:time_start_s2"])
               .drop_duplicates("pair_name").set_index("pair_name")["system:time_start_s2"])
    recovered["sat_s1"] = recovered["system:time_start_s1"]
    recovered["sat_s2"] = recovered["pair_name"].map(s2_time)
    missing = int(recovered["sat_s2"].isna().sum())
    if missing:
        print("NOTE %d recovered chips have no S2 acquisition time" % missing)

    groups = list(recovered.groupby("pair_name", sort=True))
    if args.limit:
        groups = groups[:args.limit]

    # Split by whether a bbox can represent the pair at all. Seam-straddling
    # pairs must load the 32.1 GB global atlas, so only a couple can run at once
    # on an 86 GB machine; everything else is cheap and runs wide.
    def straddles(group: pd.DataFrame) -> bool:
        west = group["bbox_w"].min() - BBOX_PAD_DEG
        east = group["bbox_e"].max() + BBOX_PAD_DEG
        return west < 0 <= east

    seam = [(p, g) for p, g in groups if straddles(g)]
    plain = [(p, g) for p, g in groups if not straddles(g)]
    print("computing tide for %d chips across %d pairs  %s"
          % (sum(len(g) for _, g in groups), len(groups), by_origin))
    print("  %d pairs by bbox (%d workers), %d straddling lon 0 by global atlas "
          "(%d workers)" % (len(plain), args.jobs, len(seam), args.global_jobs))

    rows, errors = [], []
    for label, batch, workers in (("bbox", plain, args.jobs),
                                  ("global-atlas", seam, args.global_jobs)):
        if not batch:
            continue
        print("\n-- %s pass: %d pairs, %d workers" % (label, len(batch), workers), flush=True)
        with ProcessPoolExecutor(max_workers=workers, initializer=_init) as pool:
            futures = [pool.submit(compute_pair, pair, group) for pair, group in batch]
            for done, future in enumerate(as_completed(futures), 1):
                result = future.result()
                if result["error"]:
                    errors.append(result)
                    continue
                rows.append(pd.DataFrame({
                    "pair_name": result["pair"], "chip_id": result["chip_id"],
                    "s1_tide_level_sat": result["s1_level"],
                    "s1_tide_level_sat_phase": result["s1_phase"],
                    "s2_tide_level_sat": result["s2_level"],
                    "s2_tide_level_sat_phase": result["s2_phase"],
                    "tide_point_lat": result["nearest_lat"],
                    "tide_point_lon": result["nearest_lon"],
                }))
                if done % 100 == 0:
                    print("   %d/%d pairs" % (done, len(batch)), flush=True)

    table = pd.concat(rows, ignore_index=True)
    if errors:
        print("\nERRORS on %d pairs, e.g. %s" % (len(errors), errors[0]["error"]))

    delta = (table["s1_tide_level_sat"] - table["s2_tide_level_sat"]).abs()
    print()
    print("COMPUTED for %d chips" % len(table))
    for tag in ("s1", "s2"):
        level = table["%s_tide_level_sat" % tag]
        print("  %s  level cm: min %.1f  median %.1f  max %.1f  | null %d"
              % (tag.upper(), level.min(), level.median(), level.max(),
                 int(level.isna().sum())))
        print("     phase: %s" % table["%s_tide_level_sat_phase" % tag]
              .value_counts(dropna=False).to_dict())
    print()
    print("  |s1 - s2| delta: median %.2f  p90 %.2f  max %.2f cm"
          % (delta.median(), delta.quantile(0.9), delta.max()))
    print("    <= 10 cm (the original retention bar): %d (%.1f%%)"
          % (int((delta <= 10).sum()), 100 * float((delta <= 10).mean())))

    if not args.write:
        print("\n(dry run; pass --write to save)")
        return 0

    paths.ensure_out()
    target = paths.MANIFESTS / "computed_chip_tide.parquet"

    # Accumulate rather than replace. The script only ever computes chips that
    # are still missing tide, so a second run covers a different population --
    # replacing would silently discard the first run's work.
    if target.exists():
        previous = pd.read_parquet(target)
        overlap = pd.MultiIndex.from_frame(previous[["pair_name", "chip_id"]]).isin(
            pd.MultiIndex.from_frame(table[["pair_name", "chip_id"]]))
        table = pd.concat([previous[~overlap], table], ignore_index=True)
        print("\nmerged with %d existing rows" % int((~overlap).sum()))
    if table.duplicated(["pair_name", "chip_id"]).any():
        raise AssertionError("(pair_name, chip_id) is not unique in the tide table")

    table.to_parquet(target, index=False, compression="zstd")
    print("wrote %s  (%d rows)" % (target, len(table)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
