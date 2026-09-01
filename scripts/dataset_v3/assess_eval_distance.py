#!/usr/bin/env python3
"""Distance from every passing chip to the nearest evaluation-site AOI.

One number per chip -- the minimum distance from its bbox to any of the seven
evaluation AOIs -- plus which site is nearest. No decisions are made here; the
30 km exclusion itself is applied by ``apply_splits.py``, so the threshold can
change without re-measuring anything.

**Method** (the one recorded as authoritative in
``docs/dataset_v3/EXCLUSION_BUFFER.md``): equirectangular rect-to-rect distance
between the chip bbox and the AOI bbox. The per-axis separation is zero when
the intervals overlap, so an overlapping chip scores exactly 0.0 and a chip
diagonal to the AOI gets the corner-to-corner distance, not an axis one.
Longitude degrees are scaled by cos(lat) at the AOI's mean latitude. The error
of this approximation is far below 1 km at 30 km separations for every site
(the highest-latitude AOI is Truc Vert at 44.7 deg N), which is why it is the
accepted method.

**AOI provenance.** The bboxes below are copied, not typed: they come from
``docs/roadmap_for_publication/figures/FIG01_dataset_map/sites.py``, whose own
rule is that every coordinate must be file-sourced (the four SDS sites resolve
to ``SDS_Benchmark_slim/scripts/sds/sds_core.py`` ``SITE_AOI_POLYGONS``). They
are embedded here rather than imported because ``sites.py`` lives inside the
nested ``docs`` git repository, which moves independently of this code root.
Values verified against sites.py on 2026-09-01. Spelling note: the Louisiana
site is ``ROCKERFELLER`` in code and artifact filenames elsewhere in the
project; this module uses the manuscript spelling for the site key.

The check that matters most here is the **regression against the independent
prior implementation** (the cost-table script that produced the numbers in
EXCLUSION_BUFFER.md): the excluded-chip counts at five radii must reproduce
exactly. Matching five thresholds from an independently written distance
computation makes a coincidental agreement implausible.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

import paths

EXPECTED_CHIPS = 1_357_354
EXPECTED_PASSING = 1_328_811

# Recovered chips occupy a disjoint id space; re-asserted at every join site.
RECOVERED_ID_OFFSET = 100_000
MAX_SOURCE_CHIP_ID = 4_634

# (minlon, minlat, maxlon, maxlat), EPSG:4326 -- FIG01 sites.py, see docstring.
AOIS = {
    "narrabeen":   (151.2986, -33.7347, 151.3107, -33.7050),
    "duck":        (-75.7571, 36.1742, -75.7401, 36.1919),
    "rockefeller": (-93.0237, 29.5283, -92.1878, 29.7666),
    "hampyeong":   (126.1222, 35.0029, 126.5308, 35.2942),
    "demak":       (110.4217, -7.0040, 110.6171, -6.8099),
    "torreypines": (-117.2769, 32.8697, -117.2431, 32.9436),
    "trucvert":    (-1.2580, 44.7237, -1.2338, 44.7622),
}

KM_PER_DEG = 111.32

# Regression anchors: the independent implementation behind EXCLUSION_BUFFER.md
# (exclusion_cost_chiplevel.py, 2026-09-01). Chip counts at dist <= R.
EXPECTED_AT_RADIUS = {0: 1_235, 20: 4_360, 25: 5_239, 30: 6_023, 50: 8_839}
EXPECTED_PER_SITE_30KM = {
    "rockefeller": 1_357, "hampyeong": 1_019, "demak": 911, "narrabeen": 909,
    "trucvert": 734, "torreypines": 724, "duck": 369,
}
EXPECTED_PAIRS_TOUCHED_30KM = 25
EXPECTED_EMPTIED_30KM = {"PAIR_2561", "PAIR_2562", "PAIR_2563"}


def rect_to_rect_km(boxes: np.ndarray, aoi: tuple[float, float, float, float]) -> np.ndarray:
    """Distance in km from each (w, s, e, n) box to one AOI box.

    Zero when the rectangles overlap on both axes. Longitude is scaled by
    cos(lat) at the AOI's mean latitude -- the AOIs are small, so one latitude
    serves the whole computation.
    """
    aw, as_, ae, an = aoi
    lat0 = np.deg2rad((as_ + an) / 2.0)
    dx_deg = np.maximum(0.0, np.maximum(aw - boxes[:, 2], boxes[:, 0] - ae))
    dy_deg = np.maximum(0.0, np.maximum(as_ - boxes[:, 3], boxes[:, 1] - an))
    return np.hypot(dx_deg * KM_PER_DEG * np.cos(lat0), dy_deg * KM_PER_DEG)


def validate_aois() -> None:
    for site, (w, s, e, n) in AOIS.items():
        if not (-180 <= w < e <= 180 and -90 <= s < n <= 90):
            raise AssertionError("AOI %s is not a valid bbox: %r" % (site, (w, s, e, n)))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()

    validate_aois()

    manifest = pd.read_parquet(
        paths.require(paths.MANIFESTS / "dataset_v3_manifest.parquet", "v3 manifest"),
        columns=["pair_name", "chip_id", "passes_all_gates",
                 "bbox_w", "bbox_s", "bbox_e", "bbox_n", "chip_origin"])
    if len(manifest) != EXPECTED_CHIPS:
        raise AssertionError("manifest is %d chips, expected %d"
                             % (len(manifest), EXPECTED_CHIPS))

    passing = manifest[manifest["passes_all_gates"]].reset_index(drop=True)
    if len(passing) != EXPECTED_PASSING:
        raise AssertionError("expected %d passing chips, found %d"
                             % (EXPECTED_PASSING, len(passing)))
    if passing.duplicated(["pair_name", "chip_id"]).any():
        raise AssertionError("(pair_name, chip_id) is not unique on the passing set")

    existing = passing.loc[passing["chip_origin"] == "existing", "chip_id"]
    recovered = passing.loc[passing["chip_origin"] == "recovered", "chip_id"]
    if int(existing.max()) > MAX_SOURCE_CHIP_ID or int(recovered.min()) <= RECOVERED_ID_OFFSET:
        raise AssertionError("chip id spaces are not disjoint at this join site")

    boxes = passing[["bbox_w", "bbox_s", "bbox_e", "bbox_n"]].to_numpy(dtype=np.float64)
    dist = np.column_stack([rect_to_rect_km(boxes, AOIS[site]) for site in sorted(AOIS)])
    sites = np.array(sorted(AOIS))
    nearest = np.argmin(dist, axis=1)
    dist_min = dist[np.arange(len(dist)), nearest]

    if not np.isfinite(dist_min).all() or (dist_min < 0).any():
        raise AssertionError("distances contain NaN, inf, or negatives")

    table = pd.DataFrame({
        "pair_name": passing["pair_name"],
        "chip_id": passing["chip_id"],
        "eval_dist_km": dist_min.astype(np.float32),
        "nearest_eval_site": sites[nearest],
    })

    print("EVAL-SITE DISTANCE over %d passing chips, %d AOIs" % (len(table), len(AOIS)))
    print()
    print("  excluded-chip counts by radius (regression vs the independent")
    print("  implementation behind EXCLUSION_BUFFER.md):")
    for radius, expected in sorted(EXPECTED_AT_RADIUS.items()):
        got = int((dist_min <= radius).sum())
        marker = "ok" if got == expected else "MISMATCH"
        print("    R = %3d km   %7d chips   expected %7d   %s"
              % (radius, got, expected, marker))
        if got != expected:
            raise AssertionError("R=%d km: %d excluded chips, expected %d -- the "
                                 "distance computation disagrees with the prior "
                                 "implementation" % (radius, got, expected))

    at30 = table[dist_min <= 30.0]
    per_site = at30["nearest_eval_site"].value_counts().to_dict()
    print()
    print("  at 30 km, per nearest site:")
    for site in sorted(AOIS):
        got = per_site.get(site, 0)
        expected = EXPECTED_PER_SITE_30KM[site]
        print("    %-12s %6d   expected %6d   %s"
              % (site, got, expected, "ok" if got == expected else "MISMATCH"))
        if got != expected:
            raise AssertionError("site %s: %d chips at 30 km, expected %d"
                                 % (site, got, expected))

    touched = at30["pair_name"].unique()
    if len(touched) != EXPECTED_PAIRS_TOUCHED_30KM:
        raise AssertionError("%d pairs touched at 30 km, expected %d"
                             % (len(touched), EXPECTED_PAIRS_TOUCHED_30KM))
    pair_sizes = passing.groupby("pair_name").size()
    emptied = {p for p in touched
               if int((at30["pair_name"] == p).sum()) == int(pair_sizes[p])}
    if emptied != EXPECTED_EMPTIED_30KM:
        raise AssertionError("pairs fully emptied at 30 km %r, expected %r"
                             % (sorted(emptied), sorted(EXPECTED_EMPTIED_30KM)))
    print()
    print("  pairs touched at 30 km: %d;  fully emptied: %s"
          % (len(touched), ", ".join(sorted(emptied))))
    print("  farthest chip from any AOI: %.0f km" % float(dist_min.max()))

    if not args.write:
        print("\n(dry run; pass --write to save)")
        return 0

    paths.ensure_out()
    target = paths.MANIFESTS / "chip_eval_distance.parquet"
    table.to_parquet(target, index=False, compression="zstd")
    print("\nwrote %s  (%d rows, %.1f MiB)"
          % (target, len(table), target.stat().st_size / 2**20))
    return 0


if __name__ == "__main__":
    sys.exit(main())
