#!/usr/bin/env python3
"""
Write per-site GeoPackages showing WHERE the training chips of each study site
actually sit, split by whether their scene was ever evaluated.

Companion to `eval_training_overlap_audit.py`, which answers "how many chips";
this answers "which ground". The audit's tiers become map layers you can open in
QGIS on top of the site AOI.

TWO CATEGORIES (the distinction the audit is built on)
------------------------------------------------------
  <site>__train_only.gpkg
      Training chips inside the site AOI whose Sentinel-1 scene was NEVER
      evaluated -- the scene falls outside the site's evaluated period, or on a
      date that was simply never scored. The model saw this coastline, but not
      the image it is graded on. Every site has such chips.

  <site>__train_and_eval.gpkg
      Training chips from the scene that IS also an evaluation scene: the
      pre-audit overlap. Only Hampyeong, Demak and Rockefeller have these
      (1 scene each). This is the layer that shows what "leakage" meant on the
      ground -- and, visibly, how small a part of each AOI it covers.

Geometry is the real chip footprint (`CHIP_BBOX`, ~0.02 deg = 224 px at 10 m),
NOT a centroid, so overlap with the AOI is visible rather than asserted. A
second file per site, `<site>__aoi.gpkg`, carries the AOI rectangle so the two
are interpretable together.

AOI RULE (stated because it changes the counts)
    A chip is "in the AOI" if its bbox INTERSECTS the site's bbox from FIG01's
    sites.py -- any overlap, not containment, and the site's *bounding box*, not
    its polygon. Intersection is the conservative choice for a leakage audit:
    it over-counts exposure rather than hiding it. Each feature carries
    `frac_in_aoi` and `center_in_aoi` so a stricter rule can be applied in QGIS
    without regenerating anything. For reference, tightening to "centre inside"
    gives 77/78/9 instead of 81/94/15 for the three overlap scenes.
    ⚠️ Rockefeller's bbox is the envelope of 947 transects along ~80 km of
    curved coast, so it encloses ground the transects never sample; its counts
    are the loosest of the seven.

Attributes per chip: pair_name, chip_id, scene_id, acq_date, split (shipped,
chip-level), split_pair_based, category, frac_in_aoi, center_in_aoi, plus the
distance to the AOI edge (0 for everything written here, kept for symmetry with
the audit CSVs).

Usage
-----
    conda run -n eda_coastsat python scripts/analysis/eval_training_overlap_gpkg.py \\
        --out-dir docs/roadmap_for_publication/supplementary_materials/\\
S12_split_leakage_disclosure/gpkg

Read-only w.r.t. the dataset; only writes into --out-dir.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from shapely.geometry import box

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "docs/roadmap_for_publication/figures/FIG01_dataset_map"))
from sites import SITES, validate as validate_sites  # noqa: E402

PARQUET = Path(
    "/Volumes/noel_wd_black_sn850x/segwater-database/DATABASE/SLIM_PARQUET/"
    "Global_Sen12_Coast_2017_2024_CHIPS_DATABASE_with_splits_w_metadata_w_tides"
    "_w_path_passedQC_memmap_selected-V2-w_histograms-CONFIRMED-RESPLIT_PATHS_FIXED.parquet"
)

# The one evaluated scene per site that also contributed training chips.
# Established by eval_training_overlap_audit.py (exact product name for
# Hampyeong/Rockefeller; +/-2 min acquisition-time match for Demak).
LEAK_SCENE = {
    "hampyeong": "S1B_IW_GRDH_1SDV_20210329T213224_20210329T213249_026235_03218E_DB19",
    "demak": "S1A_IW_GRDH_1SDV_20230812T221741_20230812T221806_049848_05FECB_D72F",
    "rockefeller": "S1A_IW_GRDH_1SDV_20210211T001021_20210211T001046_036535_044A5C_7A3B",
}

# Regression anchors: total AOI chips, and how they split. From the audit.
EXPECTED = {
    "rockefeller": (594, 15), "hampyeong": (193, 81), "demak": (112, 94),
    "torreypines": (30, 0), "narrabeen": (8, 0), "trucvert": (5, 0), "duck": (2, 0),
}

DATE_RE = re.compile(r"_(\d{8})T(\d{6})_")


def acq_date(scene: str) -> str:
    m = DATE_RE.search(scene)
    return m.group(1) if m else ""


def load_chips(parquet: Path) -> pd.DataFrame:
    if not parquet.exists():
        sys.exit(f"Build parquet not found:\n  {parquet}\nMount the black SSD.")
    cols = ["CHIP_BBOX", "system:index_s1", "pair_name", "chip_id",
            "split", "split_pair_based", "selected_for_memmap"]
    t = pq.read_table(parquet, columns=cols)
    d = pd.DataFrame({c: t.column(c).to_pylist() for c in cols})
    d = d[d["selected_for_memmap"] == True].reset_index(drop=True)  # noqa: E712
    bb = np.full((len(d), 4), np.nan)
    for i, s in enumerate(d["CHIP_BBOX"]):
        if s:
            bb[i] = json.loads(s)
    d[["lon0", "lat0", "lon1", "lat1"]] = bb
    return d.dropna(subset=["lon0"]).reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--parquet", type=Path, default=PARQUET)
    ap.add_argument("--out-dir", type=Path, required=True)
    a = ap.parse_args()

    validate_sites()
    boxes = {s["key"]: s["bbox"] for s in SITES}
    labels = {s["key"]: s["label"].replace("\n", " ") for s in SITES}

    print(f"Reading trained-on chips from\n  {a.parquet}")
    d = load_chips(a.parquet)
    print(f"  {len(d):,} trained-on chips with geometry")

    a.out_dir.mkdir(parents=True, exist_ok=True)
    summary = []

    for key, bbx in boxes.items():
        lo0, la0, lo1, la1 = bbx
        m = d[(d.lon0 < lo1) & (d.lon1 > lo0) & (d.lat0 < la1) & (d.lat1 > la0)].copy()
        if m.empty:
            print(f"  {key:<13} no chips in AOI — skipped")
            continue

        # fraction of the chip's own area that falls inside the AOI, so a
        # stricter membership rule can be applied downstream without re-running.
        ow = np.minimum(m.lon1, lo1) - np.maximum(m.lon0, lo0)
        oh = np.minimum(m.lat1, la1) - np.maximum(m.lat0, la0)
        area = (m.lon1 - m.lon0) * (m.lat1 - m.lat0)
        m["frac_in_aoi"] = ((ow * oh) / area).round(4)
        cx, cy = (m.lon0 + m.lon1) / 2, (m.lat0 + m.lat1) / 2
        m["center_in_aoi"] = ((cx >= lo0) & (cx <= lo1) & (cy >= la0) & (cy <= la1))

        leak = LEAK_SCENE.get(key)
        m["category"] = np.where(m["system:index_s1"] == leak,
                                 "train_and_eval", "train_only")
        m["scene_id"] = m["system:index_s1"]
        m["acq_date"] = m["scene_id"].map(acq_date)
        m["geometry"] = [box(r.lon0, r.lat0, r.lon1, r.lat1) for r in m.itertuples()]

        keep = ["pair_name", "chip_id", "scene_id", "acq_date", "split",
                "split_pair_based", "category", "frac_in_aoi", "center_in_aoi",
                "geometry"]
        g = gpd.GeoDataFrame(m[keep], geometry="geometry", crs="EPSG:4326")

        # AOI rectangle, so the chips are interpretable on their own.
        gpd.GeoDataFrame(
            {"site": [key], "label": [labels[key]],
             "source": [next(s["source"] for s in SITES if s["key"] == key)]},
            geometry=[box(lo0, la0, lo1, la1)], crs="EPSG:4326",
        ).to_file(a.out_dir / f"{key}__aoi.gpkg", driver="GPKG")

        counts = {}
        for cat in ("train_only", "train_and_eval"):
            sub = g[g.category == cat]
            counts[cat] = len(sub)
            if sub.empty:
                continue
            out = a.out_dir / f"{key}__{cat}.gpkg"
            sub.to_file(out, driver="GPKG")
            scenes = sub.scene_id.nunique()
            print(f"  {key:<13}{cat:<16}{len(sub):>5} chips  {scenes} scene(s) -> {out.name}")

        exp = EXPECTED.get(key)
        if exp and (counts["train_only"], counts["train_and_eval"]) != exp:
            raise SystemExit(
                f"{key}: got {(counts['train_only'], counts['train_and_eval'])}, "
                f"expected {exp} — dataset or AOI changed, investigate before using"
            )

        summary.append({
            "site": key,
            "chips_in_aoi": len(g),
            "train_only": counts["train_only"],
            "train_and_eval": counts["train_and_eval"],
            "train_only_scenes": g[g.category == "train_only"].scene_id.nunique(),
            "eval_overlap_scene": leak or "",
            "eval_overlap_date": acq_date(leak) if leak else "",
            "center_in_aoi_strict": int(g.center_in_aoi.sum()),
        })

    s = pd.DataFrame(summary).sort_values("chips_in_aoi", ascending=False)
    s.to_csv(a.out_dir / "gpkg_summary.csv", index=False)
    print("\n" + s.to_string(index=False))
    print(f"\nwrote {len(list(a.out_dir.glob('*.gpkg')))} GeoPackages -> {a.out_dir}")


if __name__ == "__main__":
    main()
