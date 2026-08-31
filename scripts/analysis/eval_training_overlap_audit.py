#!/usr/bin/env python3
"""
Audit: do any EVALUATION Sentinel-1 scenes also appear in the TRAINING chips?

Answers the reviewer question "was an image you evaluate on also trained on, and
if so how many chips?" for all seven study sites, and writes a per-scene CSV so
the numbers in the manuscript/supplement are reproducible rather than asserted.

WHY THIS EXISTS
---------------
`DATASET_BUILD_LEDGER.md` previously stated "Exact scene-ID overlap: 0". That
audit covered only Hampyeong and Demak, so it missed Duck and Rockefeller. The
true count is 5 shared products, of which 3 put training chips inside the area
actually evaluated. This script is the executable form of that correction.

THE CENTRAL DISTINCTION
-----------------------
A Sentinel-1 IW product is ~250 km long, so "the same scene ID" and "the same
ground" are different claims. We therefore report a nested funnel:

  Tier 1  LEAK      same S1 product AND >=1 training chip inside the site AOI
  Tier 2  SHARED    same S1 product, but every chip is far along the swath
  Tier 3  NEARBY    training chips inside the AOI from a DIFFERENT acquisition

Only Tier 1 is contamination of the evaluated area. Tier 2 is the same image of
a different coastline; Tier 3 is the same coastline at a different time. Quoting
Tier 2 as a leak overstates the problem; ignoring Tier 3 understates it.

MATCHING
--------
Sites differ in how eval scenes are named, so identity is established twice:

  * SAFE-named sites (Duck, Narrabeen, Torrey Pines, Truc Vert, Rockefeller,
    Hampyeong) -> exact string match on the full product name.
  * Demak -> its scene ids are compact (`S1_<YYYYMMDD>_<HHMMSS>_76_8_9`), so we
    match on ACQUISITION TIMESTAMP (<= --time-tol, default 2 min) against
    `system:time_start_s1`. Same-date-different-orbit is NOT a match: the known
    false alarm is training 2020-10-26T00:10:25 (Louisiana, lon -92) vs Demak
    eval 2020-10-26T22:17:27 -- ~22 h apart, different orbit, different site.

Geography comes from the per-chip `CHIP_BBOX` (WGS84) in the build parquet and
the file-sourced site AOIs in FIG01's `sites.py` (reused, never retyped, per the
project's "no eyeballed numbers" rule).

KNOWN-GOOD REGRESSION ANCHORS (asserted with --self-check)
    trained-on chips                     1,474,047
    distinct trained-on S1 scenes            3,196
    Hampyeong training pairs      PAIR_1385 / 3439 / 3442   (user-supplied list)
    Tier-1 leaks                  Hampyeong 20210329 (81 chips)
                                  Demak     20230812 (94 chips)
                                  Rockefeller 20210211 (15 chips)

Usage
-----
    conda run -n fes python scripts/analysis/eval_training_overlap_audit.py \\
        --out-dir docs/roadmap_for_publication/dataset_build_evidence/eval_overlap

    # fast re-check of the published numbers, no CSV written
    conda run -n fes python scripts/analysis/eval_training_overlap_audit.py --self-check

Read-only: never writes outside --out-dir.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "docs/roadmap_for_publication/figures/FIG01_dataset_map"))
from sites import SITES, validate as validate_sites  # noqa: E402

# The build parquet is the only artifact carrying BOTH scene id and chip bbox.
# See dataset-build ledger: only the *_PATHS_FIXED file is the real build input.
PARQUET = Path(
    "/Volumes/noel_wd_black_sn850x/segwater-database/DATABASE/SLIM_PARQUET/"
    "Global_Sen12_Coast_2017_2024_CHIPS_DATABASE_with_splits_w_metadata_w_tides"
    "_w_path_passedQC_memmap_selected-V2-w_histograms-CONFIRMED-RESPLIT_PATHS_FIXED.parquet"
)
SDS_SWEEP_ROOT = Path("/Volumes/noel_wd_black_sn850x/segwater_v2/experiments")

# Canonical (headline) frame per SDS site. Alternate-frame sweeps (_ron64,
# _MHWS, _RAW, ...) are sensitivity analyses, not the reported results.
SDS_FRAMES = {
    "narrabeen": "SDS_SWEEP",
    "duck": "SDS_SWEEP_DUCK_ron_4_ts_24_sn_19",
    "torreypines": "SDS_SWEEP_TORREYPINES_ron_71",
    "trucvert": "SDS_SWEEP_TRUCVERT",
}
ROCKEFELLER_LOOKUP = REPO / "experiments/global_erosion/rockerfeller/ROCKERFELLER_scene_lookup.csv"
HAMPYEONG_PAIRING = REPO / "experiments/hampyeong/gee_frame_scenes/pairing_metadata.csv"
DEMAK_RUN = REPO / (
    "experiments/demak_semarang/inference/dev_sweep_all_demak_s19-full-series_"
    "20260704T171831Z_upernet_swin_base_224_native224_weighted_224_b0_s32"
)

SAFE_RE = re.compile(r"S1[AB]_IW_GRDH_1S[DS][VH]_\d{8}T\d{6}_\d{8}T\d{6}_[0-9A-F]{6}_[0-9A-F]{6}_[0-9A-F]{4}")
DEMAK_RE = re.compile(r"^S1_(\d{8})_(\d{6})_(\d+)_(\d+)_(\d+)$")

EXPECTED = {
    "trained_on_chips": 1_474_047,
    "trained_on_scenes": 3_196,
    "leaks": {  # site -> (eval date, chips inside AOI)
        "hampyeong": ("20210329", 81),
        "demak": ("20230812", 94),
        "rockefeller": ("20210211", 15),
    },
}


# --------------------------------------------------------------------------- #
# training side
# --------------------------------------------------------------------------- #
def load_training(parquet: Path) -> pd.DataFrame:
    """Trained-on chips only, with scene id, acquisition time and bbox."""
    if not parquet.exists():
        sys.exit(
            f"Build parquet not found:\n  {parquet}\n"
            "Mount the black SSD (noel_wd_black_sn850x) or pass --parquet."
        )
    cols = [
        "system:index_s1", "system:time_start_s1", "CHIP_BBOX",
        "pair_name", "chip_id", "split", "split_pair_based", "selected_for_memmap",
    ]
    t = pq.read_table(parquet, columns=cols)
    d = pd.DataFrame({c: t.column(c).to_pylist() for c in cols})
    # selected_for_memmap is nullable: only True means the chip was trained on.
    d = d[d["selected_for_memmap"] == True].reset_index(drop=True)  # noqa: E712
    d["t"] = pd.to_datetime(d["system:time_start_s1"], utc=True, format="mixed")
    # CHIP_BBOX is a JSON string "[lonmin, latmin, lonmax, latmax]" (WGS84).
    bb = np.full((len(d), 4), np.nan)
    for i, s in enumerate(d["CHIP_BBOX"]):
        if s:
            bb[i] = json.loads(s)
    d[["lon0", "lat0", "lon1", "lat1"]] = bb
    return d


# --------------------------------------------------------------------------- #
# evaluation side
# --------------------------------------------------------------------------- #
def _first_populated_gpkg(sweep_root: Path):
    """Sweep dirs are per-arm; some threshold dirs are empty, so scan for one
    that actually holds scene gpkgs. The scene SET is identical across arms and
    thresholds, so the first populated dir is representative."""
    if not sweep_root.exists():
        return None
    for run in sorted(sweep_root.iterdir()):
        if not run.is_dir():
            continue
        for thr in sorted(run.glob("thr_*")):
            g = thr / "gpkg"
            if g.is_dir() and any(g.iterdir()):
                return g
    return None


def eval_scenes() -> pd.DataFrame:
    """One row per (site, eval scene). `scene` is a SAFE name where available."""
    rows = []

    for site, frame in SDS_FRAMES.items():
        g = _first_populated_gpkg(SDS_SWEEP_ROOT / frame)
        if g is None:
            print(f"  ! {site}: no populated gpkg dir under {frame} (drive mounted?)")
            continue
        found = {m.group(0) for f in os.listdir(g) if (m := SAFE_RE.search(f))}
        rows += [{"site": site, "scene": s, "id_kind": "safe"} for s in sorted(found)]

    if ROCKEFELLER_LOOKUP.exists():
        txt = ROCKEFELLER_LOOKUP.read_text()
        rows += [{"site": "rockefeller", "scene": s, "id_kind": "safe"}
                 for s in sorted(set(SAFE_RE.findall(txt)))]

    if HAMPYEONG_PAIRING.exists():
        txt = HAMPYEONG_PAIRING.read_text()
        rows += [{"site": "hampyeong", "scene": s, "id_kind": "safe"}
                 for s in sorted(set(SAFE_RE.findall(txt)))]

    # Demak: compact ids only -- no SAFE name is stored anywhere in the repo.
    if DEMAK_RUN.exists():
        for n in sorted(os.listdir(DEMAK_RUN)):
            m = DEMAK_RE.match(n)
            if not m:
                continue
            hhmmss = m.group(2)
            rows.append({
                "site": "demak", "scene": n, "id_kind": "compact",
                "t": pd.Timestamp(
                    f"{m.group(1)} {hhmmss[:2]}:{hhmmss[2:4]}:{hhmmss[4:]}", tz="UTC"
                ),
            })

    df = pd.DataFrame(rows)
    if "t" not in df.columns:
        df["t"] = pd.NaT
    return df


# --------------------------------------------------------------------------- #
# geometry
# --------------------------------------------------------------------------- #
def site_bboxes() -> dict:
    validate_sites()
    return {s["key"]: s["bbox"] for s in SITES}


def _km_to_bbox(lon, lat, box) -> float:
    """0 inside the AOI, else approx great-circle distance to its edge."""
    lo0, la0, lo1, la1 = box
    dx = max(lo0 - lon, 0.0, lon - lo1)
    dy = max(la0 - lat, 0.0, lat - la1)
    return math.hypot(dx * 111.32 * math.cos(math.radians(lat)), dy * 110.57)


def chips_for_scene(train: pd.DataFrame, scene: str, box) -> dict:
    sub = train[train["system:index_s1"] == scene]
    ok = sub.dropna(subset=["lon0"])
    inside = ok[(ok.lon0 < box[2]) & (ok.lon1 > box[0])
                & (ok.lat0 < box[3]) & (ok.lat1 > box[1])]
    d = [_km_to_bbox((r.lon0 + r.lon1) / 2, (r.lat0 + r.lat1) / 2, box)
         for r in ok.itertuples()]
    sp = inside["split"].value_counts().to_dict()
    return {
        "pairs": ",".join(sorted(sub["pair_name"].unique())),
        "chips_in_scene": int(len(sub)),
        "chips_in_aoi": int(len(inside)),
        "min_dist_km": round(float(min(d)), 2) if d else float("nan"),
        "median_dist_km": round(float(np.median(d)), 1) if d else float("nan"),
        "aoi_split_train": int(sp.get("train", 0)),
        "aoi_split_val": int(sp.get("val", 0)),
        "aoi_split_test": int(sp.get("test", 0)),
    }


# --------------------------------------------------------------------------- #
# audit
# --------------------------------------------------------------------------- #
def audit(train: pd.DataFrame, ev: pd.DataFrame, time_tol_min: float) -> pd.DataFrame:
    boxes = site_bboxes()
    scene_t = (train.drop_duplicates("system:index_s1")
                    .set_index("system:index_s1")["t"])
    train_scene_set = set(scene_t.index)
    tol = pd.Timedelta(minutes=time_tol_min)

    out = []
    for r in ev.itertuples():
        box = boxes.get(r.site)
        if box is None:
            continue
        matched, how = None, None
        if r.id_kind == "safe":
            if r.scene in train_scene_set:
                matched, how = r.scene, "exact-id"
        else:
            # timestamp match; a different orbit on the same day is >1 h away
            # and so is correctly rejected by the tolerance.
            dt = (scene_t - r.t).abs()
            if len(dt) and dt.min() <= tol:
                matched, how = dt.idxmin(), f"time<={time_tol_min:g}min"
        if matched is None:
            continue
        rec = {"site": r.site, "eval_scene": r.scene, "train_scene": matched,
               "match": how}
        rec.update(chips_for_scene(train, matched, box))
        rec["tier"] = "1_LEAK" if rec["chips_in_aoi"] > 0 else "2_SHARED_SWATH"
        out.append(rec)

    cols = ["site", "tier", "eval_scene", "train_scene", "match", "pairs",
            "chips_in_scene", "chips_in_aoi", "min_dist_km", "median_dist_km",
            "aoi_split_train", "aoi_split_val", "aoi_split_test"]
    df = pd.DataFrame(out, columns=cols)
    return df.sort_values(["tier", "site", "eval_scene"]).reset_index(drop=True)


def nearby(train: pd.DataFrame) -> pd.DataFrame:
    """Tier 3: any trained-on chip inside a site AOI, regardless of date."""
    rows = []
    for key, box in site_bboxes().items():
        ok = train.dropna(subset=["lon0"])
        m = ok[(ok.lon0 < box[2]) & (ok.lon1 > box[0])
               & (ok.lat0 < box[3]) & (ok.lat1 > box[1])]
        if m.empty:
            continue
        rows.append({
            "site": key,
            "chips_in_aoi": int(len(m)),
            "scenes": int(m["system:index_s1"].nunique()),
            "pairs": ",".join(sorted(m["pair_name"].unique())),
        })
    return (pd.DataFrame(rows).sort_values("chips_in_aoi", ascending=False)
            .reset_index(drop=True))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--parquet", type=Path, default=PARQUET)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--time-tol", type=float, default=2.0,
                    help="minutes; timestamp tolerance for compact-id sites")
    ap.add_argument("--self-check", action="store_true",
                    help="assert the published anchors, write nothing")
    a = ap.parse_args()

    print(f"Reading trained-on chips from\n  {a.parquet}")
    train = load_training(a.parquet)
    n_chips, n_scenes = len(train), train["system:index_s1"].nunique()
    print(f"  trained-on chips {n_chips:,} | distinct S1 scenes {n_scenes:,}")

    print("Collecting evaluation scenes ...")
    ev = eval_scenes()
    for s, n in ev.groupby("site").size().items():
        print(f"  {s:<12}{n:>5}")

    res = audit(train, ev, a.time_tol)
    nb = nearby(train)

    leaks = res[res.tier == "1_LEAK"]
    shared = res[res.tier == "2_SHARED_SWATH"]
    print(f"\nTier 1 LEAK           {len(leaks)} scenes, "
          f"{int(leaks.chips_in_aoi.sum())} chips inside evaluated AOIs")
    print(f"Tier 2 SHARED_SWATH   {len(shared)} scenes, 0 chips in AOI "
          f"(min dist {shared.min_dist_km.min() if len(shared) else float('nan'):.1f} km)")
    print(f"Tier 3 NEARBY         {int(nb.chips_in_aoi.sum())} chips across {len(nb)} sites\n")
    if len(res):
        print(res.to_string(index=False))
    print("\nTier 3 (any trained-on chip in AOI, any date):")
    print(nb.to_string(index=False))

    if n_chips:
        print(f"\nLeak fraction: {int(leaks.chips_in_aoi.sum())}/{n_chips:,} = "
              f"{100 * int(leaks.chips_in_aoi.sum()) / n_chips:.4f}% of training")

    if a.self_check:
        assert n_chips == EXPECTED["trained_on_chips"], f"chips {n_chips}"
        assert n_scenes == EXPECTED["trained_on_scenes"], f"scenes {n_scenes}"
        got = {r.site: (r.eval_scene, r.chips_in_aoi) for r in leaks.itertuples()}
        for site, (date, chips) in EXPECTED["leaks"].items():
            assert site in got, f"missing expected leak at {site}"
            assert date in got[site][0], f"{site}: {got[site][0]} !~ {date}"
            assert got[site][1] == chips, f"{site}: {got[site][1]} != {chips}"
        hp = set(nb.loc[nb.site == "hampyeong", "pairs"].iloc[0].split(","))
        assert {"PAIR_1385", "PAIR_3439", "PAIR_3442"} <= hp, hp
        print("\nself-check PASSED (counts, 3 leaks, Hampyeong pair list)")
        return

    if a.out_dir:
        a.out_dir.mkdir(parents=True, exist_ok=True)
        res.to_csv(a.out_dir / "eval_training_overlap_scenes.csv", index=False)
        nb.to_csv(a.out_dir / "eval_training_overlap_nearby.csv", index=False)
        print(f"\nwrote -> {a.out_dir}/eval_training_overlap_{{scenes,nearby}}.csv")


if __name__ == "__main__":
    main()
