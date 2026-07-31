"""Per-scene water area for the 17-scene Demak trend proxy — VM-side, self-contained.

Ports the AOI/area logic from experiments/demak_semarang/scripts/sem_core.py
VERBATIM (_meters_per_degree, load_aoi window+pixel_area_ha, read_prob) so the
areas are directly comparable to the existing proxy-trend CSVs. sem_core itself
is not importable here: it pulls in Mac-side tide/exclusion CSV paths.

Emits one long CSV over all four arms (aug/no_aug x best/last).
"""
from __future__ import annotations
import json, re, sys
from pathlib import Path
import numpy as np, pandas as pd, rasterio, yaml
from rasterio.windows import from_bounds

RUNS = Path.home() / "segwater_v2/outputs/inference/runs"
LAND_MASK_TIF = Path.home() / "ancillary/demak_semarang/aoi/GSHHG_mask.tif"
OUT = Path.home() / "demak_proxy17_water_area_timeseries.csv"
THRESHOLDS = (0.3, 0.4, 0.5, 0.6, 0.7)
SCENE_RE = re.compile(r"^S1_(\d{8})_(\d{6})_(.+)$")
A_PRIORI_EXCLUSIONS = {"S1_20250730_105715_127_1_1"}
ARMS = [("aug","best"),("aug","last"),("no_aug","best"),("no_aug","last")]


def _meters_per_degree(lat_rad):
    m_lat = 111132.954 - 559.822*np.cos(2*lat_rad) + 1.175*np.cos(4*lat_rad)
    m_lon = 111412.84*np.cos(lat_rad) - 93.5*np.cos(3*lat_rad) + 0.118*np.cos(5*lat_rad)
    return m_lat, m_lon


def load_aoi(sample_tif):
    with rasterio.open(LAND_MASK_TIF) as mds:
        land = mds.read(1) == 1
        m_bounds, m_transform = mds.bounds, mds.transform
    with rasterio.open(sample_tif) as pds:
        win = from_bounds(*m_bounds, transform=pds.transform).round_offsets().round_lengths()
        if not (abs(pds.transform.a-m_transform.a) < 1e-12 and abs(pds.transform.e-m_transform.e) < 1e-12):
            raise ValueError("pixel sizes differ")
        if (win.height, win.width) != land.shape:
            raise ValueError("window %s != mask shape %s" % (win, land.shape))
    row = np.arange(land.shape[0]) + 0.5
    lat_rad = np.deg2rad(m_transform.f + m_transform.e*row)
    m_lat, m_lon = _meters_per_degree(lat_rad)
    px = abs(m_transform.a)
    return land, win, (((px*m_lat)*(px*m_lon))/1e4).reshape(-1,1)


def find_run(aug, arm):
    hits = [d for d in sorted(RUNS.glob("demak_proxy17_%s_%s_*" % (aug, arm))) if d.is_dir()]
    assert len(hits) == 1, "%s/%s: %d run dirs" % (aug, arm, len(hits))
    return hits[0]


def audit(base, aug, arm):
    cfg = yaml.safe_load((base/"run_config.yaml").read_text())
    ck = cfg["inference"]["checkpoint_path"]
    exp_dir = "s42_no_aug" if aug == "no_aug" else "s42"
    assert "/%s/" % exp_dir in ck, "%s/%s: wrong seed dir in %s" % (aug, arm, ck)
    assert ("pmwiou" in ck) == (arm == "best"), "%s/%s: arm/ckpt mismatch %s" % (aug, arm, ck)
    assert cfg["inference"]["data"]["stride"] == 32
    return ck


def main():
    rows, seen_ck = [], {}
    for aug, arm in ARMS:
        base = find_run(aug, arm)
        ck = audit(base, aug, arm)
        assert ck not in seen_ck or seen_ck[ck] == (aug, arm), "duplicate ckpt %s" % ck
        seen_ck[ck] = (aug, arm)
        scenes = []
        for d in sorted(base.iterdir()):
            if not d.is_dir() or d.name in A_PRIORI_EXCLUSIONS: continue
            m = SCENE_RE.match(d.name)
            if m is None: continue
            t = sorted(d.glob("*_probability_water.tif"))
            assert len(t) == 1, "%s: %d tifs" % (d, len(t))
            scenes.append((d.name, pd.to_datetime(m.group(1)+m.group(2), format="%Y%m%d%H%M%S", utc=True), t[0]))
        assert len(scenes) == 17, "%s/%s: %d scenes (expect 17)" % (aug, arm, len(scenes))
        scenes.sort(key=lambda r: r[1])
        land, win, px_ha = load_aoi(scenes[0][2])
        land_w = np.where(land, np.broadcast_to(px_ha, land.shape), 0.0)
        for sid, dt, tif in scenes:
            with rasterio.open(tif) as ds:
                p = ds.read(1, window=win).astype(np.float64)
            nan = ~np.isfinite(p); n_nan = int(nan[land].sum())
            if n_nan: p = np.where(nan, 0.0, p)
            r = {"scene_id": sid, "datetime": dt, "augmentation": aug, "arm": arm, "seed": "s42",
                 "ckpt_file": ck.split("/")[-1],
                 "nan_frac_aoi": n_nan/land.sum(), "mean_prob_aoi": float(p[land].mean()),
                 "expected_area_ha": float((p*land_w).sum())}
            for thr in THRESHOLDS:
                r["area_ha_thr%.1f" % thr] = float(land_w[p > thr].sum())
            rows.append(r)
        print("  %-7s %-5s %2d scenes  %s" % (aug, arm, len(scenes), ck.split("/")[-1]))
    df = pd.DataFrame(rows).sort_values(["augmentation","arm","datetime"])
    df.to_csv(OUT, index=False)
    print("\nwrote %s  (%d rows = 4 arms x 17 scenes)" % (OUT, len(df)))
    print("distinct checkpoints: %d (expect 4)" % df.ckpt_file.nunique())


if __name__ == "__main__":
    main()
