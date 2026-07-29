"""Per-scene water area for the 3 mx630 full-series arms (213 scenes each).

AOI/area logic ported verbatim from sem_core.py so areas are comparable to the
registered trend products. Emits one long CSV keyed by `arm` (not `seed`, since
all three are s42) plus an `in_analysis_window` flag for the 206-scene window.
"""
from __future__ import annotations
import glob, os, re
from pathlib import Path
import numpy as np, pandas as pd, rasterio, yaml
from rasterio.windows import from_bounds

RUNS = Path.home()/"segwater_v2/outputs/inference/runs"
LAND_MASK_TIF = Path.home()/"ancillary/demak_semarang/aoi/GSHHG_mask.tif"
OUT = Path.home()/"demak_full_5arm_water_area_timeseries.csv"
THRESHOLDS = (0.3, 0.4, 0.5, 0.6, 0.7)
SCENE_RE = re.compile(r"^S1_(\d{8})_(\d{6})_(.+)$")
A_PRIORI_EXCLUSIONS = {"S1_20250730_105715_127_1_1"}
WINDOW_END = pd.Timestamp("2024-12-31", tz="UTC")
ARMS = ["mx630s2_best", "mx630s2_last", "mx630s2_swa5", "mx630k", "mx630k_best"]
EXPECT_ENC = {"mx630s2_best": "tu-swin_base_patch4_window7_224",
              "mx630s2_swa5": "tu-swin_base_patch4_window7_224",
              "mx630k_best": "tu-convnextv2_tiny",
              "mx630s2_last": "tu-swin_base_patch4_window7_224",
              "mx630k": "tu-convnextv2_tiny"}


def _mpd(lat_rad):
    return (111132.954 - 559.822*np.cos(2*lat_rad) + 1.175*np.cos(4*lat_rad),
            111412.84*np.cos(lat_rad) - 93.5*np.cos(3*lat_rad) + 0.118*np.cos(5*lat_rad))


def load_aoi(sample_tif):
    with rasterio.open(LAND_MASK_TIF) as mds:
        land = mds.read(1) == 1
        m_bounds, m_tr = mds.bounds, mds.transform
    with rasterio.open(sample_tif) as pds:
        win = from_bounds(*m_bounds, transform=pds.transform).round_offsets().round_lengths()
        if not (abs(pds.transform.a-m_tr.a) < 1e-12 and abs(pds.transform.e-m_tr.e) < 1e-12):
            raise ValueError("pixel sizes differ")
        if (win.height, win.width) != land.shape:
            raise ValueError("window %s != mask %s" % (win, land.shape))
    row = np.arange(land.shape[0]) + 0.5
    lat = np.deg2rad(m_tr.f + m_tr.e*row)
    m_lat, m_lon = _mpd(lat)
    px = abs(m_tr.a)
    return land, win, (((px*m_lat)*(px*m_lon))/1e4).reshape(-1, 1)


def main():
    rows, seen = [], {}
    for arm in ARMS:
        pat = re.compile(r"^demak_full_" + re.escape(arm) + r"_\d{8}T\d{6}Z_")
        hits = [d for d in sorted(RUNS.glob("demak_full_%s_*" % arm))
                if d.is_dir() and pat.match(d.name)]
        assert len(hits) == 1, "%s: %d run dirs" % (arm, len(hits))
        base = hits[0]
        cfg = yaml.safe_load((base/"run_config.yaml").read_text())
        ck = cfg["inference"]["checkpoint_path"]
        enc = cfg["model"]["encoder_name"]
        assert enc == EXPECT_ENC[arm], "%s: encoder %s != %s" % (arm, enc, EXPECT_ENC[arm])
        assert ck not in seen, "duplicate ckpt: %s" % ck
        seen[ck] = arm
        assert cfg["inference"]["data"]["stride"] == 32
        scenes = []
        for d in sorted(base.iterdir()):
            if not d.is_dir() or d.name in A_PRIORI_EXCLUSIONS:
                continue
            m = SCENE_RE.match(d.name)
            if m is None:
                continue
            t = sorted(d.glob("*_probability_water.tif"))
            assert len(t) == 1, "%s: %d tifs" % (d, len(t))
            scenes.append((d.name, pd.to_datetime(m.group(1)+m.group(2), format="%Y%m%d%H%M%S", utc=True), t[0]))
        assert len(scenes) == 213, "%s: %d scenes" % (arm, len(scenes))
        scenes.sort(key=lambda r: r[1])
        land, win, px_ha = load_aoi(scenes[0][2])
        land_w = np.where(land, np.broadcast_to(px_ha, land.shape), 0.0)
        for sid, dt, tif in scenes:
            with rasterio.open(tif) as ds:
                p = ds.read(1, window=win).astype(np.float64)
            nan = ~np.isfinite(p); n_nan = int(nan[land].sum())
            if n_nan:
                p = np.where(nan, 0.0, p)
            r = {"scene_id": sid, "datetime": dt, "arm": arm, "seed": "s42",
                 "encoder": enc, "ckpt_file": ck.split("/")[-1],
                 "in_analysis_window": bool(dt <= WINDOW_END),
                 "nan_frac_aoi": n_nan/land.sum(), "mean_prob_aoi": float(p[land].mean()),
                 "expected_area_ha": float((p*land_w).sum())}
            for thr in THRESHOLDS:
                r["area_ha_thr%.1f" % thr] = float(land_w[p > thr].sum())
            rows.append(r)
        print("  %-14s %3d scenes  %-24s %s" % (arm, len(scenes), enc, ck.split("/")[-1][:52]))
    df = pd.DataFrame(rows).sort_values(["arm", "datetime"])
    df.to_csv(OUT, index=False)
    print("\nwrote %s (%d rows = %d arms x 213)" % (OUT, len(df), len(ARMS)))
    for a in ARMS:
        print("  %-14s in analysis window: %d"
              % (a, df[df.arm == a].in_analysis_window.sum()))


if __name__ == "__main__":
    main()
