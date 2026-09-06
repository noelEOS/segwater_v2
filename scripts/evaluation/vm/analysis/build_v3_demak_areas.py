"""Per-scene water area for the 6 dataset_v3 Demak full-series arms (213 scenes).

Arms = 2 architectures (Swin-B, ConvNeXtV2-Base) x 3 seeds, all deployed on the
``*_last.pth`` checkpoint, all bf16 + TF32 + device-stitch, stride 32.

**The AOI/area logic is ported verbatim from ``ship/build_ship_areas.py``** (which
took it from ``build_full5_areas.py``, itself from ``sem_core.py``) so v3 areas
are comparable to the registered trend products rather than a parallel
computation that happens to look similar. What differs here is only how run dirs
are found: this lineage names them ``v3_demak_full_<arch>_<seed>_<stamp>_...``,
not ``demak_full_<tag>_<seed>_<variant>_s<stride>``.

**The mask is ``GSHHG_mask.tif`` (GSHHG-only, 3,242,989 px = 32,430 ha).** That
is the adopted Demak domain for both trend and accuracy, decided so one domain
serves both analyses; it is already the trend denominator. The competing
``GSHHG_GlobalSurfaceWater_combined_mask.tif`` (2,958,577 px) carves out
permanent open water and is superseded -- do not substitute it, and note a bare
"Demak area" is ambiguous until the mask is named.

The mask grid (2144x2144) is smaller than the scene grid (2175x2160); every
scene is read through a window derived from the mask bounds, with pixel size
asserted equal, so the two always describe the same ground.

Per-pixel area is computed from the latitude-dependent metres-per-degree, not a
constant, because the AOI spans enough latitude for a flat factor to bias the
total.

Usage:
    python scripts/evaluation/vm/analysis/build_v3_demak_areas.py
    python scripts/evaluation/vm/analysis/build_v3_demak_areas.py --archs swinb
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from runsel import resolve_run_dir  # noqa: E402

RUNS = Path.home() / "segwater_v2/outputs/inference/runs"
LAND_MASK_TIF = Path.home() / "ancillary/demak_semarang/aoi/GSHHG_mask.tif"
OUT_DIR = Path.home() / "demak_trend_v3"
THRESHOLDS = (0.3, 0.4, 0.5, 0.6, 0.7)
SCENE_RE = re.compile(r"^S1_(\d{8})_(\d{6})_(.+)$")
EXPECT_SCENES = 213
# The registered trend window. Scenes after it are built but flagged out, so the
# fitted n can never drift silently with the archive.
WINDOW_END = pd.Timestamp("2024-12-31", tz="UTC")
EXPECT_IN_WINDOW = 206
EXPECT_MASK_PX = 3_242_989

SEEDS = ["s19", "s42", "s58"]
ARCHS = {
    "swinb": "tu-swin_base_patch4_window7_224",
    "cnxb": "tu-convnextv2_base",
}


def _mpd(lat_rad):
    """Metres per degree of latitude and longitude at a given latitude."""
    return (111132.954 - 559.822 * np.cos(2 * lat_rad) + 1.175 * np.cos(4 * lat_rad),
            111412.84 * np.cos(lat_rad) - 93.5 * np.cos(3 * lat_rad) + 0.118 * np.cos(5 * lat_rad))


def load_aoi(sample_tif):
    """Return (land mask, read window into a scene, per-row pixel area in ha)."""
    from rasterio.windows import from_bounds
    with rasterio.open(LAND_MASK_TIF) as mds:
        land = mds.read(1) == 1
        m_bounds, m_tr = mds.bounds, mds.transform
    n_px = int(land.sum())
    if n_px != EXPECT_MASK_PX:
        raise ValueError("GSHHG mask has %d valid px, expected %d -- wrong mask?"
                         % (n_px, EXPECT_MASK_PX))
    with rasterio.open(sample_tif) as pds:
        win = from_bounds(*m_bounds, transform=pds.transform).round_offsets().round_lengths()
        if not (abs(pds.transform.a - m_tr.a) < 1e-12 and abs(pds.transform.e - m_tr.e) < 1e-12):
            raise ValueError("pixel sizes differ between mask and scene")
        if (win.height, win.width) != land.shape:
            raise ValueError("window %s != mask %s" % (win, land.shape))
    row = np.arange(land.shape[0]) + 0.5
    lat = np.deg2rad(m_tr.f + m_tr.e * row)
    m_lat, m_lon = _mpd(lat)
    px = abs(m_tr.a)
    return land, win, (((px * m_lat) * (px * m_lon)) / 1e4).reshape(-1, 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--archs", nargs="+", default=list(ARCHS), choices=list(ARCHS))
    ap.add_argument("--stride", type=int, default=32)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    a = ap.parse_args()

    arms = ["%s_%s" % (arch, sd) for arch in a.archs for sd in SEEDS]
    atag = "" if list(a.archs) == list(ARCHS) else "_" + "-".join(a.archs)
    out = a.out_dir / ("demak_full_v3_areas%s_s%d.csv" % (atag, a.stride))
    out.parent.mkdir(parents=True, exist_ok=True)

    rows, seen = [], {}
    for arm in arms:
        arch, seed = arm.split("_")
        # Anchored on the UTC stamp the sweep always emits, so a name that is a
        # prefix of another cannot resolve to two dirs.
        base = resolve_run_dir(RUNS, "v3_demak_full_%s_%s" % (arch, seed))
        cfg = yaml.safe_load((base / "run_config.yaml").read_text())
        ck = cfg["inference"]["checkpoint_path"]
        enc = cfg["model"]["encoder_name"]
        if enc != ARCHS[arch]:
            raise AssertionError("%s: encoder %s != %s" % (arm, enc, ARCHS[arch]))
        if ck in seen:
            raise AssertionError("duplicate ckpt: %s (%s and %s)" % (ck, seen[ck], arm))
        seen[ck] = arm
        if cfg["inference"]["data"]["stride"] != a.stride:
            raise AssertionError("%s: stride %s != %d"
                                 % (arm, cfg["inference"]["data"]["stride"], a.stride))
        # The lineage guard: these areas are only comparable to each other if
        # every arm re-quantized its inputs the same way.
        enc_key = cfg["inference"]["data"].get("s1_encoding")
        if enc_key != "q55s005":
            raise AssertionError("%s: s1_encoding %r != 'q55s005'" % (arm, enc_key))

        scenes = []
        for d in sorted(base.iterdir()):
            if not d.is_dir():
                continue
            m = SCENE_RE.match(d.name)
            if m is None:
                continue
            t = sorted(d.glob("*_probability_water.tif"))
            if len(t) != 1:
                raise AssertionError("%s: %d probability tifs" % (d, len(t)))
            scenes.append((d.name,
                           pd.to_datetime(m.group(1) + m.group(2),
                                          format="%Y%m%d%H%M%S", utc=True),
                           t[0]))
        if len(scenes) != EXPECT_SCENES:
            raise AssertionError("%s: %d scenes != %d" % (arm, len(scenes), EXPECT_SCENES))
        scenes.sort(key=lambda r: r[1])

        land, win, px_ha = load_aoi(scenes[0][2])
        land_w = np.where(land, np.broadcast_to(px_ha, land.shape), 0.0)
        for sid, dt, tif in scenes:
            with rasterio.open(tif) as ds:
                p = ds.read(1, window=win).astype(np.float64)
            nan = ~np.isfinite(p)
            n_nan = int(nan[land].sum())
            if n_nan:
                p = np.where(nan, 0.0, p)
            r = {"scene_id": sid, "datetime": dt, "arm": arm,
                 "arch": arch, "seed": seed, "stride": a.stride,
                 "encoder": enc, "lineage": "v3", "ckpt_file": ck.split("/")[-1],
                 "in_analysis_window": bool(dt <= WINDOW_END),
                 "nan_frac_aoi": n_nan / land.sum(),
                 "mean_prob_aoi": float(p[land].mean()),
                 "expected_area_ha": float((p * land_w).sum())}
            for thr in THRESHOLDS:
                r["area_ha_thr%.1f" % thr] = float(land_w[p > thr].sum())
            rows.append(r)
        print("  %-12s %3d scenes  %-30s %s" % (arm, len(scenes), enc, ck.split("/")[-1][:46]))

    df = pd.DataFrame(rows).sort_values(["arm", "datetime"])
    df.to_csv(out, index=False)
    print("\nwrote %s (%d rows = %d arms x %d)" % (out, len(df), len(arms), EXPECT_SCENES))
    for arm in arms:
        n_win = int(df[df.arm == arm].in_analysis_window.sum())
        if n_win != EXPECT_IN_WINDOW:
            raise AssertionError("%s: in-window %d != %d" % (arm, n_win, EXPECT_IN_WINDOW))
        print("  %-12s in analysis window: %d" % (arm, n_win))
    print("distinct checkpoints: %d / %d arms" % (len(seen), len(arms)))


if __name__ == "__main__":
    main()
