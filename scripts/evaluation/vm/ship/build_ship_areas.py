"""Per-scene water area for the 6 mx630_stage2 SHIP arms (213 scenes each).

Arms = 3 seeds (s19, s42, s58) x 2 checkpoint variants (best, last), all
Swin-B, all bf16 + TF32 + device-stitch. Run dirs are produced one sweep per
stride, so this script takes ``--stride {32,112}`` and reads the matching set.

AOI/area logic ported verbatim from ``build_full5_areas.py`` (itself ported
from sem_core.py) so areas stay comparable to the registered trend products.
Emits one long CSV keyed by ``arm`` (= ``<seed>_<variant>``) with ``seed`` and
``variant`` broken out as their own columns, plus an ``in_analysis_window``
flag for the 206-scene window.

Usage:
    python scripts/evaluation/vm/ship/build_ship_areas.py --stride 112
"""
from __future__ import annotations
import argparse, re, sys
from pathlib import Path
import numpy as np, pandas as pd, rasterio, yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from runsel import resolve_run_dir

RUNS = Path.home()/"segwater_v2/outputs/inference/runs"
LAND_MASK_TIF = Path.home()/"ancillary/demak_semarang/aoi/GSHHG_mask.tif"
OUT_DIR = Path.home()/"workspace/results/ship_decision_2026-07/demak_trend"
THRESHOLDS = (0.3, 0.4, 0.5, 0.6, 0.7)
SCENE_RE = re.compile(r"^S1_(\d{8})_(\d{6})_(.+)$")
A_PRIORI_EXCLUSIONS = {"S1_20250730_105715_127_1_1"}
WINDOW_END = pd.Timestamp("2024-12-31", tz="UTC")

SEEDS = ["s19", "s42", "s58"]
VARIANTS = ["best", "last"]          # default arm set; --variants overrides
ALL_VARIANTS = ["best", "last", "swa5"]
ARMS = ["%s_%s" % (s, v) for s in SEEDS for v in VARIANTS]
# Default lineage is Swin-B mx630_stage2 (`--encoder` overrides). The encoder
# guard runs per arm regardless, so a mis-wired config cannot slip a different
# backbone into the table.
DEFAULT_ENCODER = "tu-swin_base_patch4_window7_224"
DEFAULT_TAG = "ship"
EXPECT_ENC = {a: DEFAULT_ENCODER for a in ARMS}


def _mpd(lat_rad):
    return (111132.954 - 559.822*np.cos(2*lat_rad) + 1.175*np.cos(4*lat_rad),
            111412.84*np.cos(lat_rad) - 93.5*np.cos(3*lat_rad) + 0.118*np.cos(5*lat_rad))


def load_aoi(sample_tif):
    from rasterio.windows import from_bounds
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
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stride", type=int, required=True, choices=(32, 112))
    ap.add_argument("--variants", nargs="+", default=VARIANTS, choices=ALL_VARIANTS,
                    help="arm variants to build (default: %(default)s). Pass a "
                         "subset to add an arm without rebuilding the others.")
    ap.add_argument("--tag", default=DEFAULT_TAG,
                    help="campaign tag in the MIDDLE of the sweep name "
                         "(default: %(default)s)")
    ap.add_argument("--encoder", default=DEFAULT_ENCODER,
                    help="encoder every run_config.yaml must declare "
                         "(default: %(default)s)")
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR,
                    help="results dir (default: %(default)s)")
    a = ap.parse_args()
    arms = ["%s_%s" % (sd, v) for sd in SEEDS for v in a.variants]
    expect_enc = {x: a.encoder for x in arms}
    # The filename carries the arm set AND the campaign tag: a --variants subset
    # must NOT overwrite the full-campaign CSV, and two lineages must not
    # overwrite each other. Default (ship + best/last) keeps the historical name
    # so existing consumers are unaffected.
    vtag = "" if list(a.variants) == VARIANTS else "_" + "-".join(a.variants)
    out = a.out_dir/("demak_full_%s_areas%s_s%d.csv" % (a.tag, vtag, a.stride))
    out.parent.mkdir(parents=True, exist_ok=True)

    rows, seen = [], {}
    for arm in arms:
        seed, variant = arm.split("_")
        base = resolve_run_dir(RUNS, "demak_full_%s_%s_s%d" % (a.tag, arm, a.stride))
        cfg = yaml.safe_load((base/"run_config.yaml").read_text())
        ck = cfg["inference"]["checkpoint_path"]
        enc = cfg["model"]["encoder_name"]
        assert enc == expect_enc[arm], "%s: encoder %s != %s" % (arm, enc, expect_enc[arm])
        assert ck not in seen, "duplicate ckpt: %s (%s and %s)" % (ck, seen.get(ck), arm)
        seen[ck] = arm
        assert cfg["inference"]["data"]["stride"] == a.stride, \
            "%s: stride %s != %d" % (arm, cfg["inference"]["data"]["stride"], a.stride)
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
            r = {"scene_id": sid, "datetime": dt, "arm": arm,
                 "seed": seed, "variant": variant, "stride": a.stride,
                 "encoder": enc, "ckpt_file": ck.split("/")[-1],
                 "in_analysis_window": bool(dt <= WINDOW_END),
                 "nan_frac_aoi": n_nan/land.sum(), "mean_prob_aoi": float(p[land].mean()),
                 "expected_area_ha": float((p*land_w).sum())}
            for thr in THRESHOLDS:
                r["area_ha_thr%.1f" % thr] = float(land_w[p > thr].sum())
            rows.append(r)
        print("  %-10s %3d scenes  %-24s %s" % (arm, len(scenes), enc, ck.split("/")[-1][:52]))
    df = pd.DataFrame(rows).sort_values(["arm", "datetime"])
    df.to_csv(out, index=False)
    print("\nwrote %s (%d rows = %d arms x 213)" % (out, len(df), len(arms)))
    for arm in arms:
        n_win = int(df[df.arm == arm].in_analysis_window.sum())
        assert n_win == 206, "%s: in-window %d != 206" % (arm, n_win)
        print("  %-10s in analysis window: %d" % (arm, n_win))
    print("distinct checkpoints: %d / %d arms" % (len(seen), len(arms)))


if __name__ == "__main__":
    main()
