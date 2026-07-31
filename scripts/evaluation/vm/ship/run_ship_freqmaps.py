"""S2-vs-S1 inundation-frequency spatial agreement for the SHIP campaign arms.

Reimplements the estimand of the registered analysis
(`optical_validation_comparisson/scripts/11_s2_frequency_maps.py`) against
mx630_stage2 campaign rasters, keeping every constant that defines it:

    THR 0.5 | GATE 0.9 (valid_frac_aoi) | MIN_OBS 10 (full) / 5 (epoch)
    EPOCH_EARLY 2017-2019 | EPOCH_LATE 2022-2024
    S1 field = 3-seed MEDIAN of per-seed frequency of P>THR
    gain/loss = |dfreq| > 0.5 between epoch means
    30 m tolerance (3 S2 px) for the coverage metric

⚠️ **Lineage.** The 20 registered CITABLE rows were computed on a CHIP-BASED S1
model. These rows are mx630_stage2. They are a COMPARISON, never a drop-in
replacement, and every output row carries `lineage`, `variant` and `stride`.

Usage:
    python run_ship_freqmaps.py --variant last --stride 32
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import from_bounds
from scipy.ndimage import distance_transform_edt
from scipy.stats import pearsonr

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ship_core_shim as shim  # noqa: E402

HOME = Path.home()
LAND_MASK_TIF = HOME / "ancillary/demak_semarang/aoi/GSHHG_mask.tif"
S2_DIR = HOME / "ancillary/demak_semarang/s2_timeseries_demak"
S2_TS = HOME / "proxy17_analysis/s2_voteveto_timeseries.csv"
# Default reproduces the Swin-B `ship` campaign; --out-dir/--tag redirect it.
DEFAULT_OUT_DIR = HOME / "workspace/results/ship_decision_2026-07/freq_maps"
# Campaign tag -> lineage slug for the `lineage` column. The slug, not the
# tag, is what makes a row poolable: checkpoint filenames collide across all
# three lineages, so `dataset=` alone is never sufficient provenance.
LINEAGE_BY_TAG = {"ship": "mx630s2", "cnxb": "mx630s2cnx", "cnxt": "mx630s2cnxt"}

THR = 0.5
GATE = 0.9
MIN_OBS, MIN_OBS_EPOCH = 10, 5
EPOCH_EARLY, EPOCH_LATE = (2017, 2019), (2022, 2024)
DFREQ = 0.5
TOL_M = 30.0


def _mpd(lat_rad):
    return (111132.954 - 559.822 * np.cos(2 * lat_rad) + 1.175 * np.cos(4 * lat_rad),
            111412.84 * np.cos(lat_rad) - 93.5 * np.cos(3 * lat_rad) + 0.118 * np.cos(5 * lat_rad))


def load_aoi(sample_tif):
    """Common analysis grid = GSHHG mask extent as a window into the rasters."""
    with rasterio.open(LAND_MASK_TIF) as mds:
        land = mds.read(1) == 1
        m_bounds, m_tr = mds.bounds, mds.transform
    with rasterio.open(sample_tif) as pds:
        win = from_bounds(*m_bounds, transform=pds.transform).round_offsets().round_lengths()
        if abs(pds.transform.a - m_tr.a) > 1e-12:
            raise ValueError("pixel size mismatch vs land mask")
    if not (win.col_off == 0 and win.row_off == 0):
        raise ValueError("AOI window not top-left: %s" % (win,))
    row = np.arange(land.shape[0]) + 0.5
    lat = np.deg2rad(m_tr.f + m_tr.e * row)
    m_lat, m_lon = _mpd(lat)
    px = abs(m_tr.a)
    px_ha = (((px * m_lat) * (px * m_lon)) / 1e4).reshape(-1, 1)
    px_m = (float(np.median(px * m_lat)), float(np.median(px * m_lon)))
    return land, win, px_ha, px_m


def s2_scenes():
    ts = pd.read_csv(S2_TS, parse_dates=["datetime"])
    ts = ts[ts.valid_frac_aoi >= GATE].copy()
    ts["path"] = ts.filename.map(lambda f: str(S2_DIR / f))
    missing = [p for p in ts.path if not Path(p).exists()]
    if missing:
        raise FileNotFoundError("%d gated S2 masks missing, e.g. %s" % (len(missing), missing[0]))
    ts["year"] = ts.datetime.dt.year
    return ts.sort_values("datetime").reset_index(drop=True)


def s2_freq(scenes, shape, win):
    sel = {"full": np.ones(len(scenes), bool),
           "early": scenes.year.between(*EPOCH_EARLY).values,
           "late": scenes.year.between(*EPOCH_LATE).values}
    aw = {k: np.zeros(shape, np.uint16) for k in sel}
    av = {k: np.zeros(shape, np.uint16) for k in sel}
    for i, row in enumerate(scenes.itertuples()):
        with rasterio.open(row.path) as ds:
            a = ds.read(1, window=win)
        valid, water = a != 255, a == 1
        for k, m in sel.items():
            if m[i]:
                aw[k] += water
                av[k] += valid
    out = {}
    for k in sel:
        mo = MIN_OBS if k == "full" else MIN_OBS_EPOCH
        f = aw[k].astype(np.float32) / np.maximum(av[k], 1)
        f[av[k] < mo] = np.nan
        out[k] = f
    out["_n"] = {k: int(m.sum()) for k, m in sel.items()}
    return out


def s1_freq(shape, win):
    """3-seed median of per-seed P>THR frequency, for full / early / late."""
    per_seed = {"full": [], "early": [], "late": []}
    for seed in shim.SEEDS:
        idx = shim.scene_index(seed)
        yr = idx.datetime.dt.year
        sel = {"full": np.ones(len(idx), bool),
               "early": yr.between(*EPOCH_EARLY).values,
               "late": yr.between(*EPOCH_LATE).values}
        acc = {k: np.zeros(shape, np.float64) for k in sel}
        for i, r in enumerate(idx.itertuples()):
            with rasterio.open(r.tif) as ds:
                w = ds.read(1, window=win) > THR
            for k, m in sel.items():
                if m[i]:
                    acc[k] += w
        for k, m in sel.items():
            per_seed[k].append((acc[k] / max(int(m.sum()), 1)).astype(np.float32))
    return ({k: np.median(np.stack(v), axis=0) for k, v in per_seed.items()},
            {k: int(m.sum()) for k, m in sel.items()})


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--variant", required=True, choices=("best", "last"))
    ap.add_argument("--stride", required=True, type=int, choices=(32, 112))
    ap.add_argument("--tag", default="ship",
                    help="campaign tag whose run dirs to read (default: "
                         "%(default)s). Must match the middle token of the "
                         "sweep names, e.g. ship|cnxb|cnxt.")
    ap.add_argument("--lineage", default=None,
                    help="lineage slug for the `lineage` column; defaults to the "
                         "slug registered for --tag. MUST differ between lineages "
                         "whose checkpoint filenames collide.")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="results dir (default: %s)" % DEFAULT_OUT_DIR)
    a = ap.parse_args()
    lineage = a.lineage or LINEAGE_BY_TAG.get(a.tag)
    if lineage is None:
        raise SystemExit(
            "unknown campaign tag %r: pass --lineage explicitly, or add it to "
            "LINEAGE_BY_TAG. Refusing to guess -- a wrong lineage slug is how "
            "two lineages get silently pooled." % a.tag)
    shim.configure(a.variant, a.stride, tag=a.tag)
    OUT_DIR = a.out_dir or DEFAULT_OUT_DIR
    tag = "%s_s%d" % (a.variant, a.stride)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("=== campaign %s / lineage %s -> %s ===" % (a.tag, lineage, OUT_DIR))

    audit = shim.audit()
    print("=== provenance (%s) ===" % tag)
    print(audit[["seed", "run_dir", "encoder", "amp_dtype"]].to_string(index=False))

    sample = shim.scene_index("s19").iloc[0].tif
    land, win, px_ha, px_m = load_aoi(sample)
    print("AOI %s  px %.1f x %.1f m" % (land.shape, px_m[0], px_m[1]))

    s2s = s2_scenes()
    print("S2 gated scenes: %d" % len(s2s))
    f2 = s2_freq(s2s, land.shape, win)
    f1, n1 = s1_freq(land.shape, win)
    print("S1 scenes full/early/late: %s" % n1)

    rows = []
    def add(metric, value, pair="S2 vs S1"):
        rows.append(dict(lineage=lineage, variant=a.variant, stride=a.stride,
                         pair=pair, metric=metric, value=value))

    # --- full-period frequency agreement --------------------------------
    m = land & ~np.isnan(f2["full"]) & ~np.isnan(f1["full"])
    d = f2["full"][m] - f1["full"][m]
    add("n_px", int(m.sum()))
    add("pearson_r", float(pearsonr(f2["full"][m], f1["full"][m])[0]))
    add("MAE", float(np.abs(d).mean()))
    add("bias_S2_minus_S1", float(d.mean()))

    # --- 3-class agreement ----------------------------------------------
    def cls(f):
        c = np.full(f.shape, 255, np.uint8)
        ok = ~np.isnan(f)
        c[ok & (f < 0.1)] = 0
        c[ok & (f >= 0.1) & (f <= 0.9)] = 1
        c[ok & (f > 0.9)] = 2
        return c
    c2, c1 = cls(f2["full"]), cls(f1["full"])
    add("class_agreement_frac", float((c2[m] == c1[m]).mean()))

    # --- gain masks and their spatial agreement -------------------------
    w = np.broadcast_to(px_ha, land.shape)
    res = {}
    for lbl, f in (("S2", f2), ("S1", f1)):
        dfq = f["late"] - f["early"]
        res[lbl] = dict(gain=land & (dfq > DFREQ), loss=land & (dfq < -DFREQ))
        add("%s_gain_ha" % lbl, float(w[res[lbl]["gain"]].sum()))
        add("%s_loss_ha" % lbl, float(w[res[lbl]["loss"]].sum()))

    ga, gb = res["S2"]["gain"], res["S1"]["gain"]
    inter, union = ga & gb, ga | gb
    add("gain_IoU", float(w[inter].sum() / w[union].sum()) if union.any() else np.nan)
    da = distance_transform_edt(~ga, sampling=px_m)
    db = distance_transform_edt(~gb, sampling=px_m)
    a_ha, b_ha = float(w[ga].sum()), float(w[gb].sum())
    add("gain_S2_within_30m_of_S1", float(w[ga & (db <= TOL_M)].sum() / a_ha) if a_ha else np.nan)
    add("gain_S1_within_30m_of_S2", float(w[gb & (da <= TOL_M)].sum() / b_ha) if b_ha else np.nan)
    if (ga & ~gb).any():
        add("median_dist_S2only_to_S1_m", float(np.median(db[ga & ~gb])))
    if (gb & ~ga).any():
        add("median_dist_S1only_to_S2_m", float(np.median(da[gb & ~ga])))

    df = pd.DataFrame(rows)
    out = OUT_DIR / ("freq_stats_%s.csv" % tag)
    df.to_csv(out, index=False)
    audit.to_csv(OUT_DIR / ("provenance_%s.csv" % tag), index=False)
    print("\nwrote %s (%d rows)" % (out, len(df)))
    print(df[["metric", "value"]].to_string(index=False))


if __name__ == "__main__":
    main()
