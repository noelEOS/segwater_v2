"""S2-vs-S1 inundation-frequency spatial agreement for the SHIP campaign arms.

Reimplements the estimand of the registered analysis
(`optical_validation_comparisson/scripts/11_s2_frequency_maps.py`) against
mx630_stage2 campaign rasters, keeping every constant that defines it:

    THR 0.5 | GATE 0.9 (valid_frac_aoi) | MIN_OBS 10 (full) / 5 (epoch)
    EPOCH_EARLY 2017-2019 | EPOCH_LATE 2022-2024
    S1 field = 3-seed MEDIAN of per-seed frequency of P>THR
    gain/loss = |dfreq| > 0.5 between epoch means
    30 m tolerance (3 S2 px) for the coverage metric

⚠️ **Lineage.** The 20 registered rows (18 CITABLE) in
`experiments/demak_semarang/results_registry/` were computed on a DIFFERENT S1
model from these campaign arms. They are a COMPARISON, never a drop-in
replacement, and every output row carries `lineage`, `variant`, `stride` and
`scene_set` so the two can never be pooled by accident.

THREE SCENE SETS — pick the one matching the registered row you compare against
------------------------------------------------------------------------------
``--scene-set allscenes`` (DEFAULT)
    All 213 orbit-76 S1 scenes, no window restriction, epochs from the S1 date.
    This is what the script did before the scene-set flag existed; it is the
    default so every previously-written `freq_stats_<variant>_s<stride>.csv`
    stays reproducible. It corresponds to **no** registered row exactly.

``--scene-set samewin``
    The **186** orbit-76 scenes inside the S2 date span (2017-08-22 →
    2024-10-29), weight 1, epochs from the S1 date. This is the registry's
    **unmatched-calendars** comparison (registered gain IoU **0.37**).

``--scene-set matched``
    The nearest S1 scene to each S2 date -- one row per S2 date, so a scene
    matched twice carries weight 2 -- with epochs from the **S2** date. Removes
    the acquisition-calendar difference between sensors and is the registry's
    **PRIMARY** comparison (registered gain IoU **0.40**). Verified to
    reproduce the registry's sampling: **53 unique S1 scenes for 78 S2 dates,
    median |dt| 3.8 d**.

⚠️ These are three different estimands. Comparing an `allscenes` or `samewin`
number against the registered 0.40, or a `matched` number against 0.37, mixes
them. The scene set is in the output filename and in a `scene_set` column for
exactly this reason.

Usage:
    python run_ship_freqmaps.py --variant last --stride 32
    python run_ship_freqmaps.py --variant last --stride 32 --scene-set matched
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


def s1_scene_plan(scene_set: str, s2s):
    """Which S1 scenes contribute, with what multiplicity, per epoch part.

    Replicates the two scene sets of the registered
    ``11_s2_frequency_maps.py``. Returns ``{part: {scene_id: weight}}``.

    ``samewin`` -- every orbit-76 S1 scene inside the S2 date span, weight 1,
        epochs taken from the S1 date. This is a *same-window* reference: it
        answers "what does S1 see over the same period", and it is what the
        registry calls the **unmatched-calendars** comparison.

    ``matched`` -- the nearest S1 scene to EACH S2 date (one row per S2 date,
        so a scene matched by two S2 dates carries weight 2), with epochs taken
        from the **S2** date, not the S1 date. This is the registry's
        **calendar-matched** comparison and its PRIMARY gain-IoU row, because it
        removes the acquisition-calendar difference between the sensors.

    ⚠️ The two are different estimands and their numbers are NOT interchangeable
    -- in the registry, gain IoU is 0.40 matched vs 0.37 unmatched.
    """
    idx = shim.scene_index(shim.SEEDS[0])[["scene_id", "datetime"]]
    if scene_set == "allscenes":
        yr = idx.datetime.dt.year
        sel = {"full": np.ones(len(idx), bool),
               "early": yr.between(*EPOCH_EARLY).values,
               "late": yr.between(*EPOCH_LATE).values}
        frame = idx
    elif scene_set == "samewin":
        t0, t1 = s2s.datetime.min(), s2s.datetime.max()
        sw = idx[(idx.datetime >= t0) & (idx.datetime <= t1)].copy()
        yr = sw.datetime.dt.year
        sel = {"full": np.ones(len(sw), bool),
               "early": yr.between(*EPOCH_EARLY).values,
               "late": yr.between(*EPOCH_LATE).values}
        frame = sw
    elif scene_set == "matched":
        s1t = idx.datetime.values
        j = np.array([np.abs(s1t - np.datetime64(d)).argmin()
                      for d in s2s.datetime])
        dt_days = (np.abs(s1t[j] - s2s.datetime.values.astype("datetime64[ns]"))
                   / np.timedelta64(1, "D"))
        frame = idx.iloc[j].reset_index(drop=True)
        s2yr = s2s.datetime.dt.year.values
        sel = {"full": np.ones(len(frame), bool),
               "early": pd.Series(s2yr).between(*EPOCH_EARLY).values,
               "late": pd.Series(s2yr).between(*EPOCH_LATE).values}
        print("S1 matched: %d unique scenes for %d S2 dates; |dt| median %.1f d, "
              "max %.1f d" % (frame.scene_id.nunique(), len(frame),
                              float(np.median(dt_days)), float(dt_days.max())))
    else:
        raise ValueError("scene_set must be allscenes|samewin|matched, got %r"
                         % scene_set)

    plan, counts = {}, {}
    for part, m in sel.items():
        sub = frame.loc[m]
        if len(sub):
            # multiplicity: a scene matched by two S2 dates contributes twice
            plan[part] = sub.groupby("scene_id").size().to_dict()
            counts[part] = int(m.sum())
    return plan, counts


def s1_freq(shape, win, scene_set, s2s):
    """3-seed median of per-seed weighted P>THR frequency, per epoch part."""
    plan, counts = s1_scene_plan(scene_set, s2s)
    needed = sorted({sid for p in plan.values() for sid in p})
    per_seed = {k: [] for k in plan}
    for seed in shim.SEEDS:
        idx = shim.scene_index(seed).set_index("scene_id")
        acc = {k: np.zeros(shape, np.float64) for k in plan}
        for sid in needed:
            with rasterio.open(idx.loc[sid, "tif"]) as ds:
                w = ds.read(1, window=win) > THR
            for k, p in plan.items():
                if sid in p:
                    acc[k] += p[sid] * w
        for k in plan:
            denom = max(sum(plan[k].values()), 1)
            per_seed[k].append((acc[k] / denom).astype(np.float32))
    return ({k: np.median(np.stack(v), axis=0) for k, v in per_seed.items()},
            counts)


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
    ap.add_argument("--scene-set", default="allscenes",
                    choices=("allscenes", "samewin", "matched"),
                    help="S1 scene set. `allscenes` = all 213 orbit-76 scenes, "
                         "no window restriction -- what this script did before "
                         "the scene-set flag existed, kept as the DEFAULT so "
                         "prior outputs stay reproducible. `samewin` = the 186 "
                         "scenes inside the S2 date span (the registry's "
                         "UNMATCHED-calendars comparison, gain IoU 0.37). "
                         "`matched` = nearest S1 scene per S2 date, epochs from "
                         "the S2 date (the registry's PRIMARY calendar-matched "
                         "comparison, gain IoU 0.40). Three different estimands "
                         "-- never pool them. (default: %(default)s)")
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
    # The scene set is part of the filename: a `matched` run must NOT overwrite
    # the `samewin` table -- they are different estimands, not two takes on one.
    # `samewin` keeps the historical name so existing consumers are unaffected.
    tag = "%s_s%d%s" % (a.variant, a.stride,
                        "" if a.scene_set == "allscenes" else "_" + a.scene_set)
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
    f1, n1 = s1_freq(land.shape, win, a.scene_set, s2s)
    print("S1 scenes full/early/late: %s" % n1)

    rows = []
    def add(metric, value, pair="S2 vs S1"):
        rows.append(dict(lineage=lineage, variant=a.variant, stride=a.stride,
                         scene_set=a.scene_set,
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
