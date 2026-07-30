"""S2-date-matched Demak trend for the 6 SHIP arms, both strides -> 12 rows.

Replicates fit_s2matched.py / trend_variants_csvonly.py exactly:
  - design: const + t_years + annual + semiannual harmonics, hour folded into
    the fractional year;
  - Newey-West HAC, maxlags = max(1, floor(4*(n/100)**(2/9)));
  - window <= 2024-12-31 23:59:59 UTC, orbit-127 scene dropped;
  - match: nearest S1 scene within +/-6 d of each gated optical date
    (valid_frac_aoi >= 0.90), deduplicated by positional index -> n=48.

⚠️ HAC, never plain OLS.

Usage:
    python scripts/evaluation/vm/ship/fit_ship_s2matched.py \
        --area-csv S112_CSV --stride 112 [--area-csv S32_CSV --stride 32] ...
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np, pandas as pd, statsmodels.api as sm

W0 = pd.Timestamp("2017-03-16", tz="UTC")
W1 = pd.Timestamp("2024-12-31 23:59:59", tz="UTC")
ORBIT127 = "S1_20250730_105715_127_1_1"
TOL = pd.Timedelta("6D")
GATE = 0.90
S2_CSV = Path.home()/"proxy17_analysis/s2_voteveto_timeseries.csv"
OUT_DIR = Path.home()/"workspace/results/ship_decision_2026-07/demak_trend"
SEEDS = ["s19", "s42", "s58"]
VARIANTS = ["best", "last"]
ARMS = ["%s_%s" % (s, v) for s in SEEDS for v in VARIANTS]


def design(d):
    t = d["datetime"]
    frac = (t.dt.dayofyear - 1 + t.dt.hour / 24) / 365.25
    year = t.dt.year + frac
    doy = 2 * np.pi * t.dt.dayofyear / 365.25
    X = pd.DataFrame({"t_years": year - year.mean(),
                      "sin1": np.sin(doy), "cos1": np.cos(doy),
                      "sin2": np.sin(2 * doy), "cos2": np.cos(2 * doy)},
                     index=d.index)
    return sm.add_constant(X)


def fit(d, ycol):
    X = design(d)
    n = len(d)
    lags = max(1, int(np.floor(4 * (n / 100) ** (2 / 9))))
    f = sm.OLS(d[ycol], X).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    ci = f.conf_int()
    return (n, float(f.params["t_years"]), float(f.bse["t_years"]),
            float(ci.loc["t_years", 0]), float(ci.loc["t_years", 1]), lags)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--area-csv", action="append", required=True, dest="csvs")
    ap.add_argument("--stride", action="append", required=True, type=int, dest="strides")
    ap.add_argument("--col", default="area_ha_thr0.5")
    ap.add_argument("--variants", nargs="+", default=VARIANTS,
                    choices=["best", "last", "swa5"],
                    help="arm variants present in the area CSVs "
                         "(default: %(default)s)")
    ap.add_argument("--out", type=Path, default=None,
                    help="output CSV (default: %s/demak_full_ship_trend_s2matched.csv). "
                         "Pass an explicit path for a non-`ship` campaign — the "
                         "default name would overwrite the Swin-B table."
                         % OUT_DIR)
    a = ap.parse_args()
    assert len(a.csvs) == len(a.strides), "--area-csv / --stride must pair up"
    arms = ["%s_%s" % (s, v) for s in SEEDS for v in a.variants]

    o = pd.read_csv(S2_CSV, parse_dates=["datetime"])
    o = o[(o.datetime >= W0) & (o.datetime <= W1)]
    o = o[o.valid_frac_aoi >= GATE]
    opt = o.sort_values("datetime")["datetime"].reset_index(drop=True)
    assert len(opt) == 78, "S2 gated dates %d != 78" % len(opt)

    print("Demak trend, S2-date-matched (canonical harmonics+HAC, %s)" % a.col)
    print("%-8s %-10s %10s %9s %22s %6s %5s"
          % ("stride", "arm", "ha/yr", "HAC se", "95% CI", "lags", "n"))
    out, matched_sets = [], {}
    for csv, stride in zip(a.csvs, a.strides):
        df = pd.read_csv(csv, parse_dates=["datetime"])
        for arm in arms:
            s = df[(df.arm == arm) & (df.scene_id != ORBIT127)]
            s = s[s.datetime <= W1].sort_values("datetime").reset_index(drop=True)
            assert len(s) == 206, "%s s%d: in-window %d != 206" % (arm, stride, len(s))
            sd = s["datetime"].values
            idx = sorted({int(np.abs(sd - t).argmin()) for t in opt.values
                          if np.abs(sd - t).min() <= TOL})
            assert len(idx) == 48, "%s s%d: matched %d != 48" % (arm, stride, len(idx))
            matched_sets[(stride, arm)] = set(s.iloc[idx].scene_id)
            n, sl, se, lo, hi, lags = fit(s.iloc[idx].reset_index(drop=True), a.col)
            print("%-8d %-10s %+10.1f %9.1f  [%+8.1f, %+8.1f] %6d %5d"
                  % (stride, arm, sl, se, lo, hi, lags, n))
            out.append(dict(seed=s.seed.iloc[0], variant=s.variant.iloc[0], arm=arm,
                            stride=stride, slope_ha_yr=sl, hac_se=se, ci_lo=lo,
                            ci_hi=hi, maxlags=lags, n=n, estimand="s2matched"))

    ref = matched_sets[(a.strides[0], arms[0])]
    for k, v in matched_sets.items():
        assert v == ref, "%s: matched scene set differs" % (k,)
    print("\nmatched scene set identical across all %d arm x stride cells (n=48)"
          % len(matched_sets))

    t = pd.DataFrame(out).sort_values(["stride", "variant", "seed"])
    p = a.out or OUT_DIR/"demak_full_ship_trend_s2matched.csv"
    p.parent.mkdir(parents=True, exist_ok=True)
    t.to_csv(p, index=False)
    print("wrote %s (%d rows)" % (p, len(t)))

    print("\n--- per-variant x stride 3-seed mean +/- SD (ddof=1) ---  [NOT the HAC se]")
    for (st, v), g in t.groupby(["stride", "variant"]):
        assert len(g) == 3, "s%d %s: %d seeds != 3" % (st, v, len(g))
        print("  s%-4d %-4s  %+.1f +/- %.1f"
              % (st, v, g.slope_ha_yr.mean(), g.slope_ha_yr.std(ddof=1)))


if __name__ == "__main__":
    main()
