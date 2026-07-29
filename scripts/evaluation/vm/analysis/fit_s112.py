"""Demak trend for the s112 bf16+device-stitch run: full window AND S2-matched.

Canonical spec throughout (scripts/trend_variants_csvonly.py):
  design  = const + t_years + annual + semiannual harmonics, hour folded into
            the fractional year;
  cov     = Newey-West HAC, maxlags = max(1, floor(4*(n/100)**(2/9)));
  window  = datetime <= 2024-12-31 23:59:59 UTC, orbit-127 scene dropped;
  match   = nearest S1 scene within +/-6 d of each gated optical date
            (valid_frac_aoi >= 0.90), deduplicated by positional index.

Single seed (s42), so HAC se is the uncertainty -- no seed-SD.
"""
import sys

import numpy as np
import pandas as pd
import statsmodels.api as sm

W0 = pd.Timestamp("2017-03-16", tz="UTC")
W1 = pd.Timestamp("2024-12-31 23:59:59", tz="UTC")
ORBIT127 = "S1_20250730_105715_127_1_1"
TOL = pd.Timedelta("6D")
GATE = 0.90
S2_CSV = "/home/noel/proxy17_analysis/s2_voteveto_timeseries.csv"
COL = "area_ha_thr0.5"


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


def fit(d):
    n = len(d)
    lags = max(1, int(np.floor(4 * (n / 100) ** (2 / 9))))
    f = sm.OLS(d[COL], design(d)).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    ci = f.conf_int()
    return n, f.params["t_years"], f.bse["t_years"], ci.loc["t_years", 0], ci.loc["t_years", 1], lags


def main():
    s1 = pd.read_csv(sys.argv[1], parse_dates=["datetime"])
    if "arm" in s1.columns and s1["arm"].nunique() > 1:
        raise SystemExit("expected a single-arm CSV, got %s" % sorted(s1["arm"].unique()))

    s1 = s1[s1["scene_id"] != ORBIT127]
    s1 = s1[s1["datetime"] <= W1].sort_values("datetime").reset_index(drop=True)
    assert len(s1) == 206, "in-window n=%d != 206" % len(s1)

    o = pd.read_csv(S2_CSV, parse_dates=["datetime"])
    o = o[(o.datetime >= W0) & (o.datetime <= W1)]
    o = o[o.valid_frac_aoi >= GATE]
    opt = o.sort_values("datetime")["datetime"].reset_index(drop=True)
    assert len(opt) == 78, "S2 gated dates %d != 78" % len(opt)

    sd = s1["datetime"].values
    idx = sorted({int(np.abs(sd - t).argmin()) for t in opt.values
                  if np.abs(sd - t).min() <= TOL})
    assert len(idx) == 48, "matched %d != 48" % len(idx)

    print("Demak trend -- source: %s" % sys.argv[1])
    print("canonical spec (harmonics + Newey-West HAC), col=%s\n" % COL)
    print("%-14s %10s %9s %22s %6s %5s" % ("estimand", "ha/yr", "HAC se", "95% CI", "lags", "n"))
    for label, d in [("full window", s1), ("S2-matched", s1.iloc[idx].reset_index(drop=True))]:
        n, sl, se, lo, hi, lags = fit(d)
        print("%-14s %+10.1f %9.1f  [%+8.1f, %+8.1f] %6d %5d" % (label, sl, se, lo, hi, lags, n))


if __name__ == "__main__":
    main()
