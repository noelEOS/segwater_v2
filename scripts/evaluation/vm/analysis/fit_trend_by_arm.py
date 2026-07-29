"""Demak trend under the CANONICAL fit spec, grouped by ARM (all s42).

Canonical spec (experiments/demak_semarang/scripts/03_trend_models.py):
    area ~ 1 + t + annual harmonics x2   (OLS, Newey-West/HAC SEs)
maxlags = floor(4*(n/100)^(2/9)). Single seed per arm, so the HAC SE is the
uncertainty -- there is NO seed-SD here. Plain OLS printed for reference only.
"""
from __future__ import annotations
import argparse
import numpy as np, pandas as pd, statsmodels.api as sm


def harm_hac(dt, y):
    d = pd.to_datetime(dt, utc=True)
    year = d.dt.year + (d.dt.dayofyear - 1 + d.dt.hour / 24) / 365.25
    doy = 2 * np.pi * d.dt.dayofyear / 365.25
    X = sm.add_constant(pd.DataFrame({
        "t_years": year - year.mean(),
        "sin1": np.sin(doy), "cos1": np.cos(doy),
        "sin2": np.sin(2 * doy), "cos2": np.cos(2 * doy),
    }).reset_index(drop=True))
    yy = y.reset_index(drop=True)
    lags = int(np.floor(4 * (len(yy) / 100) ** (2 / 9)))
    f = sm.OLS(yy, X).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    ci = f.conf_int().loc["t_years"]
    return (float(f.params["t_years"]), float(f.bse["t_years"]),
            float(ci.iloc[0]), float(ci.iloc[1]), lags)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("area_csv")
    ap.add_argument("--col", default="area_ha_thr0.5")
    ap.add_argument("--group", default="arm")
    a = ap.parse_args()
    df = pd.read_csv(a.area_csv, parse_dates=["datetime"])
    w = df[df.in_analysis_window] if "in_analysis_window" in df else df
    print("Demak trend, canonical spec (harmonics + Newey-West HAC), col=%s" % a.col)
    print("%-14s %10s %9s %22s %6s %6s" % (a.group, "ha/yr", "HAC se", "95% CI", "lags", "n"))
    for g, sub in w.groupby(a.group):
        s, se, lo, hi, lags = harm_hac(sub.datetime, sub[a.col])
        x = (pd.to_datetime(sub.datetime, utc=True).dt.year
             + (pd.to_datetime(sub.datetime, utc=True).dt.dayofyear - 1) / 365.25).values
        ols = float(np.polyfit(x, sub[a.col].values, 1)[0])
        print("%-14s %+10.1f %9.1f  [%+8.1f, %+8.1f] %6d %6d   (plain-OLS %+.1f)"
              % (g, s, se, lo, hi, lags, len(sub), ols))
    print()
    print("Single seed per arm -> HAC se is the uncertainty; no seed-SD available.")


if __name__ == "__main__":
    main()
