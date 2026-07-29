"""Demak water-extent trend under the CANONICAL fit spec, for any area CSV.

Canonical spec (experiments/demak_semarang/scripts/03_trend_models.py):
    area ~ 1 + t + annual harmonics x2   (OLS, Newey-West/HAC SEs)
with maxlags = floor(4 * (n/100)^(2/9)). Reported as the 3-seed mean of the
per-seed slopes +/- seed-SD (ddof=1), on the 206-scene analysis window.

A plain OLS slope is ALSO printed for reference, because an earlier comparison
in this project mixed the two specs and overstated a delta by ~20 ha/yr.

    python fit_demak_trend.py <area_csv> [--label NAME]
"""
from __future__ import annotations
import argparse, statistics as st
import numpy as np, pandas as pd, statsmodels.api as sm


def harmonic_hac_slope(dt: pd.Series, y: pd.Series):
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
    fit = sm.OLS(yy, X).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    return float(fit.params["t_years"]), float(fit.bse["t_years"]), lags


def plain_ols_slope(dt: pd.Series, y: pd.Series) -> float:
    d = pd.to_datetime(dt, utc=True)
    x = (d.dt.year + (d.dt.dayofyear - 1 + d.dt.hour / 24) / 365.25).values
    return float(np.polyfit(x, y.values, 1)[0])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("area_csv")
    ap.add_argument("--label", default="")
    ap.add_argument("--col", default="area_ha_thr0.5")
    args = ap.parse_args()

    df = pd.read_csv(args.area_csv, parse_dates=["datetime"])
    if "in_analysis_window" in df.columns:
        w = df[df.in_analysis_window]
    else:
        w = df[df.datetime <= pd.Timestamp("2024-12-31", tz="UTC")]
    print("%s  (%s)" % (args.label or args.area_csv, args.col))
    print("  window: %d scenes/seed" % (len(w) // w.seed.nunique()))
    hac, ols = [], []
    for seed, g in w.groupby("seed"):
        s, se, lags = harmonic_hac_slope(g.datetime, g[args.col])
        o = plain_ols_slope(g.datetime, g[args.col])
        hac.append(s); ols.append(o)
        print("  %-5s canonical(harm+HAC) %+8.1f (HAC se %.1f, lags %d)   plain-OLS %+8.1f"
              % (seed, s, se, lags, o))
    print("  --")
    print("  CANONICAL 3-seed: %+.1f +/- %.1f ha/yr" % (st.mean(hac), st.stdev(hac)))
    print("  plain-OLS 3-seed: %+.1f +/- %.1f ha/yr  (NOT the registered spec)"
          % (st.mean(ols), st.stdev(ols)))


if __name__ == "__main__":
    main()
