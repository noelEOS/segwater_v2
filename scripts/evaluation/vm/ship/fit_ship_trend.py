"""Demak trend for the 6 SHIP arms under the CANONICAL fit spec, grouped by arm.

Canonical spec (experiments/demak_semarang/scripts/03_trend_models.py):
    area ~ 1 + t + annual harmonics x2   (OLS, Newey-West/HAC SEs)
    maxlags = max(1, floor(4*(n/100)^(2/9)))
    window <= 2024-12-31, orbit-127 excluded (already dropped upstream),
    column area_ha_thr0.5.

⚠️ HAC, never plain OLS. Plain OLS is printed for reference only; quoting it
against a registered HAC figure overstates deltas by ~20 ha/yr.

Unlike the earlier one-seed-per-arm rounds, this campaign has 3 seeds per
variant, so a genuine across-seed SD is available. It is reported alongside --
but NOT in place of -- the per-arm HAC SE. The two measure different things:
HAC SE is within-arm sampling uncertainty from one time series; the seed-SD is
across-seed training variability.

Usage:
    python scripts/evaluation/vm/ship/fit_ship_trend.py AREA_CSV --stride 112
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np, pandas as pd, statsmodels.api as sm

OUT_DIR = Path.home()/"workspace/results/ship_decision_2026-07/demak_trend"


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
    lags = max(1, int(np.floor(4 * (len(yy) / 100) ** (2 / 9))))
    f = sm.OLS(yy, X).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    ci = f.conf_int().loc["t_years"]
    return (float(f.params["t_years"]), float(f.bse["t_years"]),
            float(ci.iloc[0]), float(ci.iloc[1]), lags)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("area_csv")
    ap.add_argument("--stride", type=int, required=True, choices=(32, 112))
    ap.add_argument("--col", default="area_ha_thr0.5")
    ap.add_argument("--expect-n", type=int, default=206)
    a = ap.parse_args()

    df = pd.read_csv(a.area_csv, parse_dates=["datetime"])
    w = df[df.in_analysis_window]
    print("Demak trend, canonical spec (harmonics + Newey-West HAC), col=%s, stride=%d"
          % (a.col, a.stride))
    print("%-10s %10s %9s %22s %6s %6s" % ("arm", "ha/yr", "HAC se", "95% CI", "lags", "n"))
    out = []
    for arm, sub in w.groupby("arm"):
        sub = sub.sort_values("datetime")
        assert len(sub) == a.expect_n, "%s: n=%d != %d" % (arm, len(sub), a.expect_n)
        s, se, lo, hi, lags = harm_hac(sub.datetime, sub[a.col])
        x = (pd.to_datetime(sub.datetime, utc=True).dt.year
             + (pd.to_datetime(sub.datetime, utc=True).dt.dayofyear - 1) / 365.25).values
        ols = float(np.polyfit(x, sub[a.col].values, 1)[0])
        print("%-10s %+10.1f %9.1f  [%+8.1f, %+8.1f] %6d %6d   (plain-OLS %+.1f, ref only)"
              % (arm, s, se, lo, hi, lags, len(sub), ols))
        out.append(dict(seed=sub.seed.iloc[0], variant=sub.variant.iloc[0], arm=arm,
                        stride=a.stride, slope_ha_yr=s, hac_se=se,
                        ci_lo=lo, ci_hi=hi, maxlags=lags, n=len(sub)))
    t = pd.DataFrame(out).sort_values(["variant", "seed"])
    # Carry the input's arm-set tag through to the output. A subset run
    # (e.g. areas_swa5_s112.csv) must NOT overwrite the full-campaign
    # trend table -- it did once, and only the Mac mirror saved it.
    stem = Path(a.area_csv).stem            # demak_full_ship_areas[_tag]_s112
    tag = ""
    if stem.startswith("demak_full_ship_areas") and stem.endswith("_s%d" % a.stride):
        mid = stem[len("demak_full_ship_areas"):-len("_s%d" % a.stride)]
        tag = mid  # "" for the default set, "_swa5" etc. otherwise
    p = OUT_DIR/("demak_full_ship_trend%s_s%d.csv" % (tag, a.stride))
    p.parent.mkdir(parents=True, exist_ok=True)
    t.to_csv(p, index=False)
    print("\nwrote %s (%d rows)" % (p, len(t)))

    print("\n--- per-variant 3-seed mean +/- SD (ddof=1) ---   [NOT the HAC se]")
    for v, g in t.groupby("variant"):
        assert len(g) == 3, "%s: %d seeds != 3" % (v, len(g))
        print("  %-4s  %+.1f +/- %.1f  (seeds: %s)"
              % (v, g.slope_ha_yr.mean(), g.slope_ha_yr.std(ddof=1),
                 ", ".join("%s %+.1f" % (r.seed, r.slope_ha_yr) for r in g.itertuples())))


if __name__ == "__main__":
    main()
