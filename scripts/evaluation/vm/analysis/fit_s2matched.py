"""S2-date-matched Demak trend for the five mx630 arms.

Replicates scripts/trend_variants_csvonly.py exactly:
  - design: const + t_years + annual + semiannual harmonics, hour folded into
    the fractional year (use_hour=True);
  - Newey-West HAC, maxlags = max(1, floor(4*(n/100)**(2/9)));
  - window <= 2024-12-31 23:59:59 UTC, drop orbit-127 scene;
  - match: nearest S1 scene within +/-6 d of each gated optical date (valid_frac
    >= 0.90), deduplicated by positional index.
Single seed per arm, so no 3-seed median step (the registered series took a
median over 3 seeds; these arms have one seed each).
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm

W0 = pd.Timestamp("2017-03-16", tz="UTC")
W1 = pd.Timestamp("2024-12-31 23:59:59", tz="UTC")
ORBIT127 = "S1_20250730_105715_127_1_1"
TOL = pd.Timedelta("6D")
GATE = 0.90
S1_CSV = "/home/noel/demak_full_5arm_water_area_timeseries.csv"
S2_CSV = "/home/noel/proxy17_analysis/s2_voteveto_timeseries.csv"
ARMS = ["mx630s2_best", "mx630s2_last", "mx630s2_swa5", "mx630k", "mx630k_best"]


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
    return (n, f.params["t_years"], f.bse["t_years"],
            ci.loc["t_years", 0], ci.loc["t_years", 1], lags)


o = pd.read_csv(S2_CSV, parse_dates=["datetime"])
o = o[(o.datetime >= W0) & (o.datetime <= W1)]
o = o[o.valid_frac_aoi >= GATE]
opt = o.sort_values("datetime")["datetime"].reset_index(drop=True)
assert len(opt) == 78, "S2 gated dates %d != 78" % len(opt)

df = pd.read_csv(S1_CSV, parse_dates=["datetime"])
print("Demak trend, S2-date-matched (canonical harmonics+HAC, thr 0.5)")
print("%-14s %10s %9s %22s %6s %5s" % ("arm", "ha/yr", "HAC se", "95% CI", "lags", "n"))

matched_sets = {}
for arm in ARMS:
    s = df[(df.arm == arm) & (df.scene_id != ORBIT127)]
    s = s[s.datetime <= W1].sort_values("datetime").reset_index(drop=True)
    assert len(s) == 206, "%s: in-window %d != 206" % (arm, len(s))
    sd = s["datetime"].values
    idx = sorted({int(np.abs(sd - t).argmin()) for t in opt.values
                  if np.abs(sd - t).min() <= TOL})
    assert len(idx) == 48, "%s: matched %d != 48" % (arm, len(idx))
    matched_sets[arm] = set(s.iloc[idx].scene_id)
    n, sl, se, lo, hi, lags = fit(s.iloc[idx].reset_index(drop=True), "area_ha_thr0.5")
    print("%-14s %+10.1f %9.1f  [%+8.1f, %+8.1f] %6d %5d" % (arm, sl, se, lo, hi, lags, n))

ref = matched_sets[ARMS[0]]
for a in ARMS[1:]:
    assert matched_sets[a] == ref, "%s: matched scene set differs" % a
print("\nmatched scene set identical across all %d arms (n=48)" % len(ARMS))
