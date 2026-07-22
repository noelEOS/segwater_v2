"""Peer-review-proof statistical assessment: ConvNeXtV2 vs Swin-B at Hampyeong Bay.

The accuracy comparison (hampyeong_model_comparison.py) reports point estimates only.
This adds the inferential layer, built to survive a hostile methods reviewer. Its
governing constraint is that the site has only THREE dates: if date is the unit of
replication, the smallest achievable exact p is 1/8 = 0.125. No amount of machinery
changes that, so this script does not pretend otherwise.

Estimand (stated explicitly, per the writeup):
  E1  descriptive, these 3 fixed tidal acquisitions at this site  -- primary
  E2  this site across the tidal cycle (n=3, weak)                -- reported, not oversold
  E3  coastal SAR in general                                      -- NOT claimed

CLAIM 1 (direction): exact sign test on the 3 DATE-MEANS -> p = 0.125, plus the
  equivalent date-level sign-flip randomization. The 9/9 date x seed agreement and
  3/3 seed agreement are reported as DESCRIPTIVE corroboration only -- the 9 pairs are
  spatially/seed correlated (cross-model error r~=0.9), so 1/2^9 is not a valid p and
  is deliberately not computed as one.

CLAIM 2 (magnitude): mean paired diff with the n=3 DATE-as-unit 95% t-CI, reported as
  NOT excluding zero. The cluster-bootstrap-over-dates CI is computed only to document,
  in one clearly-flagged row, that it is anticonservative at n=3 clusters; it is not a
  headline result.

Spatial robustness lives in a separate script (hampyeong_block_bootstrap.py).

Run with an interpreter that has scipy + rasterio, e.g.
    /opt/homebrew/Caskroom/miniforge/base/envs/eda/bin/python
"""

from __future__ import annotations

import argparse
import sys
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

_SCRIPTS = Path(__file__).resolve().parent.parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

DEFAULT_METRICS_CSV = "experiments/hampyeong/evaluation/per_date_metrics.csv"
DEFAULT_OUT_DIR = "experiments/hampyeong/evaluation"

ARCH_A = "ConvNeXtV2"   # the direction is defined as A - B (>0 favors A)
ARCH_B = "Swin-B"
SEEDS = (19, 42, 58)
METRIC = "iou"
CLUSTER_BOOT_R = 20000
CLUSTER_BOOT_SEED = 42


def paired_diffs(df: pd.DataFrame, metric: str = METRIC) -> pd.DataFrame:
    """9 date x seed paired differences A - B for the given metric."""
    new = df[df.arch.isin([ARCH_A, ARCH_B])]
    wide = new.pivot_table(index=["date", "seed"], columns="arch", values=metric)
    wide["diff"] = wide[ARCH_A] - wide[ARCH_B]
    return wide.reset_index()


def sign_test_one_sided_greater(values: np.ndarray) -> tuple[int, int, float]:
    """Exact one-sided sign test that the median difference is > 0. Ties dropped."""
    v = values[values != 0.0]
    n_pos = int((v > 0).sum())
    n = int(v.size)
    p = float(stats.binomtest(n_pos, n, 0.5, alternative="greater").pvalue)
    return n_pos, n, p


def sign_flip_randomization(date_means: np.ndarray) -> tuple[float, int]:
    """Exact date-level sign-flip randomization test on the date-mean diffs.

    Null: each date's diff is equally likely + or -. Enumerates all 2^n_dates sign
    assignments; p = fraction with mean >= observed. At n=3 the floor is 1/8 = 0.125.
    """
    observed = float(np.mean(date_means))
    n = date_means.size
    perms = [float(np.mean(date_means * np.array(signs)))
             for signs in product((1.0, -1.0), repeat=n)]
    p = float(np.mean(np.array(perms) >= observed))
    return p, 2 ** n


def t_ci_mean(values: np.ndarray, conf: float = 0.95) -> tuple[float, float, float]:
    """Two-sided t confidence interval for the mean (small n)."""
    n = values.size
    m = float(np.mean(values))
    se = float(np.std(values, ddof=1) / np.sqrt(n))
    tcrit = float(stats.t.ppf(0.5 + conf / 2, n - 1))
    return m, m - tcrit * se, m + tcrit * se


def cluster_bootstrap_over_dates(new: pd.DataFrame, metric: str, rng: np.random.Generator,
                                 n_boot: int = CLUSTER_BOOT_R) -> tuple[float, float, float, float]:
    """Resample the 3 dates with replacement; each draw uses that date's seed-mean diff.

    Returns (boot_mean, ci_lo, ci_hi, frac_le_zero). Flagged in output as
    anticonservative at n=3 clusters -- documented, not a headline result.
    """
    per_date = (new[new.arch == ARCH_A].groupby("date")[metric].mean()
                - new[new.arch == ARCH_B].groupby("date")[metric].mean())
    dates = per_date.index.to_numpy()
    vals = per_date.to_numpy()
    boots = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, dates.size, size=dates.size)
        boots[i] = float(np.mean(vals[idx]))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(np.mean(boots)), float(lo), float(hi), float((boots <= 0).mean())


def build_rows(df: pd.DataFrame) -> pd.DataFrame:
    new = df[df.arch.isin([ARCH_A, ARCH_B])]
    pairs = paired_diffs(df)
    diffs9 = pairs["diff"].to_numpy()

    date_means = (new[new.arch == ARCH_A].groupby("date")[METRIC].mean()
                  - new[new.arch == ARCH_B].groupby("date")[METRIC].mean())
    seed_means = (new[new.arch == ARCH_A].groupby("seed")[METRIC].mean()
                  - new[new.arch == ARCH_B].groupby("seed")[METRIC].mean())

    rows: list[dict] = []

    # --- CLAIM 1: direction ---
    npos_d, n_d, p_date_sign = sign_test_one_sided_greater(date_means.to_numpy())
    p_perm, n_perm = sign_flip_randomization(date_means.to_numpy())
    npos_s, n_s, _ = sign_test_one_sided_greater(seed_means.to_numpy())
    npos9 = int((diffs9 > 0).sum())

    rows.append({"claim": "1_direction", "test": "sign_test_date_means", "unit": "date",
                 "statistic": f"{npos_d}/{n_d} positive", "p_value": p_date_sign,
                 "interpretation": "PRIMARY direction test; n=3 exact floor is 0.125",
                 "estimand": "E2", "is_headline": True})
    rows.append({"claim": "1_direction", "test": "sign_flip_randomization_date_means", "unit": "date",
                 "statistic": f"observed mean {date_means.mean():.4f}; {n_perm} sign assignments",
                 "p_value": p_perm, "interpretation": "equivalent exact test; min possible p = 1/8",
                 "estimand": "E2", "is_headline": True})
    rows.append({"claim": "1_direction", "test": "descriptive_9of9_pairs", "unit": "date x seed",
                 "statistic": f"{npos9}/9 pairs favor {ARCH_A}", "p_value": np.nan,
                 "interpretation": "DESCRIPTIVE ONLY: 9 pairs correlated (cross-model r~0.9); no valid 1/2^9 p",
                 "estimand": "E1", "is_headline": True})
    rows.append({"claim": "1_direction", "test": "descriptive_3of3_seeds", "unit": "seed",
                 "statistic": f"{npos_s}/{n_s} seed-means favor {ARCH_A}", "p_value": np.nan,
                 "interpretation": "DESCRIPTIVE corroboration across seeds", "estimand": "E1",
                 "is_headline": False})

    # --- CLAIM 2: magnitude ---
    m_d, lo_d, hi_d = t_ci_mean(date_means.to_numpy())
    rows.append({"claim": "2_magnitude", "test": "t_ci_date_as_unit", "unit": "date",
                 "statistic": f"mean diff {m_d:.4f} IoU", "p_value": np.nan,
                 "ci_lo": lo_d, "ci_hi": hi_d, "excludes_zero": bool(lo_d > 0 or hi_d < 0),
                 "interpretation": "HEADLINE magnitude interval; does NOT exclude zero at n=3",
                 "estimand": "E2", "is_headline": True})
    m_s, lo_s, hi_s = t_ci_mean(seed_means.to_numpy())
    rows.append({"claim": "2_magnitude", "test": "t_ci_seed_as_unit", "unit": "seed",
                 "statistic": f"mean diff {m_s:.4f} IoU", "p_value": np.nan,
                 "ci_lo": lo_s, "ci_hi": hi_s, "excludes_zero": bool(lo_s > 0 or hi_s < 0),
                 "interpretation": "secondary; seed-as-unit t-CI", "estimand": "E1",
                 "is_headline": False})
    rng = np.random.default_rng(CLUSTER_BOOT_SEED)
    bm, blo, bhi, frac = cluster_bootstrap_over_dates(new, METRIC, rng)
    rows.append({"claim": "2_magnitude", "test": "cluster_bootstrap_over_dates", "unit": "date",
                 "statistic": f"boot mean {bm:.4f}; frac<=0 = {frac:.4f}", "p_value": np.nan,
                 "ci_lo": blo, "ci_hi": bhi, "excludes_zero": bool(blo > 0 or bhi < 0),
                 "interpretation": "FLAGGED anticonservative at n=3 clusters; NOT evidence; documented only",
                 "estimand": "E2", "is_headline": False})

    # per-date and per-seed point diffs, descriptive
    for date, val in date_means.items():
        rows.append({"claim": "0_descriptive", "test": "per_date_diff", "unit": "date",
                     "statistic": f"{date}: {val:+.4f} IoU", "p_value": np.nan,
                     "interpretation": "seed-mean diff for this date", "estimand": "E1",
                     "is_headline": False})
    for seed, val in seed_means.items():
        rows.append({"claim": "0_descriptive", "test": "per_seed_diff", "unit": "seed",
                     "statistic": f"s{int(seed)}: {val:+.4f} IoU", "p_value": np.nan,
                     "interpretation": "date-mean diff for this seed", "estimand": "E1",
                     "is_headline": False})

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--metrics-csv", type=Path, default=Path(DEFAULT_METRICS_CSV))
    parser.add_argument("--out-dir", type=Path, default=Path(DEFAULT_OUT_DIR))
    args = parser.parse_args()

    df = pd.read_csv(args.metrics_csv)
    rows = build_rows(df)

    # --- assertions: the honest numbers must hold ---
    def _get(test):
        return rows[rows.test == test].iloc[0]
    assert abs(_get("sign_test_date_means").p_value - 0.125) < 1e-9, "date-mean sign p must be 0.125"
    assert abs(_get("sign_flip_randomization_date_means").p_value - 0.125) < 1e-9, "sign-flip p must be 0.125"
    tci = _get("t_ci_date_as_unit")
    assert not tci.excludes_zero, "date-unit t-CI must NOT exclude zero (honest result)"
    assert abs(tci.ci_lo - (-0.0257)) < 2e-3 and abs(tci.ci_hi - 0.0581) < 2e-3, "t-CI must reproduce [-0.026,+0.058]"
    assert "9/9" in _get("descriptive_9of9_pairs").statistic, "9/9 descriptive fact"

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / "statistical_assessment.csv"
    rows.to_csv(out, index=False)

    print("Peer-review-proof statistical assessment (ConvNeXtV2 - Swin-B, IoU)\n")
    print("CLAIM 1 (direction):")
    print(f"  sign test on 3 date-means : p = {_get('sign_test_date_means').p_value:.3f}  "
          f"({_get('sign_test_date_means').statistic})")
    print(f"  sign-flip randomization   : p = {_get('sign_flip_randomization_date_means').p_value:.3f}  "
          f"(floor 1/8 = 0.125)")
    print(f"  descriptive               : {_get('descriptive_9of9_pairs').statistic}, "
          f"{_get('descriptive_3of3_seeds').statistic}")
    print("\nCLAIM 2 (magnitude):")
    print(f"  mean paired diff          : {tci.statistic}")
    print(f"  95% t-CI (date as unit)   : [{tci.ci_lo:+.4f}, {tci.ci_hi:+.4f}]  "
          f"-> {'excludes' if tci.excludes_zero else 'does NOT exclude'} zero")
    cb = _get("cluster_bootstrap_over_dates")
    print(f"  cluster boot (FLAGGED)    : [{cb.ci_lo:+.4f}, {cb.ci_hi:+.4f}]  "
          f"anticonservative at n=3; documented, not evidence")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
