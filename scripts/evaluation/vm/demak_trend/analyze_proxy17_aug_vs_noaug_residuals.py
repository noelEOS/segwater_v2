#!/usr/bin/env python
"""S2-minus-S1 bias and residual drift for the four proxy17 augmentation arms.

Uses the same method as ../scripts/temporal_fidelity_diagnostics.py:
  - S2 vote+veto scenes with valid_frac_aoi >= 0.90
  - nearest S1/S2 pairing within +/-2 days
  - area at probability threshold 0.5
  - residual = S2 water area - S1 water area
  - regression time coordinate = S1 acquisition datetime
  - OLS residual ~ time + annual/semiannual harmonics
  - Newey-West HAC uncertainty with the project-standard lag rule

Writes:
  proxy17_aug_vs_noaug_s2_pairs.csv
  proxy17_aug_vs_noaug_s2_residual_drift.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

# Defaults reproduce the Mac layout (script sitting beside its inputs in
# experiments/.../proxy-trend/). Override with --s1/--s2/--out-dir to run from
# the tracked eval-kit copy or VM-side. Note this stage is pure CSV work -- no
# rasters -- so it does NOT have to run on the VM.
HERE = Path(__file__).resolve().parent
DEFAULT_S1 = HERE / "proxy17_aug_vs_noaug_water_area_timeseries.csv"
DEFAULT_S2 = (
    Path(__file__).resolve().parents[4]
    / "experiments/demak_semarang/optical_validation_comparisson/outputs"
    / "s2_voteveto_timeseries.csv"
)

VALID_GATE = 0.90
PAIR_WINDOW = pd.Timedelta("2D")
EPOCH_EARLY = (2017, 2019)
EPOCH_LATE = (2022, 2024)


def epoch_for_year(year: int) -> str:
    if EPOCH_EARLY[0] <= year <= EPOCH_EARLY[1]:
        return "early"
    if EPOCH_LATE[0] <= year <= EPOCH_LATE[1]:
        return "late"
    return "middle"


def harmonic_hac_fit(datetime: pd.Series, values: pd.Series) -> dict[str, float]:
    dt = pd.to_datetime(datetime, utc=True)
    year = dt.dt.year + (dt.dt.dayofyear - 1 + dt.dt.hour / 24) / 365.25
    doy = 2 * np.pi * dt.dt.dayofyear / 365.25
    X = sm.add_constant(
        pd.DataFrame(
            {
                "t_years": year - year.mean(),
                "sin1": np.sin(doy),
                "cos1": np.cos(doy),
                "sin2": np.sin(2 * doy),
                "cos2": np.cos(2 * doy),
            }
        ).reset_index(drop=True)
    )
    y = values.reset_index(drop=True)
    maxlags = int(np.floor(4 * (len(y) / 100) ** (2 / 9)))
    fit = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": maxlags})
    ci = fit.conf_int().loc["t_years"]
    return {
        "residual_drift_ha_per_yr": float(fit.params["t_years"]),
        "residual_drift_se": float(fit.bse["t_years"]),
        "residual_drift_p": float(fit.pvalues["t_years"]),
        "residual_drift_ci_low": float(ci.iloc[0]),
        "residual_drift_ci_high": float(ci.iloc[1]),
        "s1_excess_trend_vs_s2_ha_per_yr": float(-fit.params["t_years"]),
        "hac_maxlags": maxlags,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--s1", type=Path, default=DEFAULT_S1,
                    help="proxy17 water-area timeseries (long: augmentation, arm, seed)")
    ap.add_argument("--s2", type=Path, default=DEFAULT_S2,
                    help="S2 vote+veto timeseries")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="where to write the two CSVs (default: alongside --s1)")
    ap.add_argument("--expect-pairs", type=int, default=17,
                    help="assert this many S1/S2 pairs per arm (0 disables)")
    args = ap.parse_args()

    out_dir = args.out_dir or args.s1.resolve().parent
    out_dir.mkdir(parents=True, exist_ok=True)

    for p in (args.s1, args.s2):
        if not p.exists():
            raise SystemExit("missing input: %s" % p)

    s1 = pd.read_csv(args.s1, parse_dates=["datetime"])
    s2 = pd.read_csv(args.s2, parse_dates=["datetime"])
    s2 = (
        s2[s2["valid_frac_aoi"] >= VALID_GATE][
            ["datetime", "water_area_ha", "valid_frac_aoi"]
        ]
        .rename(
            columns={
                "datetime": "s2_datetime",
                "water_area_ha": "s2_area_ha",
            }
        )
        .sort_values("s2_datetime")
    )

    pair_frames = []
    summary_rows = []
    for (augmentation, arm, seed), group in s1.groupby(
        ["augmentation", "arm", "seed"], sort=True
    ):
        candidate = (
            group[["scene_id", "datetime", "area_ha_thr0.5", "ckpt_file"]]
            .rename(
                columns={
                    "datetime": "s1_datetime",
                    "area_ha_thr0.5": "s1_area_ha",
                }
            )
            .sort_values("s1_datetime")
        )
        pairs = pd.merge_asof(
            s2,
            candidate,
            left_on="s2_datetime",
            right_on="s1_datetime",
            direction="nearest",
            tolerance=PAIR_WINDOW,
        ).dropna(subset=["s1_area_ha"])
        if args.expect_pairs:
            assert len(pairs) == args.expect_pairs, (
                augmentation, arm, seed, len(pairs), args.expect_pairs)
        pairs["augmentation"] = augmentation
        pairs["arm"] = arm
        pairs["seed"] = seed
        pairs["residual_s2_minus_s1_ha"] = pairs["s2_area_ha"] - pairs["s1_area_ha"]
        pairs["pair_delta_days"] = (
            pairs["s2_datetime"] - pairs["s1_datetime"]
        ).dt.total_seconds() / 86400
        pairs["year"] = pairs["s2_datetime"].dt.year
        pairs["epoch"] = pairs["year"].map(epoch_for_year)
        pair_frames.append(pairs)

        early = pairs[pairs["epoch"] == "early"]["residual_s2_minus_s1_ha"]
        late = pairs[pairs["epoch"] == "late"]["residual_s2_minus_s1_ha"]
        summary_rows.append(
            {
                "augmentation": augmentation,
                "arm": arm,
                "seed": seed,
                "ckpt_file": pairs["ckpt_file"].iloc[0],
                "fit_time_basis": "s1_datetime",
                "n_pairs": len(pairs),
                "mean_bias_s2_minus_s1_ha": float(
                    pairs["residual_s2_minus_s1_ha"].mean()
                ),
                "early_bias_s2_minus_s1_ha": float(early.mean()),
                "late_bias_s2_minus_s1_ha": float(late.mean()),
                "late_minus_early_bias_ha": float(late.mean() - early.mean()),
                **harmonic_hac_fit(
                    pairs["s1_datetime"], pairs["residual_s2_minus_s1_ha"]
                ),
            }
        )

    pair_table = pd.concat(pair_frames, ignore_index=True).sort_values(
        ["augmentation", "arm", "s2_datetime"]
    )
    summary = pd.DataFrame(summary_rows).sort_values(["augmentation", "arm"])
    pair_table.to_csv(out_dir / "proxy17_aug_vs_noaug_s2_pairs.csv", index=False)
    summary.to_csv(
        out_dir / "proxy17_aug_vs_noaug_s2_residual_drift.csv", index=False
    )
    print("wrote %s/proxy17_aug_vs_noaug_s2_{pairs,residual_drift}.csv\n" % out_dir)

    cols = [
        "augmentation",
        "arm",
        "n_pairs",
        "mean_bias_s2_minus_s1_ha",
        "early_bias_s2_minus_s1_ha",
        "late_bias_s2_minus_s1_ha",
        "residual_drift_ha_per_yr",
        "residual_drift_se",
        "s1_excess_trend_vs_s2_ha_per_yr",
    ]
    print(summary[cols].round(2).to_string(index=False))


if __name__ == "__main__":
    main()
