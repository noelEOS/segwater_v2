"""New segwater architectures vs the legacy ResNet50-UNet, at Hampyeong Bay.

Apples-to-apples: each of the 7 new architectures (3 training seeds each) against the
two legacy ResNet50-UNet runs from the deprecated study -- the site-finetuned model and
the pretrained baseline (each a single run, no seeds).

Same governing constraint as the ConvNeXtV2-vs-Swin-B assessment: the site has only 3
dates, so DATE is the unit of replication and the exact p-value floor is 1/8 = 0.125.
The legacy models have no seed, so each new architecture is paired against a legacy run
at the DATE level, using the architecture's 3-seed-MEAN IoU per date -> 3 paired diffs.
Seed spread is reported descriptively, never folded into the test.

Per (new architecture, legacy reference):
  - direction: exact sign test on the 3 date-mean paired diffs (new - legacy), p in
    {0.125, 0.5, 1.0} at n=3; plus the descriptive count of dates won.
  - magnitude: mean paired diff + date-as-unit 95% t-CI, flagged excludes-zero or not.

Reads the verified per_date_metrics.csv (produced by hampyeong_model_comparison.py),
so no raster IO here -- the point estimates are already checkpoint-audited and
provenance-verified upstream.

Run: /opt/homebrew/Caskroom/miniforge/base/envs/eda/bin/python <this file>
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

# Display order: strongest legacy baseline first.
LEGACY = [("ResNet50-finetuned", "resnet50_finetuned"),
          ("ResNet50-baseline", "resnet50_baseline")]
NEW_ARCHS = ["Swin-B", "ConvNeXtV2", "DeepLabV3+", "DPT-ViT-B", "SegFormer-B4",
             "UNet-R50", "UNet++-R50"]
METRIC = "iou"
DATES_ORDER = [20210305, 20210422, 20210621]   # int, matching per_date_metrics.csv dtype


def sign_test_and_ci(date_diffs: np.ndarray) -> dict:
    """Direction (exact sign test) + magnitude (date-as-unit t-CI) for 3 paired diffs."""
    n = date_diffs.size
    n_pos = int((date_diffs > 0).sum())
    n_nonzero = int((date_diffs != 0).sum())
    # two-sided exact sign test, then one-sided p for the observed direction
    p_two = float(stats.binomtest(n_pos, n_nonzero, 0.5).pvalue) if n_nonzero else 1.0
    favored = "new" if date_diffs.mean() > 0 else "legacy"
    p_one = float(stats.binomtest(max(n_pos, n_nonzero - n_pos), n_nonzero, 0.5,
                                  alternative="greater").pvalue) if n_nonzero else 1.0
    m = float(np.mean(date_diffs))
    se = float(np.std(date_diffs, ddof=1) / np.sqrt(n))
    tcrit = float(stats.t.ppf(0.975, n - 1))
    lo, hi = m - tcrit * se, m + tcrit * se
    return {
        "dates_won_by_new": n_pos, "n_dates": n,
        "sign_p_two_sided": p_two, "sign_p_one_sided": p_one, "favored": favored,
        "mean_diff": m, "t_ci_lo": lo, "t_ci_hi": hi,
        "excludes_zero": bool(lo > 0 or hi < 0),
    }


def build(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    # per-arch 3-seed-mean IoU per date
    new = df[df.arch.isin(NEW_ARCHS)]
    arch_date_mean = new.groupby(["arch", "date"])[METRIC].mean().unstack("date")
    arch_date_sd = new.groupby(["arch", "date"])[METRIC].std(ddof=1).unstack("date")
    legacy_date = {ref: df[df.model == ref].set_index("date")[METRIC] for _, ref in LEGACY}

    contrast_rows: list[dict] = []
    summary_rows: list[dict] = []

    for arch in NEW_ARCHS:
        adm = arch_date_mean.loc[arch]           # 3-seed mean per date
        pooled = float(adm.mean())
        seed_sd_max = float(arch_date_sd.loc[arch].max())
        row = {"architecture": arch,
               **{f"iou_{d}": float(adm[d]) for d in DATES_ORDER},
               "iou_pooled": pooled, "seed_sd_max": seed_sd_max}
        for ref_label, ref in LEGACY:
            diffs = (adm[DATES_ORDER].to_numpy() - legacy_date[ref][DATES_ORDER].to_numpy())
            res = sign_test_and_ci(diffs)
            tag = "ft" if "finetuned" in ref else "base"
            row[f"vs_{tag}_won"] = f"{res['dates_won_by_new']}/{res['n_dates']}"
            row[f"vs_{tag}_mean_diff"] = res["mean_diff"]
            row[f"vs_{tag}_ci"] = f"[{res['t_ci_lo']:+.4f}, {res['t_ci_hi']:+.4f}]"
            row[f"vs_{tag}_excl0"] = res["excludes_zero"]
            row[f"vs_{tag}_sign_p"] = res["sign_p_one_sided"]
            contrast_rows.append({"architecture": arch, "legacy_ref": ref_label,
                                  **{f"diff_{d}": float(x) for d, x in zip(DATES_ORDER, diffs)},
                                  **res})
        summary_rows.append(row)

    summary = pd.DataFrame(summary_rows).sort_values("iou_pooled", ascending=False).reset_index(drop=True)
    contrasts = pd.DataFrame(contrast_rows)
    return summary, contrasts


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--metrics-csv", type=Path, default=Path(DEFAULT_METRICS_CSV))
    ap.add_argument("--out-dir", type=Path, default=Path(DEFAULT_OUT_DIR))
    args = ap.parse_args()

    df = pd.read_csv(args.metrics_csv)
    assert set(NEW_ARCHS).issubset(set(df.arch)), "per_date_metrics.csv missing new architectures"
    summary, contrasts = build(df)

    # sanity: at n=3 the one-sided sign p can only be 0.125 (3/0), 0.5 (2/1), or 1.0 (>=), etc.
    allowed = {0.125, 0.5, 1.0}
    got = set(round(p, 3) for p in contrasts.sign_p_one_sided)
    assert got.issubset({round(x, 3) for x in allowed} | {1.0}), f"unexpected sign p at n=3: {got}"

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.out_dir / "all_models_vs_legacy_summary.csv", index=False)
    contrasts.to_csv(args.out_dir / "all_models_vs_legacy_contrasts.csv", index=False)

    ft = df[df.model == "resnet50_finetuned"].iou.mean()
    bl = df[df.model == "resnet50_baseline"].iou.mean()
    print(f"Legacy ResNet50-UNet pooled IoU: finetuned {ft:.4f}, baseline {bl:.4f}\n")
    print(f"{'architecture':13s} {'pooled':>7s} {'sd':>6s}  {'vs finetuned':>22s}  {'vs baseline':>22s}")
    for _, r in summary.iterrows():
        vf = f"{r.vs_ft_won} {r.vs_ft_mean_diff:+.4f}{'*' if r.vs_ft_excl0 else ' '}"
        vb = f"{r.vs_base_won} {r.vs_base_mean_diff:+.4f}{'*' if r.vs_base_excl0 else ' '}"
        print(f"{r.architecture:13s} {r.iou_pooled:7.4f} {r.seed_sd_max:6.4f}  {vf:>22s}  {vb:>22s}")
    print("\n* = date-as-unit 95% t-CI excludes zero (n=3). Sign-test floor at n=3 is p=0.125.")
    print(f"Wrote 2 CSVs to {args.out_dir}")


if __name__ == "__main__":
    main()
