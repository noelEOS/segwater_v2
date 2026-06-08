"""
Threshold Analysis: TOST Equivalence + Risk-adjusted Optimum.

Two complementary analyses to identify a statistically and practically defensible
optimal threshold range for binary segmentation models evaluated across multiple
image pairs.  All analyses run independently for each requested metric.

Usage
-----
    python scripts/analysis/threshold_analysis.py \\
        --input path/to/threshold_sweep.csv \\
        [--output-dir path/to/output]           \\
        [--delta 0.01]                          \\
        [--pair-col s1_id]                      \\
        [--alpha 0.05]                          \\
        [--metric iou mcc f1]

Input CSV schema (required columns)
------------------------------------
    model_name  : model identifier
    <pair_col>  : scene / pair identifier (default: s1_id)
    threshold   : float in [0, 1]
    iou / mcc / f1 / precision / recall  : metric values
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from scipy.stats import t as t_dist
from scipy.stats import ttest_1samp

VALID_METRICS = ["iou", "mcc", "f1", "precision", "recall"]
DEFAULT_T = 0.50  # conventional threshold shown on all panels


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Threshold analysis: TOST equivalence + risk-adjusted optimum."
    )
    p.add_argument("--input", required=True, help="Path to threshold_sweep.csv")
    p.add_argument("--output-dir", default=None,
                   help="Output directory (default: same folder as input CSV)")
    p.add_argument("--delta", type=float, default=0.01,
                   help="TOST margin: minimum metric difference considered practically "
                        "meaningful (default: 0.01)")
    p.add_argument("--pair-col", default="s1_id",
                   help="Column identifying individual scenes / pairs (default: s1_id)")
    p.add_argument("--alpha", type=float, default=0.05,
                   help="Significance level for TOST (default: 0.05)")
    p.add_argument("--metric", nargs="+", default=["iou"],
                   choices=VALID_METRICS, metavar="METRIC",
                   help=f"Metric(s) to analyse. Choices: {VALID_METRICS}. "
                        "Default: iou. Example: --metric iou mcc f1")
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# Analysis 1 — TOST Equivalence Testing
# ---------------------------------------------------------------------------

def run_tost(
    pivot: pd.DataFrame,
    thresholds: np.ndarray,
    t_peak: float,
    delta: float,
    alpha: float,
    n_pairs: int,
) -> dict:
    """
    For each threshold, run two one-sided paired t-tests (TOST) against t_peak.

    Bonferroni-corrects across all thresholds tested. Returns:
      equivalent  : bool array, one entry per threshold
      p_lower     : p-value for the lower one-sided test
      p_upper     : p-value for the upper one-sided test
      tost_low    : lower bound of the contiguous equivalence interval around t_peak
      tost_high   : upper bound
      skipped     : True if fewer than 5 pairs (unreliable TOST)
    """
    n_thresh = len(thresholds)

    if n_pairs < 5:
        warnings.warn(
            f"Only {n_pairs} pairs — TOST unreliable; treating all thresholds as equivalent.",
            stacklevel=2,
        )
        return {
            "equivalent": np.ones(n_thresh, dtype=bool),
            "p_lower": np.full(n_thresh, np.nan),
            "p_upper": np.full(n_thresh, np.nan),
            "tost_low": thresholds[0],
            "tost_high": thresholds[-1],
            "skipped": True,
        }

    peak_vec = pivot[t_peak].values
    alpha_corrected = alpha / max(n_thresh - 1, 1)  # Bonferroni

    p_lower = np.zeros(n_thresh)
    p_upper = np.zeros(n_thresh)

    for i, t in enumerate(thresholds):
        if t == t_peak:
            p_lower[i] = 1.0
            p_upper[i] = 1.0
            continue
        d = peak_vec - pivot[t].values
        res_lower = ttest_1samp(d - delta, popmean=0, alternative="less")
        res_upper = ttest_1samp(d + delta, popmean=0, alternative="greater")
        p_lower[i] = res_lower.pvalue
        p_upper[i] = res_upper.pvalue

    equivalent = (p_lower < alpha_corrected) & (p_upper < alpha_corrected)
    equivalent[np.searchsorted(thresholds, t_peak)] = True

    tost_low, tost_high = _contiguous_interval(thresholds, equivalent, t_peak)

    return {
        "equivalent": equivalent,
        "p_lower": p_lower,
        "p_upper": p_upper,
        "tost_low": tost_low,
        "tost_high": tost_high,
        "skipped": False,
    }


# ---------------------------------------------------------------------------
# Analysis 2 — Risk-adjusted Threshold
# ---------------------------------------------------------------------------

def run_risk_analysis(
    thresholds: np.ndarray,
    mean_metric: np.ndarray,
    std_metric: np.ndarray,
    tost_low: float,
    tost_high: float,
) -> dict:
    """
    Compute risk-adjusted curves mu(t) - λ·sigma(t) for λ ∈ {0.5, 1.0, 2.0}.

    Also computes the coefficient of variation and flags the low-CV zone
    (CV below its 25th percentile) as the 'low-variance interval'.
    """
    lambdas = [0.5, 1.0, 2.0]
    risk_curves: dict[float, np.ndarray] = {}
    t_risk: dict[float, float] = {}
    for lam in lambdas:
        curve = mean_metric - lam * std_metric
        risk_curves[lam] = curve
        t_risk[lam] = float(thresholds[np.argmax(curve)])

    with np.errstate(invalid="ignore", divide="ignore"):
        cv = np.where(mean_metric > 0, std_metric / mean_metric, np.nan)

    cv_25th = np.nanpercentile(cv, 25)
    low_var_mask = cv <= cv_25th
    anchor = thresholds[np.argmax(mean_metric)]
    low_var_low, low_var_high = _contiguous_interval(thresholds, low_var_mask, anchor)

    t_risk_in_tost = {lam: (tost_low <= t_risk[lam] <= tost_high) for lam in lambdas}

    return {
        "risk_curves": risk_curves,
        "t_risk": t_risk,
        "t_risk_in_tost": t_risk_in_tost,
        "cv": cv,
        "low_var_low": low_var_low,
        "low_var_high": low_var_high,
    }


# ---------------------------------------------------------------------------
# Consolidated Recommendation (per metric)
# ---------------------------------------------------------------------------

def make_recommendation(
    thresholds: np.ndarray,
    mean_metric: np.ndarray,
    t_peak: float,
    metric_peak: float,
    tost: dict,
    risk: dict,
    pivot: pd.DataFrame,
) -> dict:
    """
    Intersect TOST and low-variance intervals; fall back to risk-adjusted
    optimum at λ=1 if empty.  Computes per-pair t=0.50 comparison statistics.
    """
    tost_low, tost_high = tost["tost_low"], tost["tost_high"]
    lv_low, lv_high = risk["low_var_low"], risk["low_var_high"]

    inter_low = max(tost_low, lv_low)
    inter_high = min(tost_high, lv_high)
    in_intersection = thresholds[(thresholds >= inter_low) & (thresholds <= inter_high)]

    if len(in_intersection) > 0:
        m_in = mean_metric[(thresholds >= inter_low) & (thresholds <= inter_high)]
        rec_t = float(in_intersection[np.argmax(m_in)])
        justification = "TOST+low-var intersection"
    else:
        rec_t = risk["t_risk"][1.0]
        justification = "risk-adjusted (no intersection)"

    rec_idx = min(np.searchsorted(thresholds, rec_t), len(thresholds) - 1)
    rec_metric = float(mean_metric[rec_idx])

    # metric at t=0.50
    t05_idx = min(np.searchsorted(thresholds, DEFAULT_T), len(thresholds) - 1)
    t05_actual = float(thresholds[t05_idx])
    metric_at_t05 = float(mean_metric[t05_idx])
    default_in_tost = bool(tost_low <= DEFAULT_T <= tost_high)

    # per-pair signed differences: peak vs t=0.50
    peak_vec = pivot[t_peak].values
    t05_vec = pivot[t05_actual].values
    diffs = peak_vec - t05_vec
    n = len(diffs)

    sign_consistent = int((diffs > 0).sum())
    sign_consistency_str = f"{sign_consistent}/{n}"

    mean_diff = float(diffs.mean())
    if n > 1:
        se = float(diffs.std(ddof=1) / np.sqrt(n))
        t_crit = float(t_dist.ppf(0.975, df=n - 1))
        ci_lo = mean_diff - t_crit * se
        ci_hi = mean_diff + t_crit * se
    else:
        ci_lo = ci_hi = mean_diff

    def _fs(v: float) -> str:
        return f"{v:+.3f}"

    mean_signed_diff_str = f"{_fs(mean_diff)} [{_fs(ci_lo)}, {_fs(ci_hi)}]"

    return {
        "t_peak": t_peak,
        "metric_peak": metric_peak,
        "recommended_t": rec_t,
        "recommended_metric": rec_metric,
        "metric_loss_vs_peak": metric_peak - rec_metric,
        "tost_interval": f"[{tost_low:.4f}, {tost_high:.4f}]",
        "low_var_interval": f"[{lv_low:.4f}, {lv_high:.4f}]",
        "intersection_found": len(in_intersection) > 0,
        "justification": justification,
        "default_t05_in_tost": default_in_tost,
        "metric_loss_at_t05": round(metric_peak - metric_at_t05, 4),
        "metric_at_t05": round(metric_at_t05, 4),
        "sign_consistency_t05": sign_consistency_str,
        "mean_signed_diff_t05": mean_signed_diff_str,
        "tost_low": tost_low,
        "tost_high": tost_high,
    }


# ---------------------------------------------------------------------------
# Per-metric analysis driver
# ---------------------------------------------------------------------------

def analyse_metric(
    df_m: pd.DataFrame,
    pair_col: str,
    thresholds: np.ndarray,
    metric: str,
    delta: float,
    alpha: float,
    n_pairs: int,
) -> tuple[dict, dict, dict]:
    """Run TOST + risk analysis + recommendation for one model × one metric."""
    agg = df_m.groupby("threshold")[metric].agg(["mean", "std"]).reindex(thresholds)
    mean_vals = agg["mean"].values
    std_vals = agg["std"].fillna(0).values

    peak_idx = int(np.argmax(mean_vals))
    t_peak = float(thresholds[peak_idx])
    metric_peak = float(mean_vals[peak_idx])

    if metric_peak == 0:
        warnings.warn(
            f"Peak {metric} is 0 for this model — results may be unreliable.",
            stacklevel=2,
        )

    pivot = df_m.pivot_table(index=pair_col, columns="threshold", values=metric)

    tost = run_tost(pivot, thresholds, t_peak, delta, alpha, n_pairs)
    risk = run_risk_analysis(thresholds, mean_vals, std_vals,
                             tost["tost_low"], tost["tost_high"])
    rec = make_recommendation(thresholds, mean_vals, t_peak, metric_peak,
                               tost, risk, pivot)
    rec["tost_skipped"] = tost["skipped"]

    return (
        rec,
        {"mean": mean_vals, "std": std_vals, "peak_idx": peak_idx},
        {"tost": tost, "risk": risk},
    )


# ---------------------------------------------------------------------------
# Plotting — two panels, one plot per model × metric
# ---------------------------------------------------------------------------

def make_plot(
    model_name: str,
    metric: str,
    thresholds: np.ndarray,
    mean_vals: np.ndarray,
    std_vals: np.ndarray,
    n_pairs: int,
    t_peak: float,
    rec_t: float,
    tost: dict,
    risk: dict,
    delta: float,
    output_path: Path,
) -> None:
    """
    Two-panel figure:
      Panel 1 — Mean metric curve with 95% CI ribbon, TOST equivalence band,
                t_peak, recommended_t, and t=0.50 reference line.
      Panel 2 — Risk-adjusted curves with t=0.50 reference line;
                y-axis zoomed to decision-relevant region.
    """
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except OSError:
        plt.style.use("seaborn-whitegrid")

    fig, axes = plt.subplots(2, 1, figsize=(10, 9), sharex=True)
    fig.suptitle(f"{model_name}  [{metric.upper()}]", fontsize=13, fontweight="bold")

    ci = 1.96 * std_vals / np.sqrt(n_pairs)
    tost_low, tost_high = tost["tost_low"], tost["tost_high"]
    ylabel = metric.upper()

    # ── Panel 1: metric curve + TOST equivalence ──────────────────────────
    ax = axes[0]
    ax.fill_between(thresholds, mean_vals - ci, mean_vals + ci,
                    alpha=0.2, color="steelblue", label="95% CI")
    ax.plot(thresholds, mean_vals, color="steelblue", lw=2, label=f"Mean {ylabel}")
    ax.axvspan(tost_low, tost_high, alpha=0.12, color="green",
               label=f"TOST equiv. [{tost_low:.2f}, {tost_high:.2f}]")
    ax.axvline(t_peak, color="black", lw=1.5, linestyle="--",
               label=f"t_peak = {t_peak:.2f}")
    if abs(rec_t - t_peak) > 1e-6:
        ax.axvline(rec_t, color="darkorange", lw=1.5, linestyle=":",
                   label=f"t_rec = {rec_t:.2f}")
    ax.axvline(DEFAULT_T, color="red", lw=1.5, linestyle="-.",
               label=f"t = {DEFAULT_T:.2f} (default)")
    ax.text(0.02, 0.04, f"Δ = {delta} | n_pairs = {n_pairs}",
            transform=ax.transAxes, fontsize=8, va="bottom",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7))
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=8, loc="lower left", ncol=2)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))

    # ── Panel 2: Risk-adjusted curves ─────────────────────────────────────
    ax = axes[1]
    styles = {0.5: ("--", "μ − 0.5σ"),
              1.0: ("-.", "μ − 1.0σ"),
              2.0: (":", "μ − 2.0σ")}
    risk_colors = {0.5: "tomato", 1.0: "firebrick", 2.0: "darkred"}

    ax.plot(thresholds, mean_vals, color="steelblue", lw=2, label=f"Mean {ylabel} (μ)")
    for lam, (ls, label) in styles.items():
        ax.plot(thresholds, risk["risk_curves"][lam],
                color=risk_colors[lam], lw=1.5, linestyle=ls, label=label)
        ax.axvline(risk["t_risk"][lam], color=risk_colors[lam],
                   lw=1, linestyle=ls, alpha=0.7)
    ax.axvline(DEFAULT_T, color="red", lw=1.5, linestyle="-.",
               label=f"t = {DEFAULT_T:.2f} (default)")

    tost_mask = (thresholds >= tost_low) & (thresholds <= tost_high)
    if tost_mask.any():
        y_floor = max(0.0, float(mean_vals[tost_mask].min()) - 0.05)
    else:
        y_floor = max(0.0, float(mean_vals.min()) - 0.05)
    ax.set_ylim(bottom=y_floor)

    ax.set_ylabel(f"Risk-adjusted {ylabel}")
    ax.set_xlabel("Threshold")
    ax.legend(fontsize=8, loc="lower left", ncol=2)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------

def write_report(
    models: list[str],
    metrics: list[str],
    all_recs: dict[str, dict[str, dict]],  # model -> metric -> rec
    delta: float,
    alpha: float,
    output_path: Path,
) -> None:
    """Write multi-metric narrative markdown report with metric-agreement summary."""
    lines = [
        "# Threshold Analysis Report",
        "",
        f"**TOST margin Δ = {delta}** | **α = {alpha}** (Bonferroni-corrected) | "
        f"**Metrics:** {', '.join(m.upper() for m in metrics)}",
        "",
    ]

    # ── Metric agreement summary (IoU vs MCC, if both present) ────────────
    if "iou" in metrics and "mcc" in metrics:
        lines += ["---", "", "## Metric agreement: IoU vs MCC", ""]
        lines.append(
            "Models where the IoU-optimal and MCC-optimal thresholds differ by > 0.05 "
            "(prevalence sensitivity — worth discussing):"
        )
        lines.append("")

        flagged = []
        for model in models:
            t_iou = all_recs[model]["iou"]["t_peak"]
            t_mcc = all_recs[model]["mcc"]["t_peak"]
            if abs(t_iou - t_mcc) > 0.05:
                flagged.append(
                    f"- **{model}**: IoU peak at t = {t_iou:.2f}, "
                    f"MCC peak at t = {t_mcc:.2f} "
                    f"(Δ = {abs(t_iou - t_mcc):.2f})"
                )
        if flagged:
            lines += flagged
        else:
            lines.append(
                "_No models show IoU–MCC peak threshold disagreement > 0.05._"
            )
        lines += ["", "---", ""]

    # ── Per-model, per-metric sections ────────────────────────────────────
    for model in models:
        lines.append(f"## {model}")
        lines.append("")

        for metric in metrics:
            rec = all_recs[model][metric]
            tost_low_val = rec["tost_low"]
            tost_high_val = rec["tost_high"]
            m_label = metric.upper()

            lines.append(f"### {m_label}")
            lines.append("")
            lines.append(
                f"- **t_peak** = {rec['t_peak']} &ensp; "
                f"**{m_label}_peak** = {rec['metric_peak']:.4f}"
            )
            lines.append(f"- **TOST interval** = {rec['tost_interval']}")
            lines.append(
                f"- **Recommended threshold** = {rec['recommended_t']} "
                f"({rec['justification']})"
            )
            lines.append(
                f"- **Recommended {m_label}** = {rec['recommended_metric']:.4f} "
                f"&ensp; ({m_label} loss vs peak = {rec['metric_loss_vs_peak']:.4f})"
            )
            lines.append("")

            sign_str = rec["sign_consistency_t05"]
            diff_str = rec["mean_signed_diff_t05"]
            loss = rec["metric_loss_at_t05"]

            if rec["default_t05_in_tost"]:
                lines.append(
                    f"> **t = 0.50 falls within the TOST equivalence interval "
                    f"[{tost_low_val:.4f}, {tost_high_val:.4f}]: "
                    f"the default threshold is statistically justified for this model. "
                    f"Mean {m_label} gain of peak over t = 0.50: {diff_str} "
                    f"(peak better in {sign_str} pairs).**"
                )
            else:
                lines.append(
                    f"> **t = 0.50 falls outside the TOST equivalence interval "
                    f"[{tost_low_val:.4f}, {tost_high_val:.4f}]: "
                    f"performance at the default threshold is statistically "
                    f"distinguishable from peak ({m_label} loss = {loss:.4f}; "
                    f"mean signed difference: {diff_str}; "
                    f"peak better in {sign_str} pairs). "
                    f"Recommended threshold: {rec['recommended_t']}.**"
                )
            lines.append("")

        lines += ["---", ""]

    output_path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _contiguous_interval(
    thresholds: np.ndarray, mask: np.ndarray, anchor: float
) -> tuple[float, float]:
    """Return the contiguous run of True values in mask that contains anchor."""
    if not mask.any():
        return anchor, anchor
    anchor_idx = int(np.argmin(np.abs(thresholds - anchor)))
    lo = anchor_idx
    while lo > 0 and mask[lo - 1]:
        lo -= 1
    hi = anchor_idx
    while hi < len(thresholds) - 1 and mask[hi + 1]:
        hi += 1
    return float(thresholds[lo]), float(thresholds[hi])


def _safe_dirname(name: str) -> str:
    return name.replace("/", "_").replace(" ", "_")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None) -> None:
    args = parse_args(argv)

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    output_dir = Path(args.output_dir) if args.output_dir else input_path.parent
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    delta = args.delta
    alpha = args.alpha
    pair_col = args.pair_col
    metrics = list(dict.fromkeys(args.metric))  # deduplicate, preserve order

    df = pd.read_csv(input_path)

    required = {"model_name", pair_col, "threshold"} | set(metrics)
    missing_cols = required - set(df.columns)
    if missing_cols:
        raise ValueError(f"Input CSV missing required columns: {missing_cols}")

    out_of_range = df[(df["threshold"] < 0) | (df["threshold"] > 1)]
    if len(out_of_range):
        warnings.warn(f"Dropping {len(out_of_range)} rows with threshold outside [0, 1].")
        df = df[(df["threshold"] >= 0) & (df["threshold"] <= 1)]

    models = sorted(df["model_name"].unique())

    # all_recs[model][metric] = rec dict
    all_recs: dict[str, dict[str, dict]] = {m: {} for m in models}

    n_intersection_by_metric: dict[str, int] = {metric: 0 for metric in metrics}

    for model_name in models:
        df_m = df[df["model_name"] == model_name].copy()
        thresholds = np.sort(df_m["threshold"].unique())
        n_pairs = df_m[pair_col].nunique()

        for metric in metrics:
            rec, agg_data, analysis = analyse_metric(
                df_m, pair_col, thresholds, metric, delta, alpha, n_pairs,
            )
            all_recs[model_name][metric] = rec

            if rec["intersection_found"]:
                n_intersection_by_metric[metric] += 1

            plot_path = plots_dir / f"{_safe_dirname(model_name)}_{metric}_analysis.png"
            make_plot(
                model_name=model_name,
                metric=metric,
                thresholds=thresholds,
                mean_vals=agg_data["mean"],
                std_vals=agg_data["std"],
                n_pairs=n_pairs,
                t_peak=rec["t_peak"],
                rec_t=rec["recommended_t"],
                tost=analysis["tost"],
                risk=analysis["risk"],
                delta=delta,
                output_path=plot_path,
            )

    # ── Build wide recommendations CSV (one row per model) ────────────────
    rec_rows = []
    for model_name in models:
        df_m = df[df["model_name"] == model_name]
        n_pairs = df_m[pair_col].nunique()
        thresholds = np.sort(df_m["threshold"].unique())

        row: dict = {"model_name": model_name, "n_pairs": n_pairs}
        for metric in metrics:
            rec = all_recs[model_name][metric]
            agg = df_m.groupby("threshold")[metric].agg(["mean", "std"]).reindex(thresholds)
            peak_idx = int(np.argmax(agg["mean"].values))
            std_at_peak = round(float(agg["std"].fillna(0).values[peak_idx]), 4)

            p = metric + "_"
            row[p + "t_peak"] = round(rec["t_peak"], 4)
            row[p + "metric_peak"] = round(rec["metric_peak"], 4)
            row[p + "std_across_pairs"] = std_at_peak
            row[p + "tost_interval"] = rec["tost_interval"]
            row[p + "low_var_interval"] = rec["low_var_interval"]
            row[p + "recommended_t"] = round(rec["recommended_t"], 4)
            row[p + "recommended_metric"] = round(rec["recommended_metric"], 4)
            row[p + "metric_loss_vs_peak"] = round(rec["metric_loss_vs_peak"], 4)
            row[p + "default_t05_in_tost"] = rec["default_t05_in_tost"]
            row[p + "metric_loss_at_t05"] = rec["metric_loss_at_t05"]
            row[p + "sign_consistency_t05"] = rec["sign_consistency_t05"]
            row[p + "mean_signed_diff_t05"] = rec["mean_signed_diff_t05"]
            row[p + "justification"] = rec["justification"]
            row[p + "tost_skipped"] = rec.get("tost_skipped", False)

        rec_rows.append(row)

    df_rec = pd.DataFrame(rec_rows)
    rec_csv = output_dir / "threshold_recommendation.csv"
    df_rec.to_csv(rec_csv, index=False, float_format="%.4f")

    # ── Markdown report ───────────────────────────────────────────────────
    report_path = output_dir / "threshold_analysis_report.md"
    write_report(models, metrics, all_recs, delta=delta, alpha=alpha,
                 output_path=report_path)

    # ── Summary ───────────────────────────────────────────────────────────
    print(f"\nThreshold analysis complete")
    print(f"  Models processed : {len(models)}")
    for metric in metrics:
        n = n_intersection_by_metric[metric]
        print(f"  TOST+low-var [{metric:>9s}] : {n} / {len(models)}")
    print(f"  Recommendation CSV  : {rec_csv}")
    print(f"  Markdown report     : {report_path}")
    print(f"  Plots               : {plots_dir}")


if __name__ == "__main__":
    main()
