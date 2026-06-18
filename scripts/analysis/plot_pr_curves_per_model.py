#!/usr/bin/env python3
"""
Generate manuscript-grade per-pair precision–recall curves from threshold_sweep.csv.

Each panel is one S1/S2 evaluation pair; the marker on the curve is that pair's
leave-one-scene-out (LOO) operating point (default: IoU-optimal threshold) read
from loo_threshold_evaluation.csv. For each held-out scene this threshold was
selected on the other five scenes, so it is an honest, out-of-scene operating
point and is in general different for every panel.

Outputs (into --outdir):
  <model_name>_per_pair_pr_curves_rse_style.png
  <model_name>_per_pair_pr_curves_rse_style.pdf
  per_model_pr_summary.csv
"""

from pathlib import Path
import re
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def sanitize_filename(name: str) -> str:
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", name)
    return name.strip("_")


def pair_label_from_s1_id(s1_id: str) -> str:
    """
    Example:
      S1_20190622_221715_76_8_9 -> 2019-06-22
    """
    parts = str(s1_id).split("_")
    if len(parts) >= 2 and len(parts[1]) == 8:
        d = parts[1]
        return f"{d[:4]}-{d[4:6]}-{d[6:]}"
    return str(s1_id)


def make_curve_from_threshold_sweep(sub: pd.DataFrame) -> pd.DataFrame:
    """
    Build PR curve from threshold sweep.

    Important correction:
    If TP=0 and FP=0, many pipelines store precision=0 to avoid division by zero.
    For PR-curve visualization, this should be treated as the conventional sentinel:
      recall = 0
      precision = 1
    """
    sub = sub.copy()

    no_positive_pred = (sub["tp"] == 0) & (sub["fp"] == 0)
    sub.loc[no_positive_pred, "precision"] = 1.0
    sub.loc[no_positive_pred, "recall"] = 0.0

    curve = (
        sub.groupby("recall", as_index=False)
        .agg({"precision": "max"})
        .sort_values("recall")
    )

    return curve


def plot_model_pr_curves(
    df_model: pd.DataFrame,
    model_name: str,
    outdir: Path,
    loo_by_s1_id: dict,
    dpi: int = 600,
):
    df_model = df_model.copy()
    df_model["pair"] = df_model["s1_id"].apply(pair_label_from_s1_id)

    pairs = sorted(df_model["pair"].unique())
    n_pairs = len(pairs)

    if n_pairs == 0:
        return []

    ncols = 2
    nrows = int(np.ceil(n_pairs / ncols))

    panel_labels = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

    # RSE-style figure settings
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "font.family": "DejaVu Sans",
        "font.size": 8.5,
        "axes.titlesize": 9,
        "axes.labelsize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.linewidth": 0.8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "mathtext.default": "regular",
    })

    # 3x2 for six pairs; scales if pair count changes
    fig_height = 2.75 * nrows + 0.8
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(7.1, fig_height),
        sharex=True,
        sharey=True,
    )

    axes = np.array(axes).reshape(-1)

    curve_color = "#1f4e79"   # restrained scientific blue
    marker_face = "#B05A43"       # "white"
    marker_edge = "#404040"
    text_color = "#222222"
    subtle_text = "#555555"
    spine_color = "#444444"

    summary_rows = []

    for ax, pair, panel_label in zip(axes, pairs, panel_labels):
        sub = df_model[df_model["pair"] == pair].copy()

        curve = make_curve_from_threshold_sweep(sub)

        # Approximate AP / PR-AUC from threshold sweep
        ap = float(
            np.trapezoid(
                curve["precision"].to_numpy(),
                curve["recall"].to_numpy(),
            )
        )

        # Operating point: this pair's leave-one-scene-out (LOO) IoU-optimal
        # threshold — selected on the other five scenes and applied here, so it
        # is an honest, out-of-scene point and differs from panel to panel.
        # Precision/recall are read directly from the LOO evaluation row.
        s1_id = sub["s1_id"].iloc[0]
        op = loo_by_s1_id[s1_id]

        ax.step(
            curve["recall"],
            curve["precision"],
            where="post",
            lw=1.6,
            color=curve_color,
        )

        ax.scatter(
            op["recall"],
            op["precision"],
            s=26,
            facecolor=marker_face,
            edgecolor=marker_edge,
            linewidth=0.9,
            zorder=3,
        )

        ax.set_xlim(0, 1.0)
        ax.set_ylim(0, 1.0)
        ax.set_xticks(np.linspace(0, 1, 6))
        ax.set_yticks(np.linspace(0, 1, 6))
        ax.tick_params(length=3, width=0.8, color=spine_color)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color(spine_color)
        ax.spines["bottom"].set_color(spine_color)
        ax.grid(False)

        # Panel label and pair/date
        ax.text(
            0.00,
            1.08,
            panel_label,
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=10.5,
            fontweight="bold",
            color=text_color,
        )

        ax.text(
            0.06,
            1.08,
            pair,
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=9.5,
            color=text_color,
        )

        # Compact in-panel metrics (evaluated at the LOO operating point)
        metrics = (
            f"AP = {ap:.3f}\n"
            f"F1 = {op['f1']:.3f}\n"
            f"$t_{{LOO}}$ = {op['threshold']:.2f}"
        )

        ax.text(
            0.12,
            0.24,
            metrics,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8,
            color=text_color,
        )

        summary_rows.append({
            "model_name": model_name,
            "pair": pair,
            "s1_id": s1_id,
            "approx_ap_threshold_sweep": ap,
            "loo_threshold": float(op["threshold"]),
            "f1_at_loo": float(op["f1"]),
            "iou_at_loo": float(op["iou"]),
            "precision_at_loo": float(op["precision"]),
            "recall_at_loo": float(op["recall"]),
        })

    # Hide unused axes if pair count is not exactly 2*nrows
    for ax in axes[n_pairs:]:
        ax.axis("off")

    fig.supxlabel("Recall", y=0.05, fontsize=10, color=text_color)
    fig.supylabel("Precision", x=0.04, fontsize=10, color=text_color)

    display_name = model_name.replace("_", "-")

    fig.text(
        0.08,
        0.975,
        display_name,
        ha="left",
        va="top",
        fontsize=9.5,
        color=text_color,
    )

    fig.text(
        0.08,
        0.958,
        "Per-pair precision-recall curves across the S1/S2 evaluation pairs",
        ha="left",
        va="top",
        fontsize=8.5,
        color=subtle_text,
    )

    fig.text(
        0.08,
        0.018,
        (
            "Filled circles indicate each pair's leave-one-scene-out IoU-optimal "
            "threshold, selected on the other five scenes. AP was approximated by "
            "trapezoidal integration of the threshold-sweep precision-recall curve."
        ),
        ha="left",
        va="bottom",
        fontsize=7.4,
        color=subtle_text,
    )

    plt.subplots_adjust(
        left=0.11,
        right=0.985,
        top=0.90,
        bottom=0.08,
        hspace=0.31,
        wspace=0.14,
    )

    outdir.mkdir(parents=True, exist_ok=True)
    safe_name = sanitize_filename(model_name)

    png_path = outdir / f"{safe_name}_per_pair_pr_curves_rse_style.png"
    pdf_path = outdir / f"{safe_name}_per_pair_pr_curves_rse_style.pdf"

    fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")

    return summary_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("threshold_sweep.csv"),
        help="Path to threshold_sweep.csv",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("figures/pr_curves"),
        help="Output directory for figures",
    )
    parser.add_argument(
        "--loo-evaluation",
        type=Path,
        default=None,
        help=(
            "Path to loo_threshold_evaluation.csv. The marker on each panel is "
            "that scene's leave-one-scene-out threshold (selected on the other "
            "five scenes). Defaults to loo_threshold_evaluation.csv next to --csv."
        ),
    )
    parser.add_argument(
        "--optimize-metric",
        default="iou",
        help="Which LOO optimize_metric row to use for the operating point (default: iou)",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=600,
        help="PNG export resolution",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.csv)

    loo_path = args.loo_evaluation or (args.csv.parent / "loo_threshold_evaluation.csv")
    if not loo_path.exists():
        raise FileNotFoundError(
            f"LOO threshold evaluation not found: {loo_path}. Pass --loo-evaluation explicitly."
        )
    df_loo = pd.read_csv(loo_path)
    df_loo = df_loo[df_loo["optimize_metric"] == args.optimize_metric]
    if df_loo.empty:
        raise ValueError(
            f"No rows with optimize_metric={args.optimize_metric!r} in {loo_path}"
        )
    # Per (model, scene) leave-one-scene-out operating point. Each row is already
    # evaluated at the held-out threshold, so precision/recall are read directly.
    loo_by_model = {
        model_name: {row["s1_id"]: row for _, row in g.iterrows()}
        for model_name, g in df_loo.groupby("model_name")
    }

    required_cols = {
        "model_name",
        "s1_id",
        "threshold",
        "tp",
        "fp",
        "precision",
        "recall",
        "f1",
        "iou",
    }

    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    all_summary_rows = []

    for model_name in sorted(df["model_name"].unique()):
        if model_name not in loo_by_model:
            print(f"SKIP {model_name}: no LOO threshold in {loo_path.name}")
            continue
        df_model = df[df["model_name"] == model_name].copy()
        summary_rows = plot_model_pr_curves(
            df_model=df_model,
            model_name=model_name,
            outdir=args.outdir,
            loo_by_s1_id=loo_by_model[model_name],
            dpi=args.dpi,
        )
        all_summary_rows.extend(summary_rows)

    summary_df = pd.DataFrame(all_summary_rows)
    summary_path = args.outdir / "per_model_pr_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()