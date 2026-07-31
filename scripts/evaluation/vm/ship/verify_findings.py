"""Re-derive every quantitative claim in a campaign FINDINGS.md and diff vs the prose.

WHY THIS EXISTS
---------------
In the ConvNeXtV2-Base campaign four derived values (ranges, differences,
threshold tests) were asserted in prose before being computed, and all four were
wrong -- caught only by a late manual spot-check. The pipeline CSVs were never in
doubt; the narrative around them was. This closes that gap the same way the kit
closes its others: with a check, not a resolution to be careful.

It recomputes each value from the campaign's own CSVs and searches the document
for the formatted result. A value reported as MISSING is either absent from the
prose (fine -- not every derived quantity is quoted) or quoted WRONG (not fine).
The point is that the list is short enough to eyeball, so a wrong number cannot
hide among hundreds of right ones.

Usage:
    python scripts/evaluation/vm/ship/verify_findings.py --campaign-dir DIR --tag cnxt
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--campaign-dir", type=Path, required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--doc", default="FINDINGS.md")
    a = ap.parse_args()
    D, T = a.campaign_dir, a.tag
    txt = (D / a.doc).read_text()
    res = []

    # The document is prose: it uses a Unicode minus in tables and rounds to the
    # precision that reads well, which differs per table. A checker that demands
    # one literal spelling buries a real error among dozens of formatting false
    # positives -- which defeats the purpose. So a value counts as FOUND if any
    # of its plausible renderings appears: ASCII or Unicode minus, at the
    # requested precision or one digit coarser.
    def renderings(val, fmt):
        digits = int(fmt[2]) if len(fmt) > 2 and fmt[2].isdigit() else 4
        out = set()
        for d in {digits, max(0, digits - 1)}:
            s = "%.*f" % (d, val)
            out.add(s)
            out.add(s.replace("-", "\u2212"))       # Unicode minus
            if not s.startswith("-"):
                out.add("+" + s)                    # explicit-positive trends
        return out

    def chk(label, val, fmt="%.4f"):
        cands = renderings(val, fmt)
        res.append((label, fmt % val, any(c in txt for c in cands)))

    g = pd.read_csv(D / "demak_gate" / ("demak_gate_%s_summary.csv" % T))
    for r in g.itertuples():
        chk("gate %s/%s IoU" % (r.seed, r.variant), r.iou_at_0p5)
        chk("gate %s/%s tau" % (r.seed, r.variant), r.tau_star, "%.2f")
        chk("gate %s/%s bias" % (r.seed, r.variant), r.area_bias)
        chk("gate %s/%s auc" % (r.seed, r.variant), r.roc_auc)
    for v, s in g.groupby("variant"):
        for col, f in (("iou_at_0p5", "%.4f"), ("tau_star", "%.3f"),
                       ("area_bias", "%.3f"), ("roc_auc", "%.4f")):
            chk("gate %s %s mean" % (v, col), s[col].mean(), f)
            chk("gate %s %s sd" % (v, col), s[col].std(ddof=1), f)
    chk("gate AUC spread", g.roc_auc.max() - g.roc_auc.min())

    h = pd.read_csv(D / "hampyeong" / ("hampyeong_%s_per_date_metrics.csv" % T))
    hm = h.groupby(["seed", "variant"]).iou.mean().reset_index()
    for r in hm.itertuples():
        chk("hamp s%s/%s" % (r.seed, r.variant), r.iou)
    for v, s in hm.groupby("variant"):
        chk("hamp %s mean" % v, s.iou.mean()); chk("hamp %s sd" % v, s.iou.std(ddof=1))

    n = pd.read_csv(D / "narrabeen" / ("sds_narrabeen_%s_msl.csv" % T))
    t = n[(n.threshold - 0.5).abs() < 1e-9]
    for st in (8, 32, 112):
        for v in ("best", "last"):
            s = t[(t.stride == st) & (t.variant == v)]
            for col in ("rmse", "bias", "std"):
                chk("sds s%d/%s %s" % (st, v, col), s[col].mean(), "%.2f")
                chk("sds s%d/%s %s sd" % (st, v, col), s[col].std(ddof=1), "%.2f")
    w = t.pivot_table(index=["seed", "stride"], columns="variant", values="rmse")
    chk("sds max margin", (w["best"] - w["last"]).max(), "%.2f")

    for st in (32, 112):
        c = pd.read_csv(D / "demak_trend" / ("demak_full_%s_trend_s%d.csv" % (T, st)))
        for r in c.itertuples():
            chk("trend s%d %s/%s" % (st, r.seed, r.variant), r.slope_ha_yr, "%.1f")
        for v, s in c.groupby("variant"):
            chk("trend s%d %s mean" % (st, v), s.slope_ha_yr.mean(), "%.1f")
            chk("trend s%d %s sd" % (st, v), s.slope_ha_yr.std(ddof=1), "%.1f")
    s2 = pd.read_csv(D / "demak_trend" / ("demak_full_%s_trend_s2matched.csv" % T))
    for (st, v), s in s2.groupby(["stride", "variant"]):
        chk("s2m s%d %s mean" % (st, v), s.slope_ha_yr.mean(), "%.1f")
        chk("s2m s%d %s sd" % (st, v), s.slope_ha_yr.std(ddof=1), "%.1f")
    chk("s2m arm min", s2.slope_ha_yr.min(), "%.1f")
    chk("s2m arm max", s2.slope_ha_yr.max(), "%.1f")
    chk("s2m vs chip anchor min", s2.slope_ha_yr.min() - 340.4, "%.1f")
    chk("s2m vs chip anchor max", s2.slope_ha_yr.max() - 340.4, "%.1f")

    # seed-SD ratios (best/last) -- the derived quantities most easily fumbled
    m = g.merge(hm.assign(seed="s" + hm.seed.astype(str)), on=["seed", "variant"])
    for col in ("iou_at_0p5", "area_bias"):
        b = m[m.variant == "best"][col].std(ddof=1)
        l = m[m.variant == "last"][col].std(ddof=1)
        chk("ratio %s" % col, b / l, "%.1f")

    found = sum(1 for _, _, ok in res if ok)
    print("checked %d derived values; FOUND in prose %d; not found %d"
          % (len(res), found, len(res) - found))
    missing = [(k, v) for k, v, ok in res if not ok]
    if missing:
        print("\n=== computed from CSV, NOT present in the document ===")
        print("(each is either not quoted -- fine -- or quoted WRONG -- not fine)")
        for k, v in missing:
            print("  %-32s %s" % (k, v))
    sys.exit(0)


if __name__ == "__main__":
    main()
