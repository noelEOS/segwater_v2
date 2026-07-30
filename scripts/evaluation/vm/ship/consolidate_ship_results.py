"""Assemble the four ship-campaign gate CSVs into one decision table.

Produces `SHIP_DECISION_TABLE.csv` (one row per arm, one column block per gate)
plus a `SHIP_DECISION_SUMMARY.md` with the per-variant 3-seed aggregates.

Reads only what the scoring agents wrote; computes no metrics of its own beyond
aggregation, so a discrepancy here means a discrepancy upstream.

Two uncertainty kinds appear and are NOT interchangeable -- this script keeps
them in separate columns and labels both:
  * `*_hac_se`  -- within-arm Newey-West SE from one arm's time series;
  * `*_seed_sd` -- across-seed SD (ddof=1) over s19/s42/s58 for one variant.

Usage:
    python scripts/evaluation/vm/ship/consolidate_ship_results.py [--root DIR]
                                                                  [--strict]

`--strict` is for the FINAL consolidation pass: it refuses to succeed while any
expected source CSV is missing, any source row failed to join, or the ship
criterion (`gate_iou`) came out entirely NaN. Without it the script stays safe
to run repeatedly mid-campaign.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from naming import atomic_to_csv, atomic_write_text  # noqa: E402

SEEDS = ["s19", "s42", "s58"]
VARIANTS = ["best", "last"]

# Reference values, canonical spec, 206-scene window, thr 0.5. Context only --
# never recomputed here.
REFERENCES = {
    "S2 optical anchor": (314.0, 36.5),
    "Landsat anchor": (293.1, 31.4),
    "registered chip-based": (340.4, 30.8),
}


def _seed_key(v) -> str:
    """Normalize a seed cell to the canonical ``sNN`` key form.

    ``19``, ``"19"`` and ``"s19"`` all become ``"s19"``.

    Incident class this closes: the join key here is ``(seed, variant)``, and
    two of the four ingest loops used to compare the raw cell against the
    ``"s19"``-style keys. A gate CSV that wrote its seed as a bare integer
    (pandas reads ``19``, not ``"s19"``) therefore matched nothing and left
    ``gate_iou`` -- the ship criterion itself -- all-NaN in the decision table,
    with no error and no missing-file warning. Every loop now normalizes
    through this one helper, and unmatched rows are counted and reported.
    """
    s = str(v)
    return s if s.startswith("s") else "s" + s


def _read(p: Path, what: str):
    if not p.exists():
        print("  MISSING (%s): %s" % (what, p))
        return None
    return pd.read_csv(p)


class _Tally:
    """Per-source read/matched/unmatched accounting for the (seed, variant) join."""

    def __init__(self) -> None:
        self.stats: dict[str, list[int]] = {}
        self.unmatched: list[tuple[str, tuple]] = []
        self.missing: list[str] = []

    def missing_source(self, source: str) -> None:
        self.missing.append(source)

    def read(self, source: str, n: int) -> None:
        self.stats.setdefault(source, [0, 0, 0])[0] += n

    def matched(self, source: str) -> None:
        self.stats.setdefault(source, [0, 0, 0])[1] += 1

    def miss(self, source: str, key) -> None:
        self.stats.setdefault(source, [0, 0, 0])[2] += 1
        self.unmatched.append((source, key))

    def report(self) -> None:
        print("\n=== source join accounting ===")
        for source, (n, m, k) in self.stats.items():
            print("%s%s: %d rows read, %d matched, %d unmatched"
                  % ("!! " if k else "", source, n, m, k))
        for source, key in self.unmatched:
            print("!!   unmatched %s key as seen: %r" % (source, key))
        for source in self.missing:
            print("(missing source: %s)" % source)

    @property
    def n_unmatched(self) -> int:
        return len(self.unmatched)

    def zero_matched_sources(self) -> list[str]:
        """Sources whose CSV was read but produced no joined row at all."""
        return [s for s, (n, m, _k) in self.stats.items() if n > 0 and m == 0]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path,
                    default=Path.home() / "workspace/results/ship_decision_2026-07")
    ap.add_argument("--strict", action="store_true",
                    help="final-pass mode: exit 1 on any missing source, any "
                         "unmatched row, or an all-NaN gate_iou")
    a = ap.parse_args()
    R = a.root
    tally = _Tally()

    rows = {(s, v): {"seed": s, "variant": v, "arm": "swinb_%s_mx630s2_%s" % (s, v)}
            for s in SEEDS for v in VARIANTS}

    # --- Demak concurrent gate (the acceptance test) ------------------------
    g = _read(R / "demak_gate/demak_gate_ship_summary.csv", "demak gate")
    if g is None:
        tally.missing_source("demak_gate")
    else:
        tally.read("demak_gate", len(g))
        for _, r in g.iterrows():
            k = (_seed_key(r["seed"]), r["variant"])
            if k in rows:
                tally.matched("demak_gate")
                rows[k].update(gate_iou=r.get("iou_at_0p5"), gate_tau=r.get("tau_star"),
                               gate_iou_tau=r.get("iou_at_tau_star"),
                               gate_area_bias=r.get("area_bias"), gate_auc=r.get("roc_auc"))
            else:
                tally.miss("demak_gate", (r["seed"], r["variant"]))

    # --- Hampyeong (corroboration only; cannot gate) ------------------------
    h = _read(R / "hampyeong/hampyeong_ship_per_date_metrics.csv", "hampyeong")
    if h is None:
        tally.missing_source("hampyeong")
    else:
        groups = list(h.groupby(["seed", "variant"]))
        tally.read("hampyeong", len(groups))
        for (s0, v), sub in groups:
            s = _seed_key(s0)
            if (s, v) in rows:
                tally.matched("hampyeong")
                rows[(s, v)]["hamp_mean3_iou"] = sub["iou"].mean()
            else:
                tally.miss("hampyeong", (s0, v))

    # --- Narrabeen SDS at thr 0.5 ------------------------------------------
    n = _read(R / "narrabeen/sds_narrabeen_ship_msl.csv", "narrabeen sds")
    if n is None:
        tally.missing_source("narrabeen_sds")
    else:
        t = n[(n.threshold - 0.5).abs() < 1e-9]
        groups = list(t.groupby(["seed", "variant"]))
        tally.read("narrabeen_sds", len(groups))
        for (s0, v), sub in groups:
            s = _seed_key(s0)
            if (s, v) in rows:
                tally.matched("narrabeen_sds")
                rows[(s, v)].update(
                    sds_rmse_mean=sub["rmse"].mean(), sds_bias_mean=sub["bias"].mean(),
                    sds_std_mean=sub["std"].mean(),
                    **{"sds_rmse_s%d" % st: sub[sub.stride == st]["rmse"].mean()
                       for st in (8, 32, 112) if (sub.stride == st).any()})
            else:
                tally.miss("narrabeen_sds", (s0, v))

    # --- Demak trend, both strides -----------------------------------------
    for stride in (32, 112):
        source = "trend_s%d" % stride
        t = _read(R / ("demak_trend/demak_full_ship_trend_s%d.csv" % stride), "trend s%d" % stride)
        if t is None:
            tally.missing_source(source)
            continue
        tally.read(source, len(t))
        for _, r in t.iterrows():
            k = (_seed_key(r["seed"]), r["variant"])
            if k in rows:
                tally.matched(source)
                rows[k]["trend_s%d" % stride] = r.get("slope_ha_yr")
                rows[k]["trend_s%d_hac_se" % stride] = r.get("hac_se")
            else:
                tally.miss(source, (r["seed"], r["variant"]))

    df = pd.DataFrame([rows[(s, v)] for v in VARIANTS for s in SEEDS])
    out = R / "SHIP_DECISION_TABLE.csv"
    atomic_to_csv(df, out, index=False)
    print("\nwrote %s (%d arms)" % (out, len(df)))
    print(df.to_string(index=False))

    # --- per-variant 3-seed aggregates -------------------------------------
    lines = ["# Ship decision — consolidated\n",
             "Per-variant aggregates are **3-seed mean ± SD (ddof=1)** across "
             "s19/s42/s58. This is a different quantity from a per-arm HAC SE "
             "(within-arm, from one time series); the two are not "
             "interchangeable and are never combined.\n"]
    num = [c for c in df.columns
           if c not in ("seed", "variant", "arm")
           and pd.api.types.is_numeric_dtype(df[c]) and df[c].notna().any()]
    if not num:
        # Mid-campaign: no gate CSV has landed yet. Emit the arm skeleton and
        # stop rather than raising -- this script is meant to be safe to run
        # repeatedly while results accumulate. (A source CSV that WAS read but
        # joined nothing is a different animal and still fails below.)
        print("\n(no gate results yet -- skeleton written, aggregates skipped)")
        tally.report()
        raise SystemExit(_exit_code(tally, df, g is not None, strict=a.strict))
    agg = df.groupby("variant")[num].agg(["mean", lambda x: x.std(ddof=1)])
    agg.columns = ["%s_%s" % (c, "mean" if k == "mean" else "seed_sd")
                   for c, k in agg.columns]
    print("\n=== per-variant 3-seed aggregates ===")
    print(agg.T.to_string())
    lines.append("```\n" + agg.T.to_string() + "\n```\n")

    lines.append("\n## Trend references (context; not recomputed)\n")
    for k, (v, se) in REFERENCES.items():
        lines.append("- %s: %+.1f ± %.1f ha/yr" % (k, v, se))
    atomic_write_text(R / "SHIP_DECISION_SUMMARY.md", "\n".join(lines) + "\n")
    print("\nwrote %s" % (R / "SHIP_DECISION_SUMMARY.md"))

    tally.report()
    raise SystemExit(_exit_code(tally, df, g is not None, strict=a.strict))


def _exit_code(tally: "_Tally", df, gate_read: bool, *, strict: bool) -> int:
    """0 iff the consolidation is trustworthy at the requested strictness.

    Always fatal (any mode): a source CSV that was read yet joined ZERO rows.
    That is a key-space mismatch -- the failure the seed-token bug produced --
    not an incomplete campaign, so it must never pass silently.

    Additionally fatal under ``--strict`` (final pass): any missing source, any
    single unmatched row, or a gate CSV that was read while ``gate_iou`` is
    entirely NaN.
    """
    fatal = []
    zero = tally.zero_matched_sources()
    if zero:
        fatal.append("source(s) read but ZERO rows joined: %s" % ", ".join(zero))
    if strict:
        if tally.missing:
            fatal.append("missing source(s): %s" % ", ".join(tally.missing))
        if tally.n_unmatched:
            fatal.append("%d unmatched source row(s)" % tally.n_unmatched)
        if gate_read and ("gate_iou" not in df.columns
                          or not df["gate_iou"].notna().any()):
            fatal.append("gate_iou is entirely NaN although the gate CSV was read")
    if fatal:
        print("\n!! CONSOLIDATION FAILED%s:" % (" (--strict)" if strict else ""))
        for m in fatal:
            print("!!   %s" % m)
        return 1
    return 0


if __name__ == "__main__":
    main()
