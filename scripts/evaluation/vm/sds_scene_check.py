#!/usr/bin/env python
"""Verify SDS scene staging: every staged scene inside the GT calendar window
(widened by max_days), and no in-window scene left parked. Read-only.
Exit 1 on any violation.

The staging rule (docs/RUNBOOK_sds_vm_eval.md): stage every scene of the frame
whose date falls inside the in-situ groundtruth's calendar window -- NOT the
scenes that happen to have a survey within `max_days`, which is a scoring knob.

    python scripts/evaluation/vm/sds_scene_check.py [--sds-root ~/SDS_Benchmark_slim]
"""
from __future__ import annotations
import argparse, datetime as dt, glob, os, pickle, re, sys
from pathlib import Path

HOME = Path.home()
# site -> (canonical frame, staged dir, parked dir or None)
CANON = {
    "NARRABEEN":   ("ron_147_ts_9_sn_16", HOME/"NARRABEEN_ron_147_ts_9_sn_16",           HOME/"NARRABEEN_no_groundtruth"),
    "DUCK":        ("ron_4_ts_24_sn_19",  HOME/"Inference_input/Duck_ron_4_ts_24_sn_19", HOME/"DUCK_ron4_no_groundtruth"),
    "TORREYPINES": ("ron_71",             HOME/"Inference_input/Torreypines_ron_71",     HOME/"TORREYPINES_ron71_no_groundtruth"),
    "TRUCVERT":    ("ron_8_ts_30_sn_20",  HOME/"Inference_input/TRUCVERT_ron_8_ts_30_sn_20", HOME/"TRUCVERT_ron8_no_groundtruth"),
}
MAX_DAYS = 10  # sds_core.py SETTINGS. Used two ways below, deliberately:
#   (a) to WIDEN the GT calendar window -- a scene a few days past the last
#       survey still pairs with it, so it belongs in the staged set;
#   (b) to REPORT the scorable count. It is never a per-scene staging filter.


def scene_date(name: str):
    m = re.search(r"_(\d{8})T", name)
    return dt.datetime.strptime(m.group(1), "%Y%m%d").date() if m else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sds-root", type=Path, default=HOME/"SDS_Benchmark_slim")
    args = ap.parse_args()
    fail = 0
    print("%-12s %-20s %6s %7s %10s %s" % ("site", "frame", "staged", "parked", "scorable", "status"))
    for site, (frame, src, parked) in CANON.items():
        gtf = args.sds_root/"datasets"/site/("%s_groundtruth_MSL.pkl" % site)
        if not gtf.exists():
            print("%-12s %-20s %6s %7s %10s MISS groundtruth %s" % (site, frame, "-", "-", "-", gtf)); fail = 1; continue
        if not src.is_dir():
            print("%-12s %-20s %6s %7s %10s MISS staged dir %s" % (site, frame, "-", "-", "-", src)); fail = 1; continue
        gt = pickle.load(open(gtf, "rb"))
        dates = sorted({d.date() for v in gt.values() for d in v["dates"]})
        # The window is the GT range WIDENED by max_days at both ends: a scene a
        # few days past the final survey still pairs with it, so it belongs in
        # the staged set. (Narrabeen 2019-11-30 is 3 d past the last survey and
        # is genuinely scorable; a bare lo<=d<=hi test wrongly flags it.)
        pad = dt.timedelta(days=MAX_DAYS)
        lo, hi = min(dates) - pad, max(dates) + pad
        staged = [scene_date(os.path.basename(p)) for p in glob.glob(str(src/"*.tif"))]
        staged = [d for d in staged if d]
        pk = [scene_date(os.path.basename(p)) for p in glob.glob(str(parked/"*.tif"))] if parked and parked.is_dir() else []
        pk = [d for d in pk if d]
        out_of_window = sum(1 for d in staged if not (lo <= d <= hi))   # staged but shouldn't be
        parked_in_window = sum(1 for d in pk if lo <= d <= hi)          # parked but should be staged
        scorable = sum(1 for d in staged if any(abs((d-g).days) <= MAX_DAYS for g in dates))
        msgs = []
        if out_of_window: msgs.append("!! %d staged OUTSIDE GT window" % out_of_window)
        if parked_in_window: msgs.append("!! %d parked INSIDE GT window" % parked_in_window)
        if msgs: fail = 1
        print("%-12s %-20s %6d %7d %10d %s" % (site, frame, len(staged), len(pk), scorable,
                                               "  ".join(msgs) if msgs else "OK"))
    print()
    print("scorable = at the default max_days=%d tolerance; reported only, NOT a staging criterion." % MAX_DAYS)
    print("READY (sds scenes)" if not fail else "NOT READY (sds scenes) — see docs/RUNBOOK_sds_vm_eval.md §Scene staging")
    sys.exit(fail)


if __name__ == "__main__":
    main()
