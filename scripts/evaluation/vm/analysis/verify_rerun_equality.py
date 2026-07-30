"""Verify the re-run reproduces the superseded results exactly.

The staging change added scenes that have NO survey within max_days, so the
pipeline must drop them at pairing. Prediction: every metric identical, `n`
identical; ONLY `n_shorelines` changes (82->87 Narrabeen, 79->109 Duck).
Any metric drift means the extra scenes DID enter the estimand -- a real problem.

Directory scanning is sorted and duplicate (site, aug, arm, stride) cells are a
hard error: this is a correctness verifier, so two dirs mapping to one cell must
not be silently resolved by iteration order.
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
import re
import sys

DEFAULT_OLD_N = "/home/noel/_superseded_82_79"
DEFAULT_NEW_NARRA = "/home/noel/sds_vm_eval"
DEFAULT_NEW_DUCK = "/home/noel/sds_vm_eval_duck"
DEFAULT_OLD_DUCK_SUBDIR = "sds_vm_eval_duck_79"
METRICS = ["n", "rmse", "bias", "std", "q90", "R2"]

DIR_RE = re.compile(
    r"(narrabeen|duck)_s42(noaug|aug|)_(best|last)_\d+T\d+Z_.*_b0_(s\d+)_sweep$"
)


def key(dirname):
    """(site, arm, stride) from a sweep dir name, ignoring the timestamp."""
    m = DIR_RE.match(dirname)
    if not m:
        return None
    site, aug, arm, stride = m.groups()
    aug = "no_aug" if aug == "noaug" else "aug"
    return (site, aug, arm, stride)


def collect(root, pat):
    """Map cell key -> dir path for every sweep dir under ``root`` matching ``pat``.

    Raises SystemExit when two dirs claim the same cell (previously
    last-write-wins, and which one won depended on glob order).
    """
    found = {}
    for d in sorted(glob.glob(os.path.join(root, pat))):
        k = key(os.path.basename(d))
        if not k:
            continue
        if k in found:
            raise SystemExit(
                "duplicate cell %r from %s and %s" % (k, found[k], d)
            )
        found[k] = d
    return found


def load(root, pat):
    out = {}
    for k, d in collect(root, pat).items():
        f = os.path.join(d, "sweep_metrics.csv")
        if not os.path.exists(f):
            continue
        with open(f) as fh:
            out[k] = {float(r["threshold"]): r for r in csv.DictReader(fh)}
    return out


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--old-root", default=DEFAULT_OLD_N,
                    help="superseded results root (default: %(default)s)")
    ap.add_argument("--old-duck-subdir", default=DEFAULT_OLD_DUCK_SUBDIR,
                    help="Duck subdir inside --old-root (default: %(default)s)")
    ap.add_argument("--new-narrabeen-root", default=DEFAULT_NEW_NARRA,
                    help="re-run Narrabeen root (default: %(default)s)")
    ap.add_argument("--new-duck-root", default=DEFAULT_NEW_DUCK,
                    help="re-run Duck root (default: %(default)s)")
    return ap.parse_args(argv)


def main(argv=None):
    a = parse_args(argv)

    old = {
        **load(a.old_root, "narrabeen_s42*_sweep"),
        **load(os.path.join(a.old_root, a.old_duck_subdir), "*_sweep"),
    }
    new = {
        **load(a.new_narrabeen_root, "narrabeen_s42*_sweep"),
        **load(a.new_duck_root, "duck_s42*_sweep"),
    }

    print("old cells: %d | new cells: %d" % (len(old), len(new)))
    common = sorted(set(old) & set(new))
    print("comparable cells: %d\n" % len(common))
    if not common:
        sys.exit("no overlap -- check naming")

    bad = 0
    print("%-9s %-7s %-5s %-5s %8s %10s  %s" % ("site", "aug", "arm", "strd", "n old/new", "n_sh old/new", "metric drift"))
    for k in common:
        o, n = old[k], new[k]
        drifts = []
        for thr in sorted(set(o) & set(n)):
            for mcol in METRICS:
                x, y = float(o[thr][mcol]), float(n[thr][mcol])
                if abs(x - y) > 1e-9:
                    drifts.append("thr%.1f %s %.6f->%.6f" % (thr, mcol, x, y))
        o5, n5 = o[0.5], n[0.5]
        status = "IDENTICAL" if not drifts else "!! %d diffs: %s" % (len(drifts), drifts[0])
        if drifts:
            bad += 1
        print("%-9s %-7s %-5s %-5s %4s/%-4s %5s/%-6s  %s" % (
            k[0], k[1], k[2], k[3], o5["n"], n5["n"], o5["n_shorelines"], n5["n_shorelines"], status))

    print()
    if bad:
        print("FAIL: %d/%d cells drifted -- the extra scenes CHANGED the estimand" % (bad, len(common)))
        sys.exit(1)
    print("PASS: all %d cells bit-identical; only n_shorelines changed (staging-only change confirmed)" % len(common))


if __name__ == "__main__":
    main()
