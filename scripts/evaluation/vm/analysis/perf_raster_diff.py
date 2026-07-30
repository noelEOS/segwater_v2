"""EXPERIMENT: numeric divergence between fp32 baseline and bf16+tf32 rasters.

These levers are NOT bit-identical by design (tf32 is flagged as such in
configs/inference.yaml). The question is the MAGNITUDE of divergence relative
to the threshold sweep's 0.01 step, and how many pixels actually flip their
0.5 decision.

Run-dir matching is anchored on the _<UTC timestamp>_ that follows the sweep
name, so a baseline prefix cannot swallow its own _PERF sibling, and a
--base-ts / --perf-ts pin resolves the case where several baseline runs exist.
The anchoring itself lives in ``scripts/evaluation/vm/runsel.py`` -- this module
used to carry its own copy of the same regex.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from runsel import RunDirError, resolve_run_dirs  # noqa: E402

DEFAULT_RUNS_ROOT = "/home/noel/segwater_v2/outputs/inference/runs"


def find(runs_root, prefix, ts=None):
    """Run dirs for sweep ``prefix``, optionally pinned to one UTC timestamp.

    Thin wrapper over :func:`runsel.resolve_run_dirs` returning ``str`` paths,
    which is what the raster-diff body below consumes.
    """
    return [str(d) for d in resolve_run_dirs(runs_root, prefix, timestamp=ts)]


def resolve_pair(runs_root, base_prefix, perf_prefix, base_ts=None, perf_ts=None):
    """Resolve baseline and perf run dirs, or report why we cannot.

    Returns ``(b_dirs, p_dirs, problem)``. ``problem`` is ``None`` on success,
    otherwise a human-readable string the caller prints before continuing --
    the historical print-and-continue behavior, kept so a bad prefix in a batch
    of comparisons does not abort the rest.
    """
    try:
        b_dirs = find(runs_root, base_prefix, base_ts)
        p_dirs = find(runs_root, perf_prefix, perf_ts)
    except RunDirError as e:
        return [], [], str(e)
    return b_dirs, p_dirs, None


def duplicate_strides(b_dirs):
    """Baseline dirs sharing a stride suffix, keyed by that suffix.

    More than one baseline run for a stride means the comparison is ambiguous;
    the caller reports it and asks for a --base-ts pin.
    """
    by_stride = {}
    for d in b_dirs:
        by_stride.setdefault(d.rsplit("_", 1)[-1], []).append(d)
    return {k: v for k, v in by_stride.items() if len(v) > 1}


def scene_stats(b, p):
    """Per-scene (max|d|, mean|d|, flip fraction) for one run-dir pair."""
    import rasterio

    stats = []
    for s in sorted(os.listdir(b)):
        fb = os.path.join(b, s, s + "_probability_water.tif")
        fp = os.path.join(p, s, s + "_probability_water.tif")
        if not (os.path.exists(fb) and os.path.exists(fp)):
            continue
        with rasterio.open(fb) as db, rasterio.open(fp) as dp:
            x = db.read(1).astype(np.float64)
            y = dp.read(1).astype(np.float64)
        d = np.abs(x - y)
        stats.append((d.max(), d.mean(), np.mean((x > 0.5) != (y > 0.5))))
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("base_prefix")
    ap.add_argument("perf_prefix")
    ap.add_argument("label")
    ap.add_argument("--base-ts", default=None, help="pin baseline UTC timestamp")
    ap.add_argument("--perf-ts", default=None, help="pin perf UTC timestamp")
    ap.add_argument("--runs-root", default=DEFAULT_RUNS_ROOT,
                    help="inference runs root (default: %(default)s)")
    a = ap.parse_args()

    b_dirs, p_dirs, problem = resolve_pair(
        a.runs_root, a.base_prefix, a.perf_prefix, a.base_ts, a.perf_ts
    )
    if problem:
        print("%s: cannot resolve run dirs, skip\n  %s" % (a.label, problem))
        return
    print("%s: %d baseline dir(s), %d perf dir(s)" % (a.label, len(b_dirs), len(p_dirs)))
    dupes = duplicate_strides(b_dirs)
    if dupes:
        print("  NOTE: >1 baseline run for a stride; pin with --base-ts:")
        for k, v in sorted(dupes.items()):
            for d in v:
                print("        [%s] %s" % (k, os.path.basename(d)))
        return

    for b in b_dirs:
        stride = b.rsplit("_", 1)[-1]
        cand = [d for d in p_dirs if d.endswith("_" + stride)]
        if len(cand) != 1:
            print("  %s: %d perf match, skip" % (stride, len(cand)))
            continue
        p = cand[0]
        if os.path.basename(b) == os.path.basename(p):
            raise SystemExit(
                "self-comparison: baseline and perf resolved to the same run dir %r"
                % os.path.basename(b)
            )
        stats = scene_stats(b, p)
        if not stats:
            print("  %s: no comparable scenes" % stride)
            continue
        arr = np.array(stats)
        print("  %-6s n=%3d  max|d|=%.6f  mean|d|=%.8f  max flip=%.3e  mean flip=%.3e"
              % (stride, len(arr), arr[:, 0].max(), arr[:, 1].mean(),
                 arr[:, 2].max(), arr[:, 2].mean()))


if __name__ == "__main__":
    main()
