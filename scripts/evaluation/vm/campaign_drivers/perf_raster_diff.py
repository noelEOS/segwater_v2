"""EXPERIMENT: numeric divergence between fp32 baseline and bf16+tf32 rasters.

These levers are NOT bit-identical by design (tf32 is flagged as such in
configs/inference.yaml). The question is the MAGNITUDE of divergence relative
to the threshold sweep's 0.01 step, and how many pixels actually flip their
0.5 decision.

Run-dir matching is anchored on the _<UTC timestamp>_ that follows the sweep
name, so a baseline prefix cannot swallow its own _PERF sibling, and a
--base-ts / --perf-ts pin resolves the case where several baseline runs exist.
"""
import argparse
import glob
import os
import re

import numpy as np
import rasterio

RUNS = "/home/noel/segwater_v2/outputs/inference/runs"


def find(prefix, ts=None):
    pat = re.compile(r"^" + re.escape(prefix) + r"_(\d{8}T\d{6}Z)_")
    out = []
    for d in sorted(glob.glob(os.path.join(RUNS, prefix + "_*"))):
        m = pat.match(os.path.basename(d))
        if os.path.isdir(d) and m and (ts is None or m.group(1) == ts):
            out.append(d)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("base_prefix")
    ap.add_argument("perf_prefix")
    ap.add_argument("label")
    ap.add_argument("--base-ts", default=None, help="pin baseline UTC timestamp")
    ap.add_argument("--perf-ts", default=None, help="pin perf UTC timestamp")
    a = ap.parse_args()

    b_dirs, p_dirs = find(a.base_prefix, a.base_ts), find(a.perf_prefix, a.perf_ts)
    print("%s: %d baseline dir(s), %d perf dir(s)" % (a.label, len(b_dirs), len(p_dirs)))
    by_stride = {}
    for d in b_dirs:
        by_stride.setdefault(d.rsplit("_", 1)[-1], []).append(d)
    dupes = {k: v for k, v in by_stride.items() if len(v) > 1}
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
        assert os.path.basename(b) != os.path.basename(p), "self-comparison"
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
        if not stats:
            print("  %s: no comparable scenes" % stride)
            continue
        arr = np.array(stats)
        print("  %-6s n=%3d  max|d|=%.6f  mean|d|=%.8f  max flip=%.3e  mean flip=%.3e"
              % (stride, len(arr), arr[:, 0].max(), arr[:, 1].mean(),
                 arr[:, 2].max(), arr[:, 2].mean()))


if __name__ == "__main__":
    main()
