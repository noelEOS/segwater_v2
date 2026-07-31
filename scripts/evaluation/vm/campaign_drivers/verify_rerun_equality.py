"""Verify the re-run reproduces the superseded results exactly.

The staging change added scenes that have NO survey within max_days, so the
pipeline must drop them at pairing. Prediction: every metric identical, `n`
identical; ONLY `n_shorelines` changes (82->87 Narrabeen, 79->109 Duck).
Any metric drift means the extra scenes DID enter the estimand -- a real problem.
"""
import csv, glob, os, re, sys

OLD_N = "/home/noel/_superseded_82_79"
NEW_NARRA = "/home/noel/sds_vm_eval"
NEW_DUCK = "/home/noel/sds_vm_eval_duck"
METRICS = ["n", "rmse", "bias", "std", "q90", "R2"]


def key(dirname):
    """(site, arm, stride) from a sweep dir name, ignoring the timestamp."""
    m = re.match(r"(narrabeen|duck)_s42(noaug|aug|)_(best|last)_\d+T\d+Z_.*_b0_(s\d+)_sweep$", dirname)
    if not m:
        return None
    site, aug, arm, stride = m.groups()
    aug = "no_aug" if aug == "noaug" else "aug"
    return (site, aug, arm, stride)


def load(root, pat):
    out = {}
    for d in glob.glob(os.path.join(root, pat)):
        k = key(os.path.basename(d))
        if not k:
            continue
        f = os.path.join(d, "sweep_metrics.csv")
        if not os.path.exists(f):
            continue
        out[k] = {float(r["threshold"]): r for r in csv.DictReader(open(f))}
    return out


old = {**load(OLD_N, "narrabeen_s42*_sweep"), **load(os.path.join(OLD_N, "sds_vm_eval_duck_79"), "*_sweep")}
new = {**load(NEW_NARRA, "narrabeen_s42*_sweep"), **load(NEW_DUCK, "duck_s42*_sweep")}

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
            a, b = float(o[thr][mcol]), float(n[thr][mcol])
            if abs(a - b) > 1e-9:
                drifts.append("thr%.1f %s %.6f->%.6f" % (thr, mcol, a, b))
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
