"""Clone the mx630s2_best gate-eval config for the two new arms.

Discovers each arm's inference run dir by sweep-name prefix, asserts exactly one
match, and writes an eval config identical to the reference except for name/run_dir.
"""
import glob, os, re, sys

REF = "/home/noel/demak_mx630s2_eval/demak_gate_mx630s2_s42_best/config.yaml"
RUNS = "/home/noel/segwater_v2/outputs/inference/runs"
ARMS = {"mx630s2_swa5": "demak_gate_mx630s2_s42_swa5",
        "mx630k_best":  "demak_gate_mx630k_s42_best"}

ref = open(REF).read()
for arm, sweep in ARMS.items():
    hits = sorted(glob.glob(os.path.join(RUNS, sweep + "_*")))
    if len(hits) != 1:
        print("SKIP %s: %d run dirs %s" % (arm, len(hits), hits)); continue
    out = ref
    out = re.sub(r'  name: ".*"', '  name: "demak_gate_%s"' % arm, out, count=1)
    out = re.sub(r'    Upernet_\S+:', '    Gate_%s:' % arm, out, count=1)
    out = re.sub(r'      run_dir: ".*"', '      run_dir: "%s"' % hits[0], out, count=1)
    out = re.sub(r'  run_name: ".*"', '  run_name: "demak_gate_%s"' % arm, out, count=1)
    p = "/home/noel/configs/mx630_arms2/eval_demak_gate_%s.yaml" % arm
    open(p, "w").write(out)
    print("wrote %s -> %s" % (p, os.path.basename(hits[0])))
