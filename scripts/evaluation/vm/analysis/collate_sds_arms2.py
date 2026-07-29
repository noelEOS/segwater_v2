"""Collate Narrabeen SDS sweep metrics across all five mx630 arms.

Reads sweep_metrics.csv from each run dir, tagging arch / lineage / arm / stride
so the `model` identity is legible from the row alone (two arms of DIFFERENT
architectures share the checkpoint filename step23930_last).
"""
import glob, os, re
import pandas as pd

ROOTS = ["/home/noel/sds_vm_eval_mx630s2", "/home/noel/sds_vm_eval_mx630k",
         "/home/noel/sds_vm_eval_mx630_arms2"]
# (dir-substring) -> (arch, lineage, arm)
SPEC = [("mx630s2_s42_best", ("swinb", "mx630s2", "best")),
        ("mx630s2_s42_last", ("swinb", "mx630s2", "last")),
        ("mx630s2_s42_swa5", ("swinb", "mx630s2", "swa5")),
        ("mx630k_s42_last",  ("cnxv2t", "mx630k", "last")),
        ("mx630k_s42_best",  ("cnxv2t", "mx630k", "best"))]

rows = []
for root in ROOTS:
    for d in sorted(glob.glob(os.path.join(root, "*_sweep"))):
        base = os.path.basename(d)
        hit = [v for k, v in SPEC if k in base]
        if len(hit) != 1:
            print("SKIP (%d spec matches): %s" % (len(hit), base)); continue
        arch, lineage, arm = hit[0]
        m = re.search(r"_b0_s(\d+)_sweep$", base)
        if not m:
            print("SKIP (no stride): %s" % base); continue
        f = os.path.join(d, "sweep_metrics.csv")
        if not os.path.exists(f):
            print("SKIP (no metrics): %s" % base); continue
        t = pd.read_csv(f)
        t["arch"], t["lineage"], t["arm"] = arch, lineage, arm
        t["model"] = "%s_s42_%s_%s" % (arch, lineage, arm)
        t["stride"] = int(m.group(1))
        t["site"], t["reference"], t["seed"] = "NARRABEEN", "MSL", 42
        t["run_dir"] = base
        rows.append(t)

df = pd.concat(rows, ignore_index=True)
out = "/home/noel/sds_narrabeen_mx630_arms_msl.csv"
df.to_csv(out, index=False)
print("wrote %s (%d rows; %d model x stride cells)" % (out, len(df), df.groupby(["model","stride"]).ngroups))
print(df.groupby(["model","stride"]).size().unstack(fill_value=0).to_string())
