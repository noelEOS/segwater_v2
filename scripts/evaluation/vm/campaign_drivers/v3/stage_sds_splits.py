"""Park every SDS scene outside its site groundtruth calendar window (+/-max_days).

The staging rule (docs/RUNBOOK_sds_vm_eval.md): stage every scene of the frame
whose date falls inside the in-situ window, padded by max_days at BOTH ends so a
scene just past the last survey still pairs. Never stage on the max_days
tolerance itself -- that is a scoring knob, not a property of the data.

Parked scenes are MOVED, never deleted; each parked dir gets a restore command.
"""
import pickle, glob, os, re, datetime as dt, json, shutil

SRC = {
    "NARRABEEN":   ("Inference_input/NARRABEEN_ron_147_ts_9_sn_16", "NARRABEEN_no_groundtruth"),
    "DUCK":        ("Inference_input/Duck_ron_4_ts_24_sn_19",       None),
    "TORREYPINES": ("Inference_input/Torreypines_ron_71",           "TORREYPINES_ron71_no_groundtruth"),
    "TRUCVERT":    ("Inference_input/TRUCVERT_ron_8_ts_30_sn_20",   "TRUCVERT_ron8_no_groundtruth"),
}
EXPECTED = {"NARRABEEN": (87, 82), "DUCK": (109, 79),
            "TORREYPINES": (25, 15), "TRUCVERT": (78, 73)}
H = os.path.expanduser("~")
MAXD = 10
SLIM = os.path.join(H, "SDS_Benchmark_slim")

for site, (rel, park) in SRC.items():
    gt = pickle.load(open(f"{SLIM}/datasets/{site}/{site}_groundtruth_MSL.pkl", "rb"))
    dates = sorted({d.date() for v in gt.values() for d in v["dates"]})
    lo, hi = min(dates), max(dates)
    keep, drop, scorable = [], [], 0
    for p in sorted(glob.glob(f"{H}/{rel}/*.tif")):
        b = os.path.basename(p)
        m = re.search(r"_(\d{8})T", b)
        if not m:
            drop.append(b); continue
        d = dt.datetime.strptime(m.group(1), "%Y%m%d").date()
        if lo - dt.timedelta(days=MAXD) <= d <= hi + dt.timedelta(days=MAXD):
            keep.append(b)
            scorable += any(abs((d - g).days) <= MAXD for g in dates)
        else:
            drop.append(b)
    exp = EXPECTED[site]
    tag = "OK  " if (len(keep), scorable) == exp else "DIFF"
    print("%s %-12s staged %4d  park %4d  scorable@10d %3d   (expect %d/%d)"
          % (tag, site, len(keep), len(drop), scorable, exp[0], exp[1]))
    if drop and park:
        pd = os.path.join(H, park)
        os.makedirs(pd, exist_ok=True)
        moved = 0
        for b in drop:
            s = f"{H}/{rel}/{b}"
            if os.path.exists(s):
                shutil.move(s, os.path.join(pd, b)); moved += 1
        with open(os.path.join(pd, "README.md"), "w") as f:
            f.write("# Parked: %s scenes outside the groundtruth window\n\n"
                    "%d scenes whose acquisition date falls outside the in-situ\n"
                    "groundtruth calendar window %s..%s, padded by max_days=%d at both\n"
                    "ends. They cannot pair with any survey, so they are useless AGAINST\n"
                    "THIS GROUNDTRUTH -- but they are not junk: they stay needed for\n"
                    "full-series/trend work and if newer surveys appear.\n\n"
                    "Restore:\n  mv ~/%s/*.tif ~/%s/\n"
                    % (site, moved, lo, hi, MAXD, park, rel))
        print("     parked %d -> ~/%s/" % (moved, park))
    json.dump({"site": site, "gt_window": [str(lo), str(hi)], "max_days_pad": MAXD,
               "n_staged": len(keep), "n_parked": len(drop),
               "n_scorable_at_10d": scorable, "staged": keep, "parked": drop},
              open(f"{H}/{site}_sds_split.json", "w"), indent=1)
