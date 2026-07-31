"""Do the epsilon-floor / out-of-range pixels explain the VH moment gap?

Sample chips from the built train memmap, convert z->dB with the NEW constants, then recompute
moments under three populations:
  (a) all pixels                       -- what verify_memmaps.py measured
  (b) inside the histogram bin range   -- what the CONSTANTS were derived from
  (c) excluding only the eps/nodata floor
If (b) returns the mean to ~0, truncation is the explanation.
"""
import numpy as np

DST = "/mnt/local_ssd/dataset_mixed80/train.memmap"
N, C, H, W = 612364, 3, 224, 224
NEW = {"vv": (-15.747683, 6.706104), "vh": (-23.800326, 7.709898)}
# Histogram bin ranges the constants were derived over.
RANGE = {"vv": (-50.0, 8.0), "vh": (-60.0, 5.0)}

rng = np.random.default_rng(42)
idx = np.sort(rng.choice(N, size=3000, replace=False))
mm = np.memmap(DST, dtype=np.float16, mode="r", shape=(N, C, H, W))

for bi, band in enumerate(("vv", "vh")):
    mu, sd = NEW[band]
    lo, hi = RANGE[band]
    n_all = s_all = q_all = 0.0
    n_in = s_in = q_in = 0.0
    n_nf = s_nf = q_nf = 0.0
    for st in range(0, len(idx), 500):
        sel = idx[st:st + 500]
        z = np.asarray(mm[sel, bi], dtype=np.float64).ravel()
        db = z * sd + mu
        n_all += z.size; s_all += z.sum(); q_all += (z * z).sum()
        m = (db >= lo) & (db <= hi)                 # inside histogram range
        zi = z[m]; n_in += zi.size; s_in += zi.sum(); q_in += (zi * zi).sum()
        m2 = db > -75.0                             # exclude eps(-80)/nodata(-90) only
        zn = z[m2]; n_nf += zn.size; s_nf += zn.sum(); q_nf += (zn * zn).sum()

    def mom(n, s, q):
        m = s / n
        return m, float(np.sqrt(max(q / n - m * m, 0.0))), n

    print(f"\n=== {band.upper()} (new constants {mu:.6f} / {sd:.6f}) ===")
    for label, (m, s, n) in [
        ("(a) all pixels          ", mom(n_all, s_all, q_all)),
        (f"(b) inside [{lo},{hi}] dB", mom(n_in, s_in, q_in)),
        ("(c) excl eps/nodata     ", mom(n_nf, s_nf, q_nf)),
    ]:
        print(f"  {label}: mean={m:+.4f}  std={s:.4f}  n={n:,.0f}  kept={n/n_all:.4%}")
