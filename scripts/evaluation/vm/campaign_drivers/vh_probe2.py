"""Where does the VH offset come from? Compare the SOURCE chips' moments against the constants."""
import numpy as np, pandas as pd

KEEP="/home/noel/segwater_v2/experiments/DATASET_mixed80_blocked_612k/keeplist/chip_keeplist.parquet"
SRC="/mnt/local_ssd/dataset"
SRC_N={"train":1032004,"val":291771,"test":150272}
C,H,W=3,224,224
OLD={"vv":(-15.510920639155525,6.564080466988128),"vh":(-23.573757647507851,7.6567358472184)}
NEW={"vv":(-15.747683,6.706104),"vh":(-23.800326,7.709898)}

keep=pd.read_parquet(KEEP)
tr=keep[keep.dest_split=="train"]
rng=np.random.default_rng(42)
samp=tr.sample(3000,random_state=42).sort_values(["src_split","src_row"])
mms={s:np.memmap(f"{SRC}/{s}.memmap",dtype=np.float32,mode="r",shape=(SRC_N[s],C,H,W)) for s in SRC_N}

acc={b:[0.0,0.0,0.0] for b in ("vv","vh")}
for s,grp in samp.groupby("src_split"):
    mm=mms[s]; rows=grp.src_row.values
    for st in range(0,len(rows),500):
        r=rows[st:st+500]
        blk=np.asarray(mm[r],dtype=np.float64)
        for bi,b in enumerate(("vv","vh")):
            z=blk[:,bi].ravel()
            acc[b][0]+=z.size; acc[b][1]+=z.sum(); acc[b][2]+=(z*z).sum()

print("SOURCE chips (this lineage's train subset), measured in the OLD z-convention,")
print("then converted to dB and re-expressed under the NEW constants:\n")
for b in ("vv","vh"):
    n,s,q=acc[b]
    m=s/n; sd_meas=float(np.sqrt(q/n-m*m))
    mo,so=OLD[b]; mn,sn=NEW[b]
    db_mean=m*so+mo                 # pixel-domain mean in dB
    db_std=sd_meas*so
    print(f"{b.upper()}:")
    print(f"  measured over PIXELS : mean={db_mean:+.4f} dB   std={db_std:.4f} dB")
    print(f"  constants (binned)   : mean={mn:+.4f} dB   std={sn:.4f} dB")
    print(f"  difference           : {db_mean-mn:+.4f} dB   std ratio {db_std/sn:.4f}")
    print(f"  => z under new consts: mean={(db_mean-mn)/sn:+.4f}   std={db_std/sn:.4f}\n")
print("If these match the built-memmap moments, the offset is inherent to the")
print("binned-vs-pixel estimator difference, NOT to the build.")
