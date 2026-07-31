"""Does the SHIPPED pair-based train memmap carry the same binned-vs-pixel offset?
If yes, this is a property of the recipe, present in every lineage -- not new to mixed80."""
import numpy as np
SRC="/mnt/local_ssd/dataset/train.memmap"
N,C,H,W=1032004,3,224,224
OLD={"vv":(-15.510920639155525,6.564080466988128),"vh":(-23.573757647507851,7.6567358472184)}
rng=np.random.default_rng(42)
idx=np.sort(rng.choice(N,size=3000,replace=False))
mm=np.memmap(SRC,dtype=np.float32,mode="r",shape=(N,C,H,W))
print("SHIPPED pair-based train.memmap, moments in its OWN z-convention")
print("(if the constants described these pixels exactly, these would be 0.0 / 1.0):\n")
for bi,b in enumerate(("vv","vh")):
    n=s=q=0.0
    for st in range(0,len(idx),500):
        z=np.asarray(mm[idx[st:st+500],bi],dtype=np.float64).ravel()
        n+=z.size; s+=z.sum(); q+=(z*z).sum()
    m=s/n; sd=float(np.sqrt(q/n-m*m))
    mo,so=OLD[b]
    print(f"  {b.upper()}: mean={m:+.4f}  std={sd:.4f}   (in dB: {m*so+mo:+.4f} +/- {sd*so:.4f})")
print("\n=> Compare with mixed80: VV -0.0169/1.0272, VH -0.1432/1.0980")
