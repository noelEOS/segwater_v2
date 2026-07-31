"""Do the 10pct memmaps contain the rows their indices claim?

The subset script and the strata npz were written by the same run, so they agree with each other
by construction. This checks them against the BYTES -- the one thing that can't be self-consistent
by accident.
"""
import numpy as np
R="/mnt/local_ssd/dataset_mixed80"
C,H,W=3,224,224
z=np.load("/home/noel/segwater_v2/experiments/DATASET_mixed80_blocked_612k/strata/subset_indices.npz")

for split,n_full,idxkey,n_sub in [("train",612364,"train_idx",61236),("val",109011,"val_idx",10901)]:
    idx=z[idxkey]
    full=np.memmap(f"{R}/{split}.memmap",dtype=np.float16,mode="r",shape=(n_full,C,H,W))
    sub=np.memmap(f"{R}/{split}_10pct.memmap",dtype=np.float16,mode="r",shape=(n_sub,C,H,W))
    assert len(idx)==n_sub, f"{split}: idx {len(idx)} != {n_sub}"
    rng=np.random.default_rng(7)
    probe=np.sort(rng.choice(n_sub,size=200,replace=False))
    bad=0; worst=0.0
    for j in probe:
        a=np.asarray(sub[j],dtype=np.float32)
        b=np.asarray(full[idx[j]],dtype=np.float32)
        if not np.array_equal(a,b):
            bad+=1; worst=max(worst,float(np.abs(a-b).max()))
    print(f"{split}_10pct: {len(probe)} probed, mismatches={bad}"
          + (f" (max diff {worst:.3e})" if bad else "  -> byte-identical OK"))
    # mask still {0,1,255}?
    mv=np.unique(np.asarray(sub[probe[:50],2],dtype=np.float32))
    print(f"   mask values in subset: {mv}")
