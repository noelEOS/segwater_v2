"""Emit the Hampyeong scoring spec: 3 seeds, no-aug/last."""
from pathlib import Path
import sys
B=Path("outputs/stage2/upernet_tu-swin_base_patch4_window7_224")
OUT=Path.home()/"configs/hampyeong_3seed/spec_hampyeong_noaug_last_3seed.yaml"
HEAD="""\
# Hampyeong Bay — Swin-B no-aug / last ckpt, 3 seeds (s19/s42/s58).
# The scorer\x27s distinctness + provenance audit enforces that all 3 really are
# different checkpoints.
repo: /home/noel/segwater_v2
nas_root: /home/noel/ancillary/hampyeong/nas_root
pair_root: outputs/inference
expected_valid_pixels: 1179967
variant_column: false
output: /home/noel/hampyeong_3seed_noaug_last_per_date_metrics.csv

entries:
"""
ENTRY="""\
  - label: Swin-B
    slug: swin_b
    seed: {seedn}
    run_dir: "glob:runs/hampyeong_{seed}noaug_last_*_upernet_swin_base_224_noaug_last_native224_weighted_224_b0_s32"
    checkpoint: {ckpt}
"""
parts=[HEAD]
for seed in ["s19","s42","s58"]:
    ck=sorted((B/seed).glob("*_last.pth"))
    if len(ck)!=1: sys.exit("seed %s: %d ckpt"%(seed,len(ck)))
    parts.append(ENTRY.format(seedn=seed[1:],seed=seed,ckpt=str(ck[0])))
OUT.parent.mkdir(parents=True,exist_ok=True)
OUT.write_text("".join(parts))
print("wrote",OUT)
print("".join(parts))
