"""Hampyeong Bay inference, no-aug/last, seeds s19/s42/s58.
Mirrors the tracked hampyeong sweep config exactly (length filter OFF,
keep_top_k 5, native224-s32); only sweep.name + checkpoint differ."""
from pathlib import Path
import sys

B = Path("outputs/stage2/upernet_tu-swin_base_patch4_window7_224")
OUT = Path.home() / "configs/hampyeong_3seed"
SEEDS = ["s19", "s42", "s58"]

TMPL = """\
# Hampyeong Bay — Swin-B {seed}, NO augmentation, arm "last".
# hpo_objective = pair_macro_water_iou_mixed. seed dir `{seed}` = no-aug (canonical).
# Settings mirror the tracked dev_sweep_all_hampyeong configs: length filter OFF,
# keep_top_k 5, native224-s32.

sweep:
  name: "hampyeong_{seed}noaug_last"
  dry_run: false
  continue_on_error: true

  input_dir: "/home/noel/Inference_input/hampyeong_ron_134_ts_16_sn_15"
  input_glob: "S1B_*.tif"

  common_overrides:
    inference.post_processing.smoothing.apply_simplification: true
    inference.post_processing.smoothing.simplify_tolerance_meters: 1.0
    inference.post_processing.filtering.apply_length_filter: false
    inference.post_processing.filtering.min_length_meters: 10000.0
    inference.post_processing.filtering.keep_top_k: 5
    inference.data.edge_policy: "shift_inward"
    inference.stitching.min_weight: 0.001

  checkpoints:
    - name: "upernet_swin_base_224_noaug_last"
      checkpoint_path: "{ckpt}"
      model:
        arch: "upernet"
        encoder_name: "tu-swin_base_patch4_window7_224"

  presets:
    - name: "native224_weighted_224_b0_s32"
      overrides:
        inference.data.tile_size: 224
        inference.data.buffer_size: 0
        inference.data.stride: 32
        inference.stitching.mode: "weighted_blend"
        inference.stitching.blend_window: "hann"
"""

OUT.mkdir(parents=True, exist_ok=True)
for seed in SEEDS:
    ck = sorted((B / seed).glob("*_last.pth"))
    if len(ck) != 1:
        sys.exit("seed %s: %d _last.pth" % (seed, len(ck)))
    p = OUT / ("hampyeong_%snoaug_last.yaml" % seed)
    p.write_text(TMPL.format(seed=seed, ckpt=str(ck[0])))
    print("%-36s -> %s/%s" % (p.name, ck[0].parent.name, ck[0].name))
