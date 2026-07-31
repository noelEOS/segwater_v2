"""Demak concurrent accuracy-gate configs, Swin-B s42 AUGMENTED, best vs last."""
from pathlib import Path
import sys

STAGE2 = Path("outputs/stage2/upernet_tu-swin_base_patch4_window7_224/s42")
OUT = Path.home() / "configs/demak_gate_aug"
ARMS = {"best": "upernet_tu-swin_base_patch4_window7_224_s42_step24000_pmwiou0.872166.pth",
        "last": "upernet_tu-swin_base_patch4_window7_224_s42_step40320_last.pth"}

TEMPLATE = """\
# Demak concurrent accuracy gate — Swin-B s42 (AUGMENTED).
# hpo_objective = pair_macro_water_iou_mixed (NOT the canonical pooled-mIoU HPO).
# Arm "{arm}". Preset native224_weighted_224_b0_s32 = the canonical comparison baseline.

sweep:
  name: "demak_s42aug_{arm}"
  dry_run: false
  continue_on_error: true

  input_dir: "/home/noel/data_demak_concurrent"
  input_glob: "S1_*.tif"

  common_overrides:
    inference.post_processing.smoothing.apply_simplification: true
    inference.post_processing.smoothing.simplify_tolerance_meters: 1.0
    inference.post_processing.filtering.apply_length_filter: true
    inference.post_processing.filtering.min_length_meters: 10000.0
    inference.data.edge_policy: "shift_inward"
    inference.stitching.min_weight: 0.001
    inference.data.num_workers: 8
    inference.data.batch_size: 256
    inference.output.probability_precision: "float16"

  checkpoints:
    - name: "upernet_swin_base_224_{arm}"
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
for arm, fn in ARMS.items():
    p = STAGE2 / fn
    if not p.exists():
        sys.exit("MISSING checkpoint: %s" % p)
    cfg = OUT / ("demak_s42aug_%s.yaml" % arm)
    cfg.write_text(TEMPLATE.format(arm=arm, ckpt=str(STAGE2 / fn)))
    print("%-28s -> %s" % (cfg.name, fn))
