"""Duck SDS inference configs: aug/no_aug x best/last, Swin-B s42, 79 scorable scenes."""
from pathlib import Path
import sys

BASE = Path("outputs/stage2/upernet_tu-swin_base_patch4_window7_224")
OUT = Path.home() / "configs/duck_sds"
ARMS = {
    ("aug", "best"):    BASE/"s42"       /"upernet_tu-swin_base_patch4_window7_224_s42_step24000_pmwiou0.872166.pth",
    ("aug", "last"):    BASE/"s42"       /"upernet_tu-swin_base_patch4_window7_224_s42_step40320_last.pth",
    ("no_aug", "best"): BASE/"s42_no_aug"/"upernet_tu-swin_base_patch4_window7_224_s42_step9600_pmwiou0.869898.pth",
    ("no_aug", "last"): BASE/"s42_no_aug"/"upernet_tu-swin_base_patch4_window7_224_s42_step40320_last.pth",
}

# Mirrors the Narrabeen SDS configs: length filter OFF (SDS needs the full
# shoreline, not the longest segment), native224-s32/s8/s112 sweep.
TEMPLATE = """\
# Duck SDS — Swin-B s42, {aug}, arm "{arm}". Frame ron_4_ts_24_sn_19 (primary/reported, ASC).
# hpo_objective = pair_macro_water_iou_mixed.
# 79 scorable scenes (30 with no in-window groundtruth parked in ~/DUCK_ron4_no_groundtruth).

sweep:
  name: "duck_s42{augtag}_{arm}"
  dry_run: false
  continue_on_error: true

  input_dir: "/home/noel/Inference_input/Duck_ron_4_ts_24_sn_19"
  input_glob: "*.tif"

  common_overrides:
    inference.post_processing.smoothing.apply_simplification: true
    inference.post_processing.smoothing.simplify_tolerance_meters: 1.0
    inference.post_processing.filtering.apply_length_filter: false
    inference.post_processing.filtering.min_length_meters: 10000.0
    inference.post_processing.filtering.keep_top_k: 999
    inference.data.edge_policy: "shift_inward"
    inference.stitching.min_weight: 0.001

  checkpoints:
    - name: "upernet_swin_base_224_{augtag2}{arm}"
      checkpoint_path: "{ckpt}"
      model:
        arch: "upernet"
        encoder_name: "tu-swin_base_patch4_window7_224"

  presets:
    - name: "native224_weighted_224_b0_s112"
      overrides:
        inference.data.tile_size: 224
        inference.data.buffer_size: 0
        inference.data.stride: 112
        inference.stitching.mode: "weighted_blend"
        inference.stitching.blend_window: "hann"

    - name: "native224_weighted_224_b0_s32"
      overrides:
        inference.data.tile_size: 224
        inference.data.buffer_size: 0
        inference.data.stride: 32
        inference.stitching.mode: "weighted_blend"
        inference.stitching.blend_window: "hann"

    - name: "native224_weighted_224_b0_s8"
      overrides:
        inference.data.tile_size: 224
        inference.data.buffer_size: 0
        inference.data.stride: 8
        inference.stitching.mode: "weighted_blend"
        inference.stitching.blend_window: "hann"
"""

OUT.mkdir(parents=True, exist_ok=True)
for (aug, arm), ck in ARMS.items():
    if not ck.exists():
        sys.exit("MISSING: %s" % ck)
    tag = "noaug" if aug == "no_aug" else "aug"
    p = OUT / ("duck_s42%s_%s.yaml" % (tag, arm))
    p.write_text(TEMPLATE.format(aug=aug, arm=arm, augtag=tag, augtag2=tag + "_", ckpt=str(ck)))
    print("%-28s -> %s" % (p.name, ck.name))
