"""Trend-proxy (17-scene) inference configs: aug/no_aug x best/last, Swin-B s42."""
from pathlib import Path
import sys

BASE = Path("outputs/stage2/upernet_tu-swin_base_patch4_window7_224")
OUT = Path.home() / "configs/demak_proxy17"
ARMS = {
    ("aug", "best"):    BASE / "s42"        / "upernet_tu-swin_base_patch4_window7_224_s42_step24000_pmwiou0.872166.pth",
    ("aug", "last"):    BASE / "s42"        / "upernet_tu-swin_base_patch4_window7_224_s42_step40320_last.pth",
    ("no_aug", "best"): BASE / "s42_no_aug" / "upernet_tu-swin_base_patch4_window7_224_s42_step9600_pmwiou0.869898.pth",
    ("no_aug", "last"): BASE / "s42_no_aug" / "upernet_tu-swin_base_patch4_window7_224_s42_step40320_last.pth",
}

# Mirrors the demak gate configs; the trend proxy uses the SAME native224-s32
# preset and post-processing so areas stay comparable to the gate rasters.
TEMPLATE = """\
# Demak trend-proxy (17 scenes) — Swin-B s42, {aug}, arm "{arm}".
# hpo_objective = pair_macro_water_iou_mixed.
# Same preset/post-processing as the concurrent gate so probabilities are comparable.

sweep:
  name: "demak_proxy17_{aug}_{arm}"
  dry_run: false
  continue_on_error: true

  input_dir: "/home/noel/data_demak_proxy17"
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
    - name: "upernet_swin_base_224_{aug}_{arm}"
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
for (aug, arm), ck in ARMS.items():
    if not ck.exists():
        sys.exit("MISSING: %s" % ck)
    p = OUT / ("demak_proxy17_%s_%s.yaml" % (aug, arm))
    p.write_text(TEMPLATE.format(aug=aug, arm=arm, ckpt=str(ck)))
    print("%-34s -> %s" % (p.name, ck.name))
