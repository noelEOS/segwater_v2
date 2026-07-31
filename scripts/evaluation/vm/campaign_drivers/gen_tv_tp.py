"""Trucvert + Torreypines SDS inference configs: aug/no_aug x best/last, Swin-B s42."""
from pathlib import Path
import sys

B = Path("outputs/stage2/upernet_tu-swin_base_patch4_window7_224")
OUT = Path.home() / "configs/tv_tp_sds"
# canonical naming: s42 = NO-AUG, s42_aug = augmented
ARMS = {
    ("aug", "best"):    B/"s42_aug"/"upernet_tu-swin_base_patch4_window7_224_s42_step24000_pmwiou0.872166.pth",
    ("aug", "last"):    B/"s42_aug"/"upernet_tu-swin_base_patch4_window7_224_s42_step40320_last.pth",
    ("no_aug", "best"): B/"s42"    /"upernet_tu-swin_base_patch4_window7_224_s42_step9600_pmwiou0.869898.pth",
    ("no_aug", "last"): B/"s42"    /"upernet_tu-swin_base_patch4_window7_224_s42_step40320_last.pth",
}
SITES = {
    "trucvert":    "/home/noel/Inference_input/TRUCVERT_ron_8_ts_30_sn_20",
    "torreypines": "/home/noel/Inference_input/Torreypines_ron_71",
}

TEMPLATE = """\
# {SITE} SDS — Swin-B s42, {aug}, arm "{arm}". Canonical frame.
# hpo_objective = pair_macro_water_iou_mixed. seed dir: s42 = NO-AUG, s42_aug = augmented.
# Scenes staged by the GT-calendar-window rule (docs/RUNBOOK_sds_vm_eval.md).

sweep:
  name: "{site}_s42{augtag}_{arm}"
  dry_run: false
  continue_on_error: true

  input_dir: "{indir}"
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
    - name: "upernet_swin_base_224_{augtag}_{arm}"
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
n = 0
for site, indir in SITES.items():
    for (aug, arm), ck in ARMS.items():
        if not ck.exists():
            sys.exit("MISSING checkpoint: %s" % ck)
        tag = "noaug" if aug == "no_aug" else "aug"
        p = OUT / ("%s_s42%s_%s.yaml" % (site, tag, arm))
        p.write_text(TEMPLATE.format(SITE=site.upper(), site=site, aug=aug, arm=arm,
                                     augtag=tag, indir=indir, ckpt=str(ck)))
        print("%-34s -> %s/%s" % (p.name, ck.parent.name, ck.name.split("_")[-1]))
        n += 1
print("wrote %d configs" % n)
