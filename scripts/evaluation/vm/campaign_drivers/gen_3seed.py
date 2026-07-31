"""No-aug / last-ckpt configs for seeds s19+s58: 4 SDS sites + Demak concurrent gate.
s42 already exists and is reused, so only these two seeds are generated."""
from pathlib import Path
import sys

B = Path("outputs/stage2/upernet_tu-swin_base_patch4_window7_224")
OUT = Path.home() / "configs/noaug_last_3seed"
SEEDS = ["s19", "s58"]

SDS_SITES = {  # site -> input dir
    "narrabeen":   "/home/noel/NARRABEEN_ron_147_ts_9_sn_16",
    "duck":        "/home/noel/Inference_input/Duck_ron_4_ts_24_sn_19",
    "trucvert":    "/home/noel/Inference_input/TRUCVERT_ron_8_ts_30_sn_20",
    "torreypines": "/home/noel/Inference_input/Torreypines_ron_71",
}

SDS_TMPL = """\
# {SITE} SDS — Swin-B {seed}, NO augmentation, arm "last". Canonical frame.
# hpo_objective = pair_macro_water_iou_mixed. seed dir `{seed}` = no-aug (canonical naming).

sweep:
  name: "{site}_{seed}noaug_last"
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
    - name: "upernet_swin_base_224_noaug_last"
      checkpoint_path: "{ckpt}"
      model:
        arch: "upernet"
        encoder_name: "tu-swin_base_patch4_window7_224"

  presets:
{presets}
"""

PRESET = """\
    - name: "native224_weighted_224_b0_s{st}"
      overrides:
        inference.data.tile_size: 224
        inference.data.buffer_size: 0
        inference.data.stride: {st}
        inference.stitching.mode: "weighted_blend"
        inference.stitching.blend_window: "hann"
"""

DEMAK_TMPL = """\
# Demak concurrent accuracy gate — Swin-B {seed}, NO augmentation, arm "last".
# hpo_objective = pair_macro_water_iou_mixed. Preset native224_weighted_224_b0_s32.

sweep:
  name: "demak_{seed}noaug_last"
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
presets = "".join(PRESET.format(st=s) + "\n" for s in (112, 32, 8)).rstrip() + "\n"
n = 0
for seed in SEEDS:
    ck = sorted((B / seed).glob("*_last.pth"))
    if len(ck) != 1:
        sys.exit("seed %s: %d _last.pth" % (seed, len(ck)))
    ck = ck[0]
    for site, indir in SDS_SITES.items():
        p = OUT / ("%s_%snoaug_last.yaml" % (site, seed))
        p.write_text(SDS_TMPL.format(SITE=site.upper(), site=site, seed=seed,
                                     indir=indir, ckpt=str(ck), presets=presets))
        n += 1
    p = OUT / ("demak_%snoaug_last.yaml" % seed)
    p.write_text(DEMAK_TMPL.format(seed=seed, ckpt=str(ck)))
    n += 1
    print("%s -> %s" % (seed, ck.name))
print("wrote %d configs (expect 10)" % n)
