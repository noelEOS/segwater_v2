"""Demak FULL SERIES (213 scenes) inference, no-aug/last, seeds s19/s42/s58.
For the trend calculation. Memmaps disabled: the trend path reads only
*_probability_water.tif (sem_core.read_prob uses rasterio), so the memmap is
dead weight -- 213 scenes x 3 seeds of it."""
from pathlib import Path
import sys

B = Path("outputs/stage2/upernet_tu-swin_base_patch4_window7_224")
OUT = Path.home() / "configs/demak_full_3seed"
SEEDS = ["s19", "s42", "s58"]

TMPL = """\
# Demak FULL SERIES (213 scenes) — Swin-B {seed}, NO augmentation, arm "last".
# For the water-extent TREND. hpo_objective = pair_macro_water_iou_mixed.
# Registered analysis window is 206 scenes (213 minus 7 past 2024-12-31); the
# window is applied downstream, so all 213 are inferred here.
#
# keep_probability_memmap: false -- the trend path reads *_probability_water.tif
# only (sem_core.read_prob -> rasterio). Saves ~213 scenes x 3 seeds of memmap.

sweep:
  name: "demak_full_{seed}noaug_last"
  dry_run: false
  continue_on_error: true

  input_dir: "/home/noel/data_demak"
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
    inference.output.keep_probability_memmap: false

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
    p = OUT / ("demak_full_%snoaug_last.yaml" % seed)
    p.write_text(TMPL.format(seed=seed, ckpt=str(ck[0])))
    print("%-38s -> %s" % (p.name, ck[0].name))
