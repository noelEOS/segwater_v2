"""Demak FULL SERIES (213 scenes) for three arms on the mx630 dataset:
  mx630s2 best / mx630s2 last  (Swin-B, canonical 2-stage)
  mx630k        (ConvNeXtV2-Tiny, HPO trial_00006)
Memmaps off: the trend path reads *_probability_water.tif only."""
from pathlib import Path
import os, sys

OUT = Path.home() / "configs/demak_full_3lineage"
S2 = "outputs/mx630_stage2/s42"
MK = "outputs/mx630_sha_hpo/upernet_tu-convnextv2_tiny/01-46-52_s42/hpo_checkpoints/trial_00006"

ARMS = [
    ("mx630s2_best", f"{S2}/upernet_tu-swin_base_patch4_window7_224_s42_step12000_pmwiou0.890812.pth",
     "upernet", "tu-swin_base_patch4_window7_224",
     "mx630_stage2 lineage: Swin-B, canonical 2-stage on mx630. arm=best (pmwiou 0.890812, step 12000)"),
    ("mx630s2_last", f"{S2}/upernet_tu-swin_base_patch4_window7_224_s42_step23930_last.pth",
     "upernet", "tu-swin_base_patch4_window7_224",
     "mx630_stage2 lineage: Swin-B, canonical 2-stage on mx630. arm=last (step 23930)"),
    ("mx630k", f"{MK}/upernet_tu-convnextv2_tiny_s42_step23930_last.pth",
     "upernet", "tu-convnextv2_tiny",
     "mx630k lineage: ConvNeXtV2-TINY, HPO trial_00006 only (no full stage-2). arm=last"),
]

TMPL = """\
# Demak FULL SERIES (213 scenes) — {desc}
# seed s42. For the water-extent TREND.
# ⚠️ mx630 DATASET. Do not pool with pair-based results without the lineage tag.
# keep_probability_memmap: false -- trend path reads *_probability_water.tif only.

sweep:
  name: "demak_full_{tag}"
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
    - name: "{tag}"
      checkpoint_path: "{ckpt}"
      model:
        arch: "{arch}"
        encoder_name: "{enc}"

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
for tag, ckpt, arch, enc, desc in ARMS:
    full = Path.home() / "segwater_v2" / ckpt
    if not full.exists():
        sys.exit("MISSING: %s" % full)
    (OUT / ("demak_full_%s.yaml" % tag)).write_text(
        TMPL.format(tag=tag, ckpt=ckpt, arch=arch, enc=enc, desc=desc))
    print("  %-14s %-22s %s" % (tag, enc, os.path.basename(ckpt)))
print("wrote %d configs" % len(ARMS))
