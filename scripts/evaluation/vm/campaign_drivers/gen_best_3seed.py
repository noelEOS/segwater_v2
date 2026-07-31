"""Demak FULL SERIES + concurrent gate, no-aug BEST ckpt, seeds s19/s42/s58.
`best` = highest pair_macro_water_iou among that seed's pmwiou checkpoints,
read from the checkpoint metadata (not the filename)."""
from pathlib import Path
import glob, os, sys, torch

B = Path("outputs/stage2/upernet_tu-swin_base_patch4_window7_224")
OUT = Path.home() / "configs/demak_best_3seed"
SEEDS = ["s19", "s42", "s58"]

FULL = """\
# Demak FULL SERIES (213 scenes) — Swin-B {seed}, NO augmentation, arm "best".
# best = top pair-macro water IoU (pmwiou={pm:.8f}, step {step}).
# hpo_objective = pair_macro_water_iou_mixed.
# keep_probability_memmap: false -- trend path reads *_probability_water.tif only.

sweep:
  name: "demak_full_{seed}noaug_best"
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
    - name: "upernet_swin_base_224_noaug_best"
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

GATE = FULL.replace('name: "demak_full_{seed}noaug_best"', 'name: "demak_{seed}noaug_best"') \
           .replace('# Demak FULL SERIES (213 scenes)', '# Demak CONCURRENT GATE (6 pairs)') \
           .replace('input_dir: "/home/noel/data_demak"', 'input_dir: "/home/noel/data_demak_concurrent"') \
           .replace("    inference.output.keep_probability_memmap: false\n", "")

OUT.mkdir(parents=True, exist_ok=True)
for seed in SEEDS:
    cands = []
    for p in glob.glob(str(B / seed / "*.pth")):
        n = os.path.basename(p)
        if "_snap_" in n or n == "best.pth":
            continue
        ck = torch.load(p, map_location="cpu", weights_only=True)
        pm = ck.get("pair_macro_water_iou")
        if pm is not None:
            cands.append((float(pm), int(ck.get("step", -1)), p))
    if not cands:
        sys.exit("seed %s: no pmwiou checkpoints" % seed)
    cands.sort(key=lambda r: (-r[0], r[1]))
    pm, step, ckpt = cands[0]
    for tmpl, tag in [(FULL, "full"), (GATE, "gate")]:
        f = OUT / ("demak_%s_%snoaug_best.yaml" % (tag, seed))
        f.write_text(tmpl.format(seed=seed, ckpt=ckpt, pm=pm, step=step))
    print("%-5s best pmwiou=%.8f step=%-6s %s" % (seed, pm, step, os.path.basename(ckpt)))
print("wrote %d configs" % (2 * len(SEEDS)))
