"""mx630_stage2 (Swin-B, canonical 2-stage on the mx630 dataset), s42, best+last.
Three gates: hampyeong, narrabeen (SDS), demak concurrent."""
from pathlib import Path
import glob, os, sys, torch

D = Path("outputs/mx630_stage2/s42")
OUT = Path.home() / "configs/mx630s2"

HEAD = """\
# {SITE} — mx630_stage2 lineage: Swin-B, canonical 2-stage (HPO + full stage)
# on the mx630 dataset. s42, arm "{arm}"{extra}.
# ⚠️ Lineage tag: mx630_stage2. Same DATASET as the mx630k ConvNeXtV2-Tiny run,
#    but different architecture (Swin-B) and different training procedure.
#    Do not pool with pair-based Swin-B or with mx630k without the tag.
"""

BODY = """
sweep:
  name: "{site}_mx630s2_s42_{arm}"
  dry_run: false
  continue_on_error: true

  input_dir: "{indir}"
  input_glob: "{glob}"

  common_overrides:
    inference.post_processing.smoothing.apply_simplification: true
    inference.post_processing.smoothing.simplify_tolerance_meters: 1.0
{filt}    inference.data.edge_policy: "shift_inward"
    inference.stitching.min_weight: 0.001

  checkpoints:
    - name: "upernet_swin_base_224_mx630s2_{arm}"
      checkpoint_path: "{ckpt}"
      model:
        arch: "upernet"
        encoder_name: "tu-swin_base_patch4_window7_224"

  presets:
{presets}"""

PRESET = """    - name: "native224_weighted_224_b0_s{st}"
      overrides:
        inference.data.tile_size: 224
        inference.data.buffer_size: 0
        inference.data.stride: {st}
        inference.stitching.mode: "weighted_blend"
        inference.stitching.blend_window: "hann"
"""

# per-site: input dir, glob, filtering block, strides
SITES = {
    "hampyeong": ("/home/noel/Inference_input/hampyeong_ron_134_ts_16_sn_15", "S1B_*.tif",
                  "    inference.post_processing.filtering.apply_length_filter: false\n"
                  "    inference.post_processing.filtering.min_length_meters: 10000.0\n"
                  "    inference.post_processing.filtering.keep_top_k: 5\n", [32]),
    "narrabeen": ("/home/noel/NARRABEEN_ron_147_ts_9_sn_16", "*.tif",
                  "    inference.post_processing.filtering.apply_length_filter: false\n"
                  "    inference.post_processing.filtering.min_length_meters: 10000.0\n"
                  "    inference.post_processing.filtering.keep_top_k: 999\n", [112, 32, 8]),
    "demak_gate": ("/home/noel/data_demak_concurrent", "S1_*.tif",
                   "    inference.post_processing.filtering.apply_length_filter: true\n"
                   "    inference.post_processing.filtering.min_length_meters: 10000.0\n", [32]),
}

# resolve arms from checkpoint METADATA, not filenames
cands = []
for p in glob.glob(str(D / "*.pth")):
    n = os.path.basename(p)
    if "_snap_" in n or n == "best.pth":
        continue
    ck = torch.load(p, map_location="cpu", weights_only=True)
    cands.append((ck.get("pair_macro_water_iou"), int(ck.get("step", -1)), p))
best = max([c for c in cands if c[0] is not None], key=lambda c: (c[0], -c[1]))
last = [c for c in cands if c[0] is None and c[2].endswith("_last.pth")]
if not last:
    sys.exit("no _last.pth found")
ARMS = {"best": (best[2], " (pmwiou=%.8f, step %d)" % (best[0], best[1])),
        "last": (last[0][2], " (final step %d)" % last[0][1])}

OUT.mkdir(parents=True, exist_ok=True)
for site, (indir, gl, filt, strides) in SITES.items():
    for arm, (ck, extra) in ARMS.items():
        presets = "".join(PRESET.format(st=s) + ("\n" if s != strides[-1] else "") for s in strides)
        txt = HEAD.format(SITE=site.upper(), arm=arm, extra=extra) + BODY.format(
            site=site, arm=arm, indir=indir, glob=gl, filt=filt, ckpt=ck, presets=presets)
        (OUT / ("%s_mx630s2_s42_%s.yaml" % (site, arm))).write_text(txt)
print("arms:")
for a, (ck, ex) in ARMS.items():
    print("  %-5s %s%s" % (a, os.path.basename(ck), ex))
print("wrote %d configs" % (len(SITES) * len(ARMS)))
