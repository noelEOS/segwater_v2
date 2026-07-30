"""DEPRECATED (2026-07-30) — superseded by ``ship/gen_ship_configs.py``.

Kept as the **executable record** of the tracked rest-4 sweep configs it produced
(`configs/aucroc/*rest4*.yaml`). Do not use it to generate NEW configs; regenerate
from `ship/gen_ship_configs.py`, which is the one hardened generator.

What this file lacks that the ship generator has:
  * **String-concatenated YAML emission.** The config is built by appending
    ~40 `lines.append("  key: value")` strings and joining them — nothing
    validates that the result parses, and an indentation typo produces a
    silently different config rather than an error. The ship generator emits
    from a single format template checked against the tracked configs.
  * **No prefix-collision check on the emitted sweep names.** A generated name
    that is a prefix of another (`…_mx630k` vs `…_mx630k_best`) makes every
    later run-dir lookup ambiguous — the bug class runsel exists to catch, but
    at generation time it must be `naming.require_no_prefix_collisions`
    (which `gen_ship_configs.py` calls; this file has no equivalent).
  * **No atomic write.** Configs are written with a plain `write_text`, so an
    interrupted run leaves a truncated YAML that looks like a real config.

It DOES already resolve run dirs through `runsel.resolve_run_dir` with a
`suffix=` pin (keeping `unet_resnet50` from matching `unetplusplus_resnet50`),
which is why it was left functional rather than deleted.
"""
import glob, os, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from runsel import resolve_run_dir

RUNS = os.path.expanduser("~/segwater_v2/outputs/inference/runs")
# arch dir-token -> evaluator model key (matches chip-trained eval keys)
ARCH = [
    ("unet_resnet50",          "Unet_Resnet50_native224_weighted"),
    ("unetplusplus_resnet50",  "UnetPlusPlus_Resnet50_native224_weighted"),
    ("deeplabv3plus_resnet50", "DeepLabV3plus_Resnet50_native224_weighted"),
    ("dpt_vit_b_16",           "DPT_ViT_B_16_native224_weighted"),
]
for seed in ["s19", "s42", "s58"]:
    runs = {}
    for token, key in ARCH:
        # `suffix` keeps unet_resnet50 from matching unetplusplus_resnet50;
        # the anchored name keeps a longer sweep name from being absorbed.
        runs[key] = str(resolve_run_dir(
            RUNS, f"dev_sweep_concurrent_demak_{seed}",
            suffix=f"_{token}_native224_weighted_224_b0_s32"))
    lines = []
    lines.append(f"# Pair-based rest-4 archs, seed {seed}, stride 32.")
    lines.append(f"# Mirrors ~/aucroc_{seed}_vm.yaml; only model_runs and run_name differ.")
    lines.append("")
    lines.append("evaluation:")
    lines.append(f"  name: \"semarang_probability_aucroc_{seed}_s32_pairbased_rest4\"")
    lines.append("")
    lines.append("  reference_dir: \"/home/noel/ancillary/demak_semarang/reference_s2\"")
    lines.append("  reference_glob: \"*.tif\"")
    lines.append("  s1_id_regex: \"(S1_\\\\d{8}_\\\\d{6}_\\\\d+_\\\\d+_\\\\d+)\"")
    lines.append("")
    lines.append("  spatial_policy: \"evaluate_geospatial_overlap\"")
    lines.append("  resolution_atol: 1.0e-12")
    lines.append("")
    lines.append("  valid_mask_path: \"/home/noel/ancillary/demak_semarang/valid_mask/GSHHG_GlobalSurfaceWater_combined_mask.tif\"")
    lines.append("  valid_mask_value: 1")
    lines.append("")
    lines.append("  reference_water_values: [1]")
    lines.append("  reference_nodata_values: [255]")
    lines.append("")
    lines.append("  prediction:")
    lines.append("    type: \"probability_map\"")
    lines.append("    path_template: \"{run_dir}/{s1_id}/{s1_id}_probability_water.tif\"")
    lines.append("")
    lines.append("  threshold_sweep:")
    lines.append("    enabled: true")
    lines.append("    start: 0.0")
    lines.append("    stop: 1.0")
    lines.append("    step: 0.01")
    lines.append("    comparison: \"greater_than\"")
    lines.append("    optimize_metrics: [\"iou\", \"f1\", \"mcc\"]")
    lines.append("")
    lines.append("  model_runs:")
    for key, rd in runs.items():
        lines.append(f"    {key}:")
        lines.append(f"      run_dir: \"{rd}\"")
    lines.append("")
    lines.append("  missing_prediction_policy: \"record_and_continue\"")
    lines.append("")
    lines.append("output:")
    lines.append("  root: \"/home/noel/pairbased_vm_eval\"")
    lines.append(f"  run_name: \"semarang_probability_aucroc_{seed}_s32_pairbased_rest4\"")
    lines.append("  add_timestamp: false")
    out = os.path.expanduser(f"~/aucroc_rest4_{seed}_vm.yaml")
    with open(out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("wrote", out)
