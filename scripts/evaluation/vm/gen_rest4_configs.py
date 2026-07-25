import glob, os
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
        pat = os.path.join(RUNS, f"dev_sweep_concurrent_demak_{seed}_*_{token}_native224_weighted_224_b0_s32")
        hits = sorted(glob.glob(pat))
        assert len(hits) == 1, f"{seed} {token}: {hits}"
        runs[key] = hits[0]
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
