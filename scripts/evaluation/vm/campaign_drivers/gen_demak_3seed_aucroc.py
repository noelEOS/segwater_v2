"""AUC-ROC scoring configs for the Demak no-aug/last gate, seeds s19/s42/s58."""
from pathlib import Path
import sys
RUNS = Path.home()/"segwater_v2/outputs/inference/runs"
OUT  = Path.home()/"configs/noaug_last_3seed"
TMPL = """\
# Demak concurrent gate scoring — Swin-B {seed}, NO augmentation, arm "last".
evaluation:
  name: "{name}"
  reference_dir: "/home/noel/ancillary/demak_semarang/reference_s2"
  reference_glob: "*.tif"
  s1_id_regex: "(S1_\\\\d{{8}}_\\\\d{{6}}_\\\\d+_\\\\d+_\\\\d+)"
  spatial_policy: "evaluate_geospatial_overlap"
  resolution_atol: 1.0e-12
  valid_mask_path: "/home/noel/ancillary/demak_semarang/valid_mask/GSHHG_GlobalSurfaceWater_combined_mask.tif"
  valid_mask_value: 1
  reference_water_values: [1]
  reference_nodata_values: [255]
  prediction:
    type: "probability_map"
    path_template: "{{run_dir}}/{{s1_id}}/{{s1_id}}_probability_water.tif"
  threshold_sweep:
    enabled: true
    start: 0.0
    stop: 1.0
    step: 0.01
    comparison: "greater_than"
    optimize_metrics: ["iou", "f1", "mcc"]
  model_runs:
    Upernet_Swin_Base_224_{seed}_noaug_last_native224_weighted:
      run_dir: "{run_dir}"
  missing_prediction_policy: "record_and_continue"
output:
  root: "/home/noel/demak_3seed_eval"
  run_name: "{name}"
  add_timestamp: false
"""
prob=[]
for seed in ["s19","s42","s58"]:
    pats=[f"demak_{seed}noaug_last_*", f"demak_{seed}noaug_last_*"]
    hits=[d for d in sorted(RUNS.glob(f"demak_{seed}noaug_last_*")) if d.is_dir()]
    if len(hits)!=1: prob.append(f"{seed}: {len(hits)} run dirs"); continue
    rd=hits[0]; n=len(list(rd.glob("*/*_probability_water.tif")))
    if n!=6: prob.append(f"{seed}: {n}/6 tifs"); continue
    name=f"demak_{seed}noaug_last_s32"
    (OUT/(name+".yaml")).write_text(TMPL.format(seed=seed,name=name,run_dir=str(rd)))
    print(f"  {name} <- {rd.name}")
if prob: print("PENDING/PROBLEM:", prob)
