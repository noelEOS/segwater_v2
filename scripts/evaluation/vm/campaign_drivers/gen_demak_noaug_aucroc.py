"""AUC-ROC scoring configs for the Demak s42-aug gate (best/last)."""
from pathlib import Path
import sys

RUNS = Path.home() / "segwater_v2/outputs/inference/runs"
OUT = Path.home() / "configs/demak_gate_noaug"

TEMPLATE = """\
# Demak concurrent gate scoring — Swin-B s42 AUGMENTED, arm "{arm}".
# Same estimator as every registered accuracy table: S2 vote-and-veto reference,
# GSHHG+GSW valid mask, threshold sweep 0->1 step 0.01, greater_than.

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
    Upernet_Swin_Base_224_s42noaug_{arm}_native224_weighted:
      run_dir: "{run_dir}"

  missing_prediction_policy: "record_and_continue"

output:
  root: "/home/noel/demak_gate_noaug_eval"
  run_name: "{name}"
  add_timestamp: false
"""

problems = []
for arm in ["best", "last"]:
    hits = sorted(d for d in RUNS.glob("demak_s42noaug_%s_*" % arm) if d.is_dir())
    if len(hits) != 1:
        problems.append("%s: %d run dirs" % (arm, len(hits))); continue
    rd = hits[0]
    n = len(list(rd.glob("*/*_probability_water.tif")))
    if n != 6:
        problems.append("%s: %d/6 tifs" % (arm, n)); continue
    name = "demak_s42noaug_%s_s32" % arm
    (OUT / (name + ".yaml")).write_text(TEMPLATE.format(arm=arm, name=name, run_dir=str(rd)))
    print("%-28s <- %s" % (name, rd.name))
if problems:
    print("PROBLEMS:", problems); sys.exit(1)
