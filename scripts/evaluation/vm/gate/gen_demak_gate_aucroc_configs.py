"""Generate AUC-ROC scoring configs, one per Demak gate inference run.

Discovers run dirs by the `gate_{arm}_demak_{seed}_...` prefix so it stays in
sync with whatever the inference batch actually produced. Fails loudly if an
expected arm is missing or ambiguous rather than silently scoring 8 of 9.
"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from runsel import resolve_run_dirs  # noqa: E402

RUNS = Path("/home/noel/segwater_v2/outputs/inference/runs")
OUT = Path("/home/noel/configs/aucroc_gate")
SEEDS = ["s19", "s42", "s58"]
ARMS = ["best", "last", "swa5"]

TEMPLATE = """\
# Demak concurrent gate scoring — Swin-B stage2 seed {seed}, arm "{arm}".
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
    Upernet_Swin_Base_224_{arm}_native224_weighted:
      run_dir: "{run_dir}"

  missing_prediction_policy: "record_and_continue"

output:
  root: "/home/noel/pairbased_vm_eval"
  run_name: "{name}"
  add_timestamp: false
"""


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    written, problems = [], []
    for seed in SEEDS:
        for arm in ARMS:
            # Anchored on the UTC stamp: a bare prefix glob would also match a
            # sweep whose name extends this one (e.g. an `..._s42_PERF` re-run).
            hits = resolve_run_dirs(RUNS, "gate_%s_demak_%s" % (arm, seed))
            if len(hits) != 1:
                problems.append("%s/%s: %d run dirs%s" % (
                    seed, arm, len(hits),
                    "".join("\n      " + h.name for h in hits)))
                continue
            run_dir = hits[0]
            n_tif = len(list(run_dir.glob("*/*_probability_water.tif")))
            if n_tif != 6:
                problems.append("%s/%s: %d/6 probability tifs" % (seed, arm, n_tif))
                continue
            name = "demak_gate_%s_%s_s32" % (arm, seed)
            (OUT / (name + ".yaml")).write_text(
                TEMPLATE.format(seed=seed, arm=arm, name=name, run_dir=str(run_dir)))
            written.append((name, run_dir.name))
    print("=== WROTE %d scoring configs ===" % len(written))
    for name, rd in written:
        print("  %-28s <- %s" % (name, rd))
    if problems:
        print("=== PROBLEMS ===")
        for p in problems:
            print("  !! " + p)
        sys.exit(1)


if __name__ == "__main__":
    main()
