# Checkpoint-selection gate — config generators

Generators that turn a set of stage-2 seed dirs into the per-arm inference and
scoring configs for the four-way gate in
`docs/RUNBOOK_checkpoint_selection_gate.md`. Written for the 2026-07-25
three-seed Swin-B run (best / last / SWA-5); reusable for any arch.

All four **resolve checkpoints by glob against the real seed dir** rather than
hardcoding filenames, and refuse to emit a config for an arm whose checkpoint is
missing or ambiguous — so a missing SWA build fails loudly instead of silently
producing an 8-of-9 comparison.

| Script | Emits | Where |
|---|---|---|
| `gen_demak_gate_configs.py` | `run_inference_sweep.py` configs, one per arm×seed | `~/configs/demak_gate/` |
| `gen_demak_gate_aucroc_configs.py` | AUC-ROC scoring configs, discovered from the finished run dirs (checks 6/6 probability tifs first) | `~/configs/aucroc_gate/` |
| `gen_hampyeong_gate_configs.py` | Hampyeong sweep configs (length filter OFF, `keep_top_k` 5) | `~/configs/hampyeong_gate/` |
| `gen_hampyeong_gate_spec.py` | one 9-entry spec for `score_pairbased_hampyeong.py` | `~/configs/hampyeong_gate/spec_hampyeong_gate.yaml` |

Run each from the repo root on the VM, with the inference env on PATH:

```bash
export PATH=$HOME/miniforge3/envs/torch211_cu128_inference/bin:$PATH
export LD_LIBRARY_PATH=$HOME/miniforge3/envs/torch211_cu128_inference/lib
cd ~/segwater_v2
python scripts/evaluation/vm/gate/gen_demak_gate_configs.py
```

Edit `SEEDS` / `ARMS` / `MODEL` at the top of each script to retarget.

Then: run the sweeps → score → collect with
`scripts/evaluation/collect_demak_gate.py --root <eval root>`.
A tracked copy of the generated Hampyeong spec is
`../specs/hampyeong_gate.yaml`.

⚠️ Score VM-side and egress CSVs only — never pull probability rasters
(see the HARD RULE in `../MANIFEST.md`).
