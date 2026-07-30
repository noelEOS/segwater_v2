# ⚠️ DEPRECATED (2026-07-30) — checkpoint-selection gate config generators

> **Superseded by `../ship/gen_ship_configs.py`.** Do not generate NEW configs
> with these four scripts. They are kept as the **executable record** of the
> tracked configs they produced for the 2026-07-25 gate; the table below is
> historical.
>
> Each file's module docstring names the specific safeguards it lacks. In short:
> their four near-duplicate `resolve()` copies signal every failure by
> **returning `None`** (so `main()` emits a partial set and continues), accept a
> `best.pth` on `exists()` alone **without checking its target is not a
> `*_last.pth`**, and do **no** seed-token or sha256-distinctness check. None of
> them checks the emitted sweep names for **prefix collisions**. The ship
> generator gets all of that from `ckptsel` (`resolve_best` / `resolve_last` /
> `require_seed_token` / `assert_distinct_weights`, all raising `CkptSelError`)
> and `naming.require_no_prefix_collisions`.
>
> They were deliberately **not** hardened: adding guards to code that must not
> be used is contradictory, and the header edits keep them parseable so the
> record stays executable.

## Historical record

Generators that turned a set of stage-2 seed dirs into the per-arm inference and
scoring configs for the four-way gate in
`docs/RUNBOOK_checkpoint_selection_gate.md`. Written for the 2026-07-25
three-seed Swin-B run (best / last / SWA-5).

All four **resolve checkpoints by glob against the real seed dir** rather than
hardcoding filenames, and decline to emit a config for an arm whose checkpoint is
missing or ambiguous — but see the deprecation note above for how quietly that
declining happens.

| Script | Emits | Where |
|---|---|---|
| `gen_demak_gate_configs.py` | `run_inference_sweep.py` configs, one per arm×seed | `~/configs/demak_gate/` |
| `gen_demak_gate_aucroc_configs.py` | AUC-ROC scoring configs, discovered from the finished run dirs (checks 6/6 probability tifs first) | `~/configs/aucroc_gate/` |
| `gen_hampyeong_gate_configs.py` | Hampyeong sweep configs (length filter OFF, `keep_top_k` 5) | `~/configs/hampyeong_gate/` |
| `gen_hampyeong_gate_spec.py` | one 9-entry spec for `score_pairbased_hampyeong.py` | `~/configs/hampyeong_gate/spec_hampyeong_gate.yaml` |

Checkpoint resolution for any NEW work goes through `../ckptsel.py`
(`best.pth` symlink only — never glob `*_pmwiou*.pth`, there are several per seed
dir and a glob picks by sort order). See `docs/RUNBOOK_checkpoint_ladder.md`.

How these were run, for the record (repo root on the VM, inference env on PATH):

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
