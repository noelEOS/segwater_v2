# Campaign drivers — the scripts that actually ran the VM work

**Rescued from `~/workspace/scripts` on the GCP VM, 2026-07-31.** That directory
was not a git repo and was not backed up; 61 of its 76 files had no counterpart
anywhere in this repo. If the VM had been deleted, the operational record of how
every campaign was executed would have gone with it.

This is a **record, not a library.** These are the concrete invocations that
produced results — config generators, phase drivers, scoring loops. They contain
hardcoded `~/` paths, VM-specific env exports and campaign-specific names. Do not
import them; read them to see what was run, and copy the pattern.

⚠️ **The reusable, hardened implementations live in `../ship/` and
`../analysis/`.** Where a name appears in both, the tracked one is authoritative
— several of these VM copies are *older* than their tracked counterparts.

## What is deliberately NOT here

| dropped | why |
|---|---|
| `patch5.py`, `patch_glob.py`, `fix_best_audit.py` | Spent one-shot patchers. Each rewrites a specific line in a file at a path that no longer exists, and ends in an `assert` that would now fail. |
| `build_full3_areas.py`, `build_full5_areas.py`, `build_s112_areas.py` | Superseded. The tracked `../analysis/` copies add `runsel` run-dir resolution (the VM copies still use bare prefix globs). Restoring these would reintroduce the hazard. |
| `collate_sds_arms2.py` | Superseded — the tracked copy adds ~200 lines of hardening (`--expect-arms`, multi-match detection, atomic writes). |
| `fit_s2matched.py`, `fit_trend_by_arm.py` | Byte-identical to the tracked `../analysis/` copies; no reason for a second copy. |

## What is here

**Phase drivers** (`run_*_inference.sh`) — serial, one-GPU inference for a
campaign, cheap gates first. `run_ship_inference.sh` (Swin-B),
`run_cnxb_inference.sh`, `run_cnxt_inference.sh`, `run_swa5_inference.sh`.

**SDS scoring loops** (`run_*_sds.sh`, `sds_loop.sh`) — CPU-only with
`CUDA_VISIBLE_DEVICES=""` so they never contend with the GPU job. Idempotent:
skip logic keyed on the output artifact (`sweep_metrics.csv`), gated on input
completeness. ⚠️ Several encode the **per-site required flags** — Duck's
`--keep-top-k 999`, Trucvert/Torrey `--no-min-chainage-length` — so they are also
the operational record of how those sites were scored. Those requirements are now
enforced in code (`sds_core.SITE_REQUIRED_FLAGS`).

**Frequency maps** — `run_cnx_freqmaps.sh` (the two ConvNeXtV2 campaigns),
`run_matched.sh` (matched/samewin scene sets across all three lineages).

**Config generators** (`gen_*.py`) — emitted the sweep/eval configs for the
earlier rounds (proxy17, demak aug/noaug, 3-seed, duck, trucvert/torrey,
hampyeong, mx630s2, arms2, full-3-lineage). Superseded for *new* campaigns by
`../ship/gen_ship_configs.py`, which adds sha256 distinctness, seed-token checks
and prefix-collision detection.

**Analysis / probes** — `subset_check.py`, `verify_rerun_equality.py`,
`perf_raster_diff.py`, `vh_probe*.py`, `build_demak_*_areas.py`,
`build_proxy17_areas.py`, `fit_demak_trend.py`, `fit_s112.py`,
`gen_arms2_gate_eval.py`, `finalize_ship.sh`.

## Provenance caveats

- Paths assume the VM layout: `~/segwater_v2`, `~/configs/…`,
  `~/workspace/{results,logs}`, `~/ancillary`, `~/SDS_Benchmark_slim`.
- Every shell driver needs both env exports (`PATH` and `LD_LIBRARY_PATH` for
  `torch211_cu128_inference`) — `conda activate` does not work in
  non-interactive SSH, and without `LD_LIBRARY_PATH` model imports die on CXXABI.
- Campaign tags in these names (`ship`, `cnxb`, `cnxt`, `mx630s2`, `arms2`) are
  **campaign** identifiers, not model names. See
  `experiments/mx630_stage2/README.md` for which is which.
