# dataset_v3 — third-generation SEGWATER chip corpus

Code for assessing, and then building, a new chip corpus that applies the
Phase 3 visual-QC decisions and the revised invalid-mask rules.

This directory is **tracked**. It holds code, and nothing heavy. Every raster,
manifest, log and intermediate goes under `outputs/dataset_v3/`, which is
gitignored. That split is the point: the two previous analysis trees
(`dataset_reconstruction/`, `ancillary_local/`) mixed code with their outputs
and so could never be committed — 240 KB of `dataset_reconstruction` code has
no history because it sits inside 9.7 GB of rasters.

## Where things live

| What | Where | Tracked |
|---|---|---|
| Code (this directory) | `scripts/dataset_v3/` | yes |
| Documentation, findings, decisions | `docs/dataset_v3/` | yes, in the nested `docs` repo |
| Manifests, logs, QC tables, chips | `outputs/dataset_v3/` | **no** |

`docs/` is a **separate git repository** nested inside this one, and is
gitignored by the parent. Commit there with `git -C docs ...`. Never
`git add -f` a path under `docs/` from the parent repo.

## Rules

1. **No absolute paths in code.** Roots come from a config or environment
   variable. The build runs both on this Mac and on the VM against
   `/mnt/local_ssd`, and the two do not share a layout. Hard-coded absolute
   paths are the same defect as the stale `asset_path` values in the reviewer
   database and the origin-machine paths in the Level 2 b8 checksum manifest.

2. **Heavy artifacts never enter git.** `outputs/` is already covered by
   `.gitignore`. Do not add exceptions.

3. **Small, final manifests may be tracked** — deliberately, by name, when a
   result needs to be reproducible from a commit. Intermediates stay in
   `outputs/`.

## Inputs (staged on the VM, hash-verified)

Both archives were downloaded from `gdrive:Segwater_v2_RAW_DATASET` and
verified; see `docs/dataset_v3/` for the staging record.

- `~/SEGWATER_CHIP_REVIEWER_663/` — Phase 3 reviewer project. Archive
  SHA-256 `5830c81c...6afa342` matches the value Noel supplied. All four
  entries in `KEY_FILES_SHA256.txt` verify.
- `~/SEGWATER_V2_LABELS_S2_CSPLUS_L2/` — Level 2 labels, B8<400 alternatives,
  generation scripts, provenance. 3385/3385 and 663/663 checksums verify.
- `/mnt/local_ssd/segwater_v2_raw/` — S2 corpora (SR, CS+, s2cloudless),
  `DATABASE/`, and the extracted `chips_SCL_LOST/` GPKGs.

## Decision resolution — read before writing any consumer

`chip_decisions` in the reviewer database is **sparse**. Never count or join it
alone to derive final results. Resolution is gated on the scene:

- committed scene, `scene_mode='nir-all'` → every chip resolves `apply-nir`,
  and `chip_decisions` is intentionally empty for that scene;
- committed scene, `scene_mode='granular'` → explicit `nir` → `apply-nir`,
  explicit `reject` → `reject`, **no row → `keep-original`**.

`decision_events` is audit history and must not be used to materialize
decisions.

Prefer the already-resolved export, which carries one row per chip and a
`resolved_action` column:

```
Visual_Quality_Control/3rd_round_ee_harmonized/segwater_chip_decisions (2).csv
```

Verified totals: 663 scenes, 303,834 chip rows, 147,452 `apply-nir`,
60,263 `keep-original`, 96,119 `reject`.

`reject` means **excluding the chip from the final chip corpus**. It does not
mean writing 255 into the scene label, unless Noel asks for that representation
explicitly.

Locate reviewer bundles as `segwater-chip-reviewer-663/assets/{pair_id}.sqlite`.
Do not trust the `asset_path` column.

## Open questions

- Phase 3 covers **663** scenes; the Level 2 labels cover **3,385** pairs. How
  the 2,722 pairs with no Phase 3 decision are treated in the new corpus is
  undecided.
- The invalid-mask rules have changed. The revised rule is not yet written
  down here; `gen_label.py` in the Level 2 archive implements the *previous*
  one.
