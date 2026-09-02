# SUPP_test_split_profile — test-split composition figures

Supplementary-materials figure set for the dataset_v3 **test split**: the
131,713 chips / 316 pairs of `split_v3 == 'test'`, which is exactly what
`test.memmap` holds. Baselines throughout are the train split (925,123 chips)
and the corpus (all 1,322,788 memmap chips). Built 2026-09-01.

- `data/` — aggregate CSVs exported from the VM profiles by
  `scripts/dataset_v3/report_profile.py --split test --write`
  (`~/dataset_v3_out/qc/test_profile_figdata/`). Per-pair rows, binned
  histograms and scalars only; no chip-level or raster data, safe to commit.
- `make_figures.py` — renders all six figures. Run:
  `conda run -n eda python make_figures.py`
- `out/` — `fig_T1_geography`, `fig_T2_composition`, `fig_T3_strata`,
  `fig_T4_stability`, `fig_T5_radiometry`, `fig_T6_tide_time`
  (PDF + 300-dpi PNG each).
- `CAPTIONS.md` — self-contained supplementary captions.

Palette, rcParams and helpers are shared verbatim with
`../SUPP_dataset_v3_census/make_figures.py` so the two sets read as one system.

Frames: "water share" = n_water/(n_water+n_land); "mixed" at purity threshold
t = 1% unless stated; per-chip dB statistics from 320-bin 0.25 dB histograms,
percentiles as the left-continuous binned inverse CDF. Source of truth:
`/home/noel/dataset_v3_out/manifests/{test,train,val}_chip_profile.parquet` on
`cpu-hpo-west`, built from `dataset_v3_memmap_manifest.parquet`. Narrative and
full tables: `docs/dataset_v3/TEST_SPLIT_PROFILE.md`; the generated report is
`~/dataset_v3_out/qc/test_profile_report.md`.
