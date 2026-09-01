# Supplementary figure captions — dataset_v3 test-split profile

All statistics are computed on the **131,713 chips of the v3 test split across
316 pairs** — exactly the chips written to `test.memmap`, as recorded in
`dataset_v3_memmap_manifest.parquet` (`split_v3 == 'test'`). "Corpus" means all
1,322,788 chips in the three memmap splits (train 925,123 / val 265,952 / test
131,713 across 3,012 pairs); it excludes the 6,023 chips held back as an
evaluation buffer. "Water share" is n_water / (n_water + n_land), the water
fraction of a chip's **valid** pixels. A chip is "mixed" at purity threshold t
when t < water share < 1 − t; **unless stated otherwise t = 1%**, and the
threshold is always named because the distribution is U-shaped and the mixed
share moves with t (49.9% → 26.4% → 18.6% as t goes 0 → 1% → 5%). Per-chip
backscatter statistics come from exact per-chip histograms on a 320-bin,
0.25 dB grid spanning −55 to +25 dB; percentiles are the left-continuous binned
inverse CDF, so their resolution is one 0.25 dB bin. Aggregation code:
`scripts/dataset_v3/profile_split.py` + `report_profile.py` (VM) and
`make_figures.py` + `data/` here; no chip imagery is used.

**Figure T1 — Where the test split sits.**
(a) Pair centroids: test pairs in blue over train and validation pairs in gray,
dot area proportional to the pair's chip count. The split is whole-pair, and
test pairs interleave with train/val pairs along the same coastlines rather
than occupying a separate region — the holdout is by scene, not by geography.
(b) Chips and pairs per split; the realized shares are 69.94 / 20.11 / 9.96%
against a 70/20/10 target.
(c) Chips per 10° latitude band, test (line) against the corpus (bars). This is
the split's largest compositional deviation: total variation distance 0.077,
driven by the 30–60° N bands.
(d) Chips per pair within test (median 386, p10–p90 134–748, maximum 1,613 in
PAIR_1437).

**Figure T2 — Label composition and representativeness.**
(a) Water-share distribution over the test split (log percentage scale), with
the corpus profile overlaid as a gray outline on the same normalization. The
isolated bars are chips with exactly zero water pixels (21.7%) and exactly zero
land pixels (28.4%); interior bars are colored by the class assigned at t = 1%.
Test and corpus shapes are near-identical (W1 on water share 0.0021).
(b) Three-class composition at t = 0 / 1 / 5%, corpus (pale) beside test
(solid, labelled). The mixed share is as much a statement about t as about the
data: at t = 1% it is 26.42% of test against 25.17% of the corpus.
(c) Total variation distance between test and the corpus over every categorical
axis measured (0 = identical shares). Every axis is below 0.08, and the
composition axes that matter most for a water-segmentation metric — water-share
class, coastline class, climate — are below 0.015.

**Figure T3 — Stratification coverage of the test split.**
Test chips (cell color, log scale; upper number) and test pairs (lower number)
per GCL_FCS30 coastline class × Köppen–Geiger broad climate stratum. Cells
marked "absent" hold no chips anywhere in the corpus. **Cells marked "no test
chips" in orange are the coverage gaps**: `1_D` (biogenic × cold), `3_D`
(muddy × cold) and `5_D` (estuary × cold) exist in the corpus but were forced
whole to train by the thin-stratum rule (< 3 pairs), so the test split carries
no evidence about them at all. Five further occupied cells rest on three pairs
or fewer — biogenic × temperate (1 pair), estuary × arid (1), rocky × polar
(2), biogenic × arid (3), estuary × temperate (3) — where a single scene's
quirks move the whole cell.

**Figure T4 — Pair-macro metric stability.**
(a) Mixed chips per pair. The 14 pairs left of the orange line hold fewer than
5 mixed chips (6 hold none); a pair-macro mean weights each of those equally
with a pair holding hundreds, making them high-variance terms.
(b) Lorenz curve of chip mass over test pairs. Gini 0.36 and a Kish effective
pair count of 219 against 316 actual pairs: 219 is the N behind a pair-macro
metric, not 316 and emphatically not 131,713. Chips within a pair share a
scene, a date, a tide state and a label raster, so an interval computed as if
chips were independent will be too narrow — block-resample whole pairs.
(c) Mixed share against pair size, with the fragile pairs highlighted. They are
small pairs: all 14 hold fewer than 300 chips, so they carry little chip mass
while carrying full weight in a pair-macro mean.

**Figure T5 — Radiometric agreement with train.**
Aggregate distributions of per-chip mean backscatter, test against train, for
VV (a) and VH (b), each on its own axis. The two splits are radiometrically
near-indistinguishable: 1-D Wasserstein distance 0.140 dB (VV) and 0.091 dB
(VH), both far below the 0.25 dB bin width, so no domain shift in backscatter
separates the holdout from the training data.

**Figure T6 — Tidal state and acquisition time.**
(a) Tide level at S1 acquisition (FES2022b, cm relative to mean sea level),
test against corpus; W1 = 4.7 cm.
(b) Tidal phase at S1 acquisition. Test carries slightly more ebb and flood
(95.8% combined) and less high/low slack (4.1%) than the corpus (92.6% / 7.1%),
the split's second-largest categorical deviation (TVD 0.031).
(c) Chips per year of S1 acquisition and (d) per calendar month, test (line)
against corpus (bars), both as within-split shares. Test tracks the corpus
year profile closely (TVD 0.041), with 2019–2021 holding the bulk of both.
