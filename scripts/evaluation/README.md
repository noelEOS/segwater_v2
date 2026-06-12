  script: exhaustive_ensemble_search.py -> renamed to exhaustive_hard_vote_search.py
  combiner: hard majority vote on binary masks @0.5
  search space: all subsets within each tiling group (127 + 31)
  status: ran (twice — second run added macro)
  results live in: error_decorrelation_analysis_masked/*__exhaustive_ensembles.csv
  ────────────────────────────────────────
  script: evaluate_soft_ensembles.py
  combiner: soft probability mean (+ weighted pairs, stacker, oracle, AUC-PR, LOO threshold tuning)
  search space: 194 sets: all within-tiling subsets + all 66 pairs across the 12 members + ALL-12
  status: ran (twice — second run added macro)
  results live in: soft_ensemble_analysis/
  ────────────────────────────────────────
