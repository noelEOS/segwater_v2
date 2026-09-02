#!/usr/bin/env python3
"""Assign the v3 train/val/test split: exclusion, pins, then a stratified draw.

Writes three columns onto the manifest:

* ``split_v3``            -- train / val / test, or NA (excluded or not passing)
* ``excluded_eval_buffer`` -- True for chips within 30 km of an evaluation AOI
* ``split_v3_reason``     -- why each chip is where it is, one of
  ``eval_buffer``, ``pinned_train``, ``thin_stratum_train``,
  ``stratified_draw``, ``not_passing``

The reason column is what makes the assignment auditable: every chip can say
why it landed where it did.

**Order of operations is load-bearing.**

1. **Exclusion first.** Chips with ``eval_dist_km <= 30`` (measured by
   ``assess_eval_distance.py``) go to no split. Exclusion precedes the draw
   because it removes chips from *all* splits -- applied afterwards it would
   corrupt the chip-weighted quotas the draw was solving for. Decision record:
   ``docs/dataset_v3/EXCLUSION_BUFFER.md`` (chip-level, R = 30 km).
2. **Pins.** 32 pairs are pre-assigned to train. They were hand-curated by
   Noel (2026-09-01) as rich, diverse scenes -- 45% mixed chips against 25%
   corpus-wide, several of them heavy in recovered shoreline chips -- with the
   stated aim of maximizing generalization when the model is applied outside
   this corpus. Pinning to *train* is the safe direction: it can only shrink
   the holdout sampling pool, never hand-pick what is evaluated. Pins compose
   with exclusion: a pinned pair contributes only its surviving chips.
3. **Stratified draw** over the remaining pairs. Pairs are allocated WHOLE
   (scene purity -- the defect of the shipped v2 chip-level split was exactly
   that chips of one scene landed in different splits), stratified by
   ``stratum_coarse`` (GCL_FCS30 coastline class x Koppen-Geiger broad
   climate), chip-weighted toward 70/20/10, natural composition (no
   rebalancing -- pure-land chips stay; their land-cover diversity is signal).

**The draw, precisely.** One ``np.random.default_rng(seed)``; strata iterated
in sorted order; within a stratum the pair list is sorted by name BEFORE
``rng.permutation`` (groupby/merge order is not a stable contract). Per
stratum, over its surviving chip mass: strata with fewer than
``MIN_PAIRS_FOR_THREE_WAY`` pairs go wholly to train (a 1-2 pair stratum
cannot be represented in three splits; the choice of threshold is insensitive
-- every cutoff in [3, 11) selects the same three strata). Otherwise test is
filled first, then val -- the smallest quota is the most sensitive to a single
large pair, so it gets first pick of the shuffled order -- with the
add-while-closer rule: a pair joins the current target while doing so brings
the running total closer to the quota than stopping would. The remainder is
train. Pinned pairs are removed from the pool but their chips are subtracted
from their stratum's train quota first, so pins consume train budget instead
of inflating it, per stratum rather than only globally.

Sliver pairs (PAIR_2770 keeps 8 chips after exclusion, PAIR_3267 keeps 3) are
treated as ordinary pairs: a special rule would be a rule fitted to two pairs,
and their 11 chips cannot move any split's composition. They are disclosed by
``report_splits.py``.

Recovered chips need no handling of their own -- allocation is whole-pair and
they carry ``pair_name`` -- but their inheritance is asserted, not assumed.
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys

import numpy as np
import pandas as pd

import paths

EXPECTED_CHIPS = 1_357_354
EXPECTED_PASSING = 1_328_811
EXPECTED_EXCLUDED = 6_023
EXPECTED_SPLITTABLE = EXPECTED_PASSING - EXPECTED_EXCLUDED    # 1,322,788
EXPECTED_STRATA = 25

BUFFER_KM = 30.0
MIN_PAIRS_FOR_THREE_WAY = 3
TARGETS = {"train": 0.70, "val": 0.20, "test": 0.10}
# Assert-level tolerance on the realized global chip shares, in percentage
# points. Whole-pair allocation cannot hit the targets exactly; these bounds
# are generous against granularity but tight against a broken draw.
TOLERANCE_PP = {"train": 2.0, "val": 1.5, "test": 1.5}

# Hand-curated by Noel, 2026-09-01 -- see the module docstring for the why.
PINNED_PAIRS = frozenset(
    "PAIR_%d" % n for n in (
        1035, 1080, 1083, 1086, 1088, 1090, 1342, 1405, 1670, 1784,
        1897, 1898, 1899, 2391, 2393, 2421, 2484, 2874, 2918, 3018,
        3066, 3300, 3306, 3307, 3368, 3407, 3542, 3717, 4186, 4717,
        4720, 4835,
    )
)
EXPECTED_PINNED_CHIPS = 11_908

PAIR_SUMMARY_CSV = (paths.DB_SLIM
                    / "pair_summary_with_splits-ready_for_memmap_v2_creation.csv")


def assign_strata(chips: pd.DataFrame, seed: int) -> pd.Series:
    """Whole-pair split assignment over the splittable chips.

    ``chips`` must carry ``pair_name``, ``stratum_coarse`` and one row per
    chip. Returns a per-row Series of ``train``/``val``/``test`` plus a
    parallel reason. Deterministic for a given seed: one generator, strata
    sorted, pair lists sorted before shuffling.
    """
    rng = np.random.default_rng(seed)
    pair_split: dict[str, str] = {}
    pair_reason: dict[str, str] = {}

    pair_sizes = chips.groupby("pair_name").size()
    pair_stratum = chips.groupby("pair_name")["stratum_coarse"].first()

    for stratum in sorted(pair_stratum.unique()):
        pairs = sorted(pair_stratum.index[pair_stratum == stratum])
        pinned = [p for p in pairs if p in PINNED_PAIRS]
        drawable = [p for p in pairs if p not in PINNED_PAIRS]
        # Quotas are set on the stratum's FULL surviving mass, pins included:
        # test and val take their shares of it, and train is the remainder --
        # which the pins are already part of. That is how pins consume train
        # budget per stratum instead of inflating it.
        stratum_chips = int(pair_sizes[pairs].sum())

        for pair in pinned:
            pair_split[pair] = "train"
            pair_reason[pair] = "pinned_train"

        if len(pairs) < MIN_PAIRS_FOR_THREE_WAY:
            for pair in drawable:
                pair_split[pair] = "train"
                pair_reason[pair] = "thin_stratum_train"
            continue

        # Chip quotas within this stratum. Pins are part of the stratum's mass
        # and consume train budget; test and val quotas are untouched by them.
        quotas = {"test": TARGETS["test"] * stratum_chips,
                  "val": TARGETS["val"] * stratum_chips}

        order = rng.permutation(len(drawable))
        shuffled = [drawable[i] for i in order]

        filled = {"test": 0, "val": 0}
        cursor = 0
        for target in ("test", "val"):
            quota = quotas[target]
            while cursor < len(shuffled):
                pair = shuffled[cursor]
                n = int(pair_sizes[pair])
                running = filled[target]
                if abs(running + n - quota) < abs(running - quota):
                    pair_split[pair] = target
                    pair_reason[pair] = "stratified_draw"
                    filled[target] = running + n
                    cursor += 1
                else:
                    break
        for pair in shuffled[cursor:]:
            pair_split[pair] = "train"
            pair_reason[pair] = "stratified_draw"

    split = chips["pair_name"].map(pair_split)
    reason = chips["pair_name"].map(pair_reason)
    if split.isna().any():
        raise AssertionError("%d splittable chips left unassigned by the draw"
                             % int(split.isna().sum()))
    return split, reason


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42,
                        help="draw seed. 42 is the shipped assignment; other "
                             "values are for sensitivity checks in dry runs")
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()

    target = paths.require(paths.MANIFESTS / "dataset_v3_manifest.parquet", "v3 manifest")
    manifest = pd.read_parquet(target)
    if len(manifest) != EXPECTED_CHIPS:
        raise AssertionError("manifest is %d chips, expected %d"
                             % (len(manifest), EXPECTED_CHIPS))
    for column in ("passes_all_gates", "chip_origin"):
        if column not in manifest.columns:
            raise AssertionError("%s missing; run the earlier stages first" % column)

    # Idempotent re-run: drop what this stage owns before recomputing it.
    manifest = manifest.drop(columns=[c for c in
                                      ("split_v3", "excluded_eval_buffer",
                                       "split_v3_reason")
                                      if c in manifest.columns])

    distance = pd.read_parquet(
        paths.require(paths.MANIFESTS / "chip_eval_distance.parquet",
                      "eval distances; run assess_eval_distance.py first"))
    merged = manifest.merge(distance, on=["pair_name", "chip_id"],
                            how="left", validate="1:1")
    if len(merged) != EXPECTED_CHIPS:
        raise AssertionError("distance join changed the row count to %d" % len(merged))
    passing = merged["passes_all_gates"]
    if merged.loc[passing, "eval_dist_km"].isna().any():
        raise AssertionError("passing chips without a measured eval distance")

    # (a) Exclusion.
    excluded = passing & (merged["eval_dist_km"] <= BUFFER_KM)
    if int(excluded.sum()) != EXPECTED_EXCLUDED:
        raise AssertionError("excluded %d chips, expected %d"
                             % (int(excluded.sum()), EXPECTED_EXCLUDED))

    splittable = passing & ~excluded
    if int(splittable.sum()) != EXPECTED_SPLITTABLE:
        raise AssertionError("splittable %d chips, expected %d"
                             % (int(splittable.sum()), EXPECTED_SPLITTABLE))

    # (b) Pins -- validated before the draw so a drifted list fails loudly.
    present = set(merged.loc[splittable, "pair_name"].unique())
    missing_pins = PINNED_PAIRS - present
    if missing_pins:
        raise AssertionError("pinned pairs with no splittable chips: %s"
                             % ", ".join(sorted(missing_pins)))
    pin_mask = splittable & merged["pair_name"].isin(PINNED_PAIRS)
    if int(pin_mask.sum()) != EXPECTED_PINNED_CHIPS:
        raise AssertionError("pinned pairs carry %d surviving chips, expected %d"
                             % (int(pin_mask.sum()), EXPECTED_PINNED_CHIPS))
    if int((excluded & merged["pair_name"].isin(PINNED_PAIRS)).sum()):
        # Not an error -- pins compose with exclusion -- but today it is zero,
        # and a silent change here would drift EXPECTED_PINNED_CHIPS.
        raise AssertionError("a pinned pair has excluded chips; update "
                             "EXPECTED_PINNED_CHIPS deliberately")

    # (c) The draw, over splittable chips with their stratum.
    strata = pd.read_csv(
        paths.require(PAIR_SUMMARY_CSV, "pair summary (stratum source)"),
        usecols=["pair_name", "stratum_coarse"])
    chips = merged.loc[splittable, ["pair_name", "chip_id"]].merge(
        strata, on="pair_name", how="left", validate="m:1")
    if len(chips) != int(splittable.sum()):
        raise AssertionError("stratum join changed the chip count")
    if chips["stratum_coarse"].isna().any():
        raise AssertionError("%d splittable chips have no stratum"
                             % int(chips["stratum_coarse"].isna().sum()))
    n_strata = chips["stratum_coarse"].nunique()
    if n_strata != EXPECTED_STRATA:
        raise AssertionError("%d occupied strata, expected %d"
                             % (n_strata, EXPECTED_STRATA))

    split, reason = assign_strata(chips, args.seed)
    # Determinism self-test: an independent second run must agree exactly.
    split2, _ = assign_strata(chips, args.seed)
    if not split.equals(split2):
        raise AssertionError("the draw is not deterministic for a fixed seed")

    merged["split_v3"] = pd.NA
    merged.loc[splittable, "split_v3"] = split.values
    merged["excluded_eval_buffer"] = excluded.fillna(False).astype(bool)
    merged["split_v3_reason"] = "not_passing"
    merged.loc[excluded, "split_v3_reason"] = "eval_buffer"
    merged.loc[splittable, "split_v3_reason"] = reason.values

    # ---- the load-bearing invariants ----
    assigned = merged["split_v3"].notna()
    if int(assigned.sum()) != EXPECTED_SPLITTABLE:
        raise AssertionError("assigned %d chips, expected %d"
                             % (int(assigned.sum()), EXPECTED_SPLITTABLE))
    if (assigned & ~passing).any() or (assigned & excluded).any():
        raise AssertionError("a chip has a split without qualifying for one")

    # Scene purity, the central invariant of Option 1.
    spans = merged.loc[assigned].groupby("pair_name")["split_v3"].nunique()
    if int(spans.max()) != 1:
        raise AssertionError("%d pairs span more than one split"
                             % int((spans > 1).sum()))

    # Recovered chips inherit their pair's split -- asserted, not assumed.
    rec = merged.loc[assigned & (merged["chip_origin"] == "recovered")]
    ex = merged.loc[assigned & (merged["chip_origin"] == "existing")]
    pair_split_existing = ex.groupby("pair_name")["split_v3"].first()
    both = rec["pair_name"].isin(pair_split_existing.index)
    mismatch = rec.loc[both, "split_v3"].values != \
        pair_split_existing[rec.loc[both, "pair_name"]].values
    if mismatch.any():
        raise AssertionError("%d recovered chips differ from their pair's split"
                             % int(mismatch.sum()))

    counts = merged.loc[assigned, "split_v3"].value_counts()
    shares = counts / counts.sum()
    print("SPLIT ASSIGNMENT (seed %d) over %d splittable chips, %d pairs"
          % (args.seed, int(assigned.sum()),
             merged.loc[assigned, "pair_name"].nunique()))
    print()
    for name in ("train", "val", "test"):
        n = int(counts.get(name, 0))
        share = float(shares.get(name, 0.0))
        deviation = 100 * (share - TARGETS[name])
        print("  %-5s  %9s chips  %6.2f%%  target %2.0f%%  deviation %+5.2f pp"
              % (name, "{:,}".format(n), 100 * share, 100 * TARGETS[name], deviation))
        if abs(deviation) > TOLERANCE_PP[name]:
            raise AssertionError("%s deviates %.2f pp from target, tolerance %.1f"
                                 % (name, deviation, TOLERANCE_PP[name]))
    print()
    print("  excluded (eval buffer, %g km)  %9s chips" % (BUFFER_KM, "{:,}".format(int(excluded.sum()))))
    print("  pinned to train                %9s chips in %d pairs"
          % ("{:,}".format(int(pin_mask.sum())), len(PINNED_PAIRS)))
    thin = merged.loc[assigned & (merged["split_v3_reason"] == "thin_stratum_train")]
    print("  thin strata forced to train    %9s chips in %d pairs"
          % ("{:,}".format(len(thin)), thin["pair_name"].nunique()))
    print()
    print("  reasons: %s" % merged["split_v3_reason"].value_counts().to_dict())

    if not args.write:
        print("\n(dry run; pass --write to save)")
        return 0

    if args.seed != 42:
        raise AssertionError("refusing to write a non-default seed; 42 is the "
                             "shipped assignment")
    merged = merged.drop(columns=["eval_dist_km", "nearest_eval_site"])
    merged["v3_updated_at"] = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    merged.to_parquet(target, index=False, compression="zstd")
    print("\nwrote %s  (%d rows, %d cols, %.1f MiB)"
          % (target, len(merged), len(merged.columns), target.stat().st_size / 2**20))
    return 0


if __name__ == "__main__":
    sys.exit(main())
