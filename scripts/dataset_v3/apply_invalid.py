#!/usr/bin/env python3
"""Apply a threshold to the invalid scores and record the decision.

Separate from the scoring pass on purpose. Scoring reads 3,383 rasters and
takes minutes; deciding is a re-read of one parquet and takes seconds. Keeping
them apart means the threshold can be changed as often as the evidence
justifies without re-measuring anything.

Writes the ``invalid_*`` evidence columns onto the base table and resolves
``v3_status`` for chips that exceed the threshold. Only ever moves a chip from
``pending`` to ``reject``: this stage removes chips, it never admits them.
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys

import pandas as pd

import invalid_mask
import paths

EXPECTED_CHIPS = 1_474_047


def sweep(frac: pd.Series, label: str, comparison: str) -> None:
    print("%s, by threshold" % label)
    for threshold in (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.50):
        n = int((frac > threshold).sum() if comparison == ">" else (frac <= threshold).sum())
        print("  %s %.2f   %9d  (%.2f%%)"
              % (comparison, threshold, n, 100 * n / max(len(frac), 1)))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threshold", type=float, required=True,
                        help="a chip with invalid_frac above this is rejected; "
                             "no default, because choosing it is a decision")
    parser.add_argument("--write", action="store_true",
                        help="write the base table; without it, only summarise")
    args = parser.parse_args()

    base = pd.read_parquet(
        paths.require(paths.MANIFESTS / "dataset_v3_base.parquet", "base table"))
    scores = pd.read_parquet(
        paths.require(paths.MANIFESTS / "chip_invalid_scores.parquet", "chip scores"))

    if len(base) != EXPECTED_CHIPS:
        raise AssertionError("base is %d chips, expected %d" % (len(base), EXPECTED_CHIPS))

    # Drop any previous run's evidence columns so re-running is idempotent
    # rather than growing the table by one column each time. Everything the
    # scores table contributes is dropped, not a hand-listed subset -- a
    # hand-listed subset is how the duplicate crept in the first time.
    carried = [c for c in scores.columns if c not in ("pair_name", "chip_id")]
    base = base.drop(columns=[c for c in carried if c in base.columns])

    table = base.merge(scores, on=["pair_name", "chip_id"], how="left", validate="1:1")
    if table["invalid_frac"].isna().any():
        raise AssertionError("%d chips have no invalid score"
                             % int(table["invalid_frac"].isna().sum()))
    if len(table) != EXPECTED_CHIPS:
        raise AssertionError("join changed the row count to %d" % len(table))

    over = table["invalid_frac"] > args.threshold

    print("THRESHOLD %.2f, mask %s" % (args.threshold, invalid_mask.MASK_RULE))
    print()
    sweep(table["invalid_frac"], "EXISTING CHIPS REJECTED", ">")
    print()

    rescue_file = paths.MANIFESTS / "rescue_candidates.parquet"
    if rescue_file.exists():
        rescue = pd.read_parquet(rescue_file)
        covered = rescue[rescue["label_covers"]]
        sweep(covered["invalid_frac"], "RESCUE CANDIDATES (of %d covered; %d "
              "uncovered can never rescue)"
              % (len(covered), int((~rescue["label_covers"]).sum())), "<=")
        print()

    # Only chips this stage has an opinion about are touched, and only if no
    # earlier stage already decided them.
    touched = over & (table["v3_status"] == "pending")
    if int(over.sum()) != int(touched.sum()):
        print("NOTE: %d chips exceed the threshold but were already decided by "
              "another stage; left alone." % int((over & ~touched).sum()))

    print("DECISION at %.2f" % args.threshold)
    print("  reject          %9d  (%.2f%%)" % (int(touched.sum()),
                                               100 * float(touched.mean())))
    print("  left pending    %9d" % int((~touched & (table["v3_status"] == "pending")).sum()))
    print()
    print("  rejected chips by scene verdict:")
    for category, n in table.loc[touched, "qc_scene_category"].value_counts().items():
        print("    %-18s %9d" % (category, n))

    if not args.write:
        print("\n(dry run; pass --write to save)")
        return 0

    table.loc[touched, "v3_status"] = "reject"
    table.loc[touched, "v3_stage"] = "invalid_mask"
    table.loc[touched, "v3_reason"] = (
        "invalid_frac>%.2f (%s)" % (args.threshold, invalid_mask.MASK_RULE))
    table.loc[touched, "v3_updated_at"] = dt.datetime.now(
        dt.timezone.utc).isoformat(timespec="seconds")

    if (table["v3_status"] == "include").any():
        raise AssertionError("this stage must never mark a chip included")

    target = paths.MANIFESTS / "dataset_v3_base.parquet"
    table.to_parquet(target, index=False, compression="zstd")
    print("\nwrote %s  (%d rows, %d cols, %.1f MiB)"
          % (target, len(table), len(table.columns), target.stat().st_size / 2**20))
    return 0


if __name__ == "__main__":
    sys.exit(main())
