#!/usr/bin/env python3
"""Enact the QC verdicts already recorded in the base table.

The rounds decided what happens to each chip; this writes those decisions into
``v3_status`` and ``v3_label_variant``. It reads no rasters and makes no
judgements of its own -- the evidence columns are the authority.

The rules, in the order they are applied. Earlier rules win, because a scene
rejected outright is not then eligible for a chip-level opinion:

1. **Round-1 ``rejected`` scene** -> every chip excluded. No further action was
   ever taken on these scenes, so there is nothing else to consider.
2. **Round-3 ``reject``** -> chip excluded.
3. **Round-3 ``apply-nir``** -> included, reading the B8<400 label variant.
4. **Round-3 ``keep-original``** -> included, reading the parent label.
5. **Everything else** -> included on the parent label. These are chips in
   accepted scenes that no chip-level review ever flagged.

The invalid-mask stage runs before this one and is not overridden: a chip it
rejected stays rejected, whatever the QC verdict says. Both are exclusions, and
the first reason recorded is the one kept.

``apply-nir`` sets the label *variant*, not the status -- those chips stay in
the corpus and simply read from a different raster. The B8<400 variant's
invalid mask is bit-identical to its parent, so this cannot disturb the invalid
fractions already measured.
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys

import pandas as pd

import paths

EXPECTED_CHIPS = 1_474_047

PARENT = "parent"
B8_LT400 = "b8_lt400"


def resolve(table: pd.DataFrame) -> pd.DataFrame:
    """Assign status, variant and reason for every chip not already decided."""
    status = table["v3_status"].copy()
    variant = pd.Series(pd.NA, index=table.index, dtype="object")
    reason = pd.Series(pd.NA, index=table.index, dtype="object")

    undecided = status == "pending"
    scene = table["qc_scene_category"]
    action = table["phase3_action"]

    rules = [
        ("reject", undecided & (scene == "rejected"),
         PARENT, "round-1 scene rejected"),
        ("reject", undecided & (action == "reject"),
         PARENT, "round-3 chip rejected"),
        ("include", undecided & (action == "apply-nir"),
         B8_LT400, "round-3 apply-nir"),
        ("include", undecided & (action == "keep-original"),
         PARENT, "round-3 keep-original"),
        ("include", undecided & action.isna() & (scene == "accepted"),
         PARENT, "accepted scene, no chip-level review"),
    ]

    for new_status, mask, new_variant, why in rules:
        # Re-check `pending` each time so an earlier rule is never overwritten.
        mask = mask & (status == "pending")
        status[mask] = new_status
        variant[mask] = new_variant
        reason[mask] = why

    # A chip the invalid stage already rejected keeps that reason, but still
    # needs a variant so a consumer knows which raster it came from.
    already = table["v3_status"] == "reject"
    variant[already] = table.loc[already, "phase3_action"].map(
        {"apply-nir": B8_LT400}).fillna(PARENT)

    return pd.DataFrame({"status": status, "variant": variant, "reason": reason})


def summarise(table: pd.DataFrame, resolved: pd.DataFrame) -> None:
    before = table["v3_status"]
    print("BEFORE")
    print("  %s" % before.value_counts().to_dict())
    print()
    print("AFTER")
    print("  %s" % resolved["status"].value_counts().to_dict())
    print()
    print("BY RULE (chips this stage decided)")
    touched = before == "pending"
    for why, n in resolved.loc[touched, "reason"].value_counts().items():
        status = resolved.loc[touched & (resolved["reason"] == why), "status"].iloc[0]
        print("  %-42s %-8s %9d" % (why, status, n))
    print()
    print("LABEL VARIANT (included chips)")
    included = resolved["status"] == "include"
    print("  %s" % resolved.loc[included, "variant"].value_counts().to_dict())
    print()
    still = int((resolved["status"] == "pending").sum())
    print("STILL PENDING  %d" % still)
    if still:
        print("  ^ every chip should be decided; investigate before writing")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true",
                        help="write the base table; without it, only summarise")
    args = parser.parse_args()

    target = paths.require(paths.MANIFESTS / "dataset_v3_base.parquet", "base table")
    table = pd.read_parquet(target)
    if len(table) != EXPECTED_CHIPS:
        raise AssertionError("base is %d chips, expected %d"
                             % (len(table), EXPECTED_CHIPS))
    if "invalid_frac" not in table.columns:
        raise AssertionError("run apply_invalid.py first; the invalid-mask "
                             "decision must precede the QC verdicts")

    resolved = resolve(table)
    summarise(table, resolved)

    # Every chip must land somewhere: a chip left pending fell through every
    # rule, which is a bug in the rules rather than a property of the chip.
    if (resolved["status"] == "pending").any():
        raise AssertionError("%d chips matched no rule"
                             % int((resolved["status"] == "pending").sum()))
    # An exclusion already recorded must survive this stage unchanged.
    kept = (table["v3_status"] == "reject") == (
        (resolved["status"] == "reject") & (table["v3_status"] == "reject"))
    if not kept.all():
        raise AssertionError("this stage changed a chip the invalid stage rejected")
    # Only round-3 apply-nir chips may read the B8 variant.
    b8 = resolved["variant"] == B8_LT400
    if not (table.loc[b8, "phase3_action"] == "apply-nir").all():
        raise AssertionError("a chip reads the B8 variant without an apply-nir decision")

    if not args.write:
        print("\n(dry run; pass --write to save)")
        return 0

    now = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    touched = table["v3_status"] == "pending"
    table["v3_status"] = resolved["status"]
    table["v3_label_variant"] = resolved["variant"]
    table.loc[touched, "v3_reason"] = resolved.loc[touched, "reason"]
    table.loc[touched, "v3_stage"] = "qc_verdicts"
    table.loc[touched, "v3_updated_at"] = now

    table.to_parquet(target, index=False, compression="zstd")
    print("\nwrote %s  (%d rows, %.1f MiB)"
          % (target, len(table), target.stat().st_size / 2**20))
    return 0


if __name__ == "__main__":
    sys.exit(main())
