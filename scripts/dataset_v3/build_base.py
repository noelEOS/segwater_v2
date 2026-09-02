#!/usr/bin/env python3
"""Build the dataset_v3 base table: one row per memmap-selected chip.

The base is every chip with ``selected_for_memmap`` true in
``Sen12Coast_CHIP_MEMMAP_INDEX.parquet`` -- 1,474,047 chips across 3,383
pairs. Those are the chips that actually reached a memmap, so they are the
only ones a model can be trained on.

Two of the corpus's 3,385 pairs (PAIR_1266, PAIR_3254) contribute no selected
chip and are therefore absent. They are not missing data: both had chips in
earlier lineages (734 in the first round, 973 in V2) and were cut at the
``CONFIRMED`` step, which removed 223 pairs and 110,821 rows. What survives in
the final lineage is a single unselected Class-7 recovery chip each. Since
neither is in a memmap, neither can be used.

Each chip carries the whole QC lineage that produced it: the round-1 scene
verdict, the round-2 verdict after SCL=11 was released, whether its label was
regenerated in that round, and its round-3 per-chip decision if it had one.
See ``docs/dataset_v3/QC_ROUNDS.md`` for the sequence in prose.

Evidence and decisions are kept apart. The ``qc_*`` and ``phase3_*`` columns
are evidence -- what review actually said -- written once and never revised.
The ``v3_*`` columns are decisions, rewritten as the build progresses. A rule
change therefore rewrites decisions without destroying the record they rest on.

Writes ``manifests/dataset_v3_base.parquet`` under the outputs root. Nothing
is written unless --write is passed.
"""

from __future__ import annotations

import argparse
import datetime as dt
import sqlite3
import sys

import pandas as pd
import pyarrow.parquet as pq

import paths


# The corpus has 3,385 pairs; 3,383 contribute a selected chip. See the module
# docstring. Asserted so a corpus swap cannot silently change the base.
EXPECTED_CHIPS = 1_474_047
EXPECTED_PAIRS = 3_383
EXPECTED_ORPHAN_PAIRS = {"PAIR_1266", "PAIR_3254"}

# Phase 3 resolved totals, over all 303,834 reviewed rows.
EXPECTED_RESOLVED = {"apply-nir": 147_452, "keep-original": 60_263, "reject": 96_119}

# Scene-level QC. See docs/dataset_v3/QC_ROUNDS.md for the full sequence.
# Round 1 reviewed all 3,385 pairs; round 2 released SCL=11 on the 206 pairs
# carrying a snow problem, which resolved the 159 snow-only pairs to accepted
# and left the 47 both-corrections pairs still needing water work; round 3
# reviewed the resulting 663 water-correction pairs per chip.
EXPECTED_QC1 = {"accepted": 2259, "water-correction": 616, "rejected": 304,
                "snow-correction": 159, "both-corrections": 47}
EXPECTED_QC2 = {"accepted": 2418, "water-correction": 663, "rejected": 304}
EXPECTED_SCL11_RELEASED = 206

# Scenes rejected wholesale in round 3 ("select all chips", then "reject"),
# rather than chip by chip. See detect_scene_rejects.
EXPECTED_SCENE_REJECTS = 60
EXPECTED_SCENE_REJECT_CHIPS = 23_210

INDEX_COLUMNS = [
    "pair_name", "chip_id", "selected_for_memmap",
    "chipbased_split", "chipbased_row",
    "pairbased_split", "pairbased_row",
    "system:index_s1", "system:index_s2", "system:time_start_s1",
    "intersects_gcl_coastline",
    "bbox_w", "bbox_s", "bbox_e", "bbox_n",
]


def load_base() -> pd.DataFrame:
    """Every memmap-selected chip, keyed (pair_name, chip_id)."""
    source = paths.require(paths.DB_CHIP_MEMMAP_INDEX, "chip memmap index")
    index = pq.read_table(source, columns=INDEX_COLUMNS).to_pandas()

    # selected_for_memmap is True/null in some lineages, never False, so a bare
    # boolean mask raises on the nulls. Normalise before masking.
    selected = index["selected_for_memmap"].fillna(False).astype(bool)
    base = index[selected].drop(columns=["selected_for_memmap"]).reset_index(drop=True)

    absent = EXPECTED_ORPHAN_PAIRS - set(base["pair_name"])
    if absent != EXPECTED_ORPHAN_PAIRS:
        raise AssertionError(
            "expected %s to contribute no selected chip; present: %s"
            % (sorted(EXPECTED_ORPHAN_PAIRS), sorted(EXPECTED_ORPHAN_PAIRS - absent))
        )
    if base.duplicated(["pair_name", "chip_id"]).any():
        raise AssertionError("(pair_name, chip_id) is not unique in the base")
    return base


def load_phase3() -> pd.DataFrame:
    """The resolved Phase 3 export: one row per reviewed chip.

    Read this rather than resolving the sparse ``chip_decisions`` table by
    hand. A missing row there means ``keep-original``, so counting or joining
    that table alone silently understates the corpus.
    """
    source = paths.require(paths.RESOLVED_CSV, "resolved Phase 3 export")
    resolved = pd.read_csv(
        source,
        usecols=["pair_id", "chip_id", "resolved_action", "scene_mode",
                 "source_category", "committed_at", "revision"],
    )
    counts = resolved["resolved_action"].value_counts().to_dict()
    if counts != EXPECTED_RESOLVED:
        raise AssertionError("resolved_action counts changed: %r" % counts)
    if resolved.duplicated(["pair_id", "chip_id"]).any():
        raise AssertionError("(pair_id, chip_id) is not unique in the export")
    return resolved.rename(columns={"pair_id": "pair_name"})


def detect_scene_rejects(resolved: pd.DataFrame) -> set[str]:
    """Scenes rejected wholesale in round 3, rather than chip by chip.

    The reviewer has no "reject the scene" control and the database records no
    such flag: ``scene_mode`` is only ``granular`` or ``nir-all``. A scene-wide
    rejection was done by selecting every chip and rejecting, which leaves a
    signature this reads back.

    Two independent signatures, both required:

    * every chip of the scene resolves to ``reject``; **and**
    * every one of those is an *explicit* row in the sparse ``chip_decisions``
      table. Ordinary per-chip work leaves untouched chips implicit, so a
      scene with no implicit chips at all was acted on in bulk.

    Corroborated by ``decision_events``, where each of these scenes is a single
    commit whose payload is uniformly ``reject`` -- one action, not hundreds.

    This is inference from a signature, not a recorded fact. A scene rejected
    chip by chip until none remained would look identical. The distribution
    makes that unlikely: only 8 scenes fall between 75% and 100% rejected,
    then 60 sit exactly at 100%.
    """
    database = paths.require(paths.REVIEWER_DB, "reviewer database")
    with sqlite3.connect("file:%s?mode=ro" % database, uri=True) as connection:
        explicit = pd.read_sql(
            "SELECT pair_id, COUNT(*) AS n_explicit_reject FROM chip_decisions "
            "WHERE action = 'reject' GROUP BY pair_id",
            connection,
        )

    chips = resolved.groupby("pair_name").size().rename("n_chips")
    rejects = (resolved[resolved["resolved_action"] == "reject"]
               .groupby("pair_name").size().rename("n_reject"))
    counts = pd.concat([chips, rejects], axis=1).fillna({"n_reject": 0})
    counts = counts.join(explicit.set_index("pair_id")["n_explicit_reject"]).fillna(
        {"n_explicit_reject": 0})

    whole = counts["n_reject"] == counts["n_chips"]
    all_explicit = counts["n_explicit_reject"] == counts["n_chips"]
    scenes = set(counts.index[whole & all_explicit])

    # A scene that resolves fully to reject but keeps implicit chips would mean
    # the two signatures disagree, and the inference above would not hold.
    if int((whole & ~all_explicit).sum()):
        raise AssertionError(
            "%d fully-rejected scenes have implicit chips; the bulk-reject "
            "signature is no longer reliable" % int((whole & ~all_explicit).sum())
        )
    if len(scenes) != EXPECTED_SCENE_REJECTS:
        raise AssertionError("expected %d scene-level rejects, found %d"
                             % (EXPECTED_SCENE_REJECTS, len(scenes)))
    return scenes


def load_qc_rounds() -> pd.DataFrame:
    """Scene-level verdicts from QC rounds 1 and 2, one row per pair.

    Round 2 regenerated the 206 pairs with a snow problem under
    ``--scl11-policy ignore``, so ``scl11_released`` records which labels were
    rebuilt with SCL=11 no longer invalidating on its own. That flag is a
    property of the label file, not a review verdict, and the two are kept
    apart deliberately.
    """
    directory = paths.LABELS_PROVENANCE / "round_2_scl11_correction"
    round1 = pd.read_csv(paths.require(directory / "source_round1_review.csv", "round 1 review"),
                         usecols=["pair_id", "source_category"])
    round2 = pd.read_csv(paths.require(directory / "output_round2_review.csv", "round 2 review"),
                         usecols=["pair_id", "source_category"])
    released = {
        line.strip()
        for line in paths.require(directory / "scl11_ignore_pairs.txt",
                                  "SCL=11 released pair list").read_text().split()
        if line.strip()
    }

    for name, frame, expected in (("round 1", round1, EXPECTED_QC1), ("round 2", round2, EXPECTED_QC2)):
        counts = frame["source_category"].value_counts().to_dict()
        if counts != expected:
            raise AssertionError("%s categories changed: %r" % (name, counts))
    if len(released) != EXPECTED_SCL11_RELEASED:
        raise AssertionError("expected %d SCL=11-released pairs, found %d"
                             % (EXPECTED_SCL11_RELEASED, len(released)))

    qc = round1.merge(round2, on="pair_id", suffixes=("_r1", "_r2"), validate="1:1")
    qc["pair_name"] = "PAIR_" + qc["pair_id"].astype(str)
    qc["scl11_released"] = qc["pair_name"].isin(released)

    # The released set must be exactly the snow-carrying pairs of round 1.
    snow = set(qc.loc[qc["source_category_r1"].isin(["snow-correction", "both-corrections"]), "pair_name"])
    if set(qc.loc[qc["scl11_released"], "pair_name"]) != snow:
        raise AssertionError("SCL=11-released set is not the round-1 snow-carrying set")

    return qc.rename(columns={"source_category_r1": "qc_round1_category",
                              "source_category_r2": "qc_scene_category"})[
        ["pair_name", "qc_round1_category", "qc_scene_category", "scl11_released"]
    ]


def build(base: pd.DataFrame, resolved: pd.DataFrame, qc: pd.DataFrame,
          scene_rejects: set[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Join Phase 3 evidence onto the base and seed the v3 decision columns."""
    reviewed_pairs = set(resolved["pair_name"])

    # Reviewed chips that are not memmap-selected. They were excluded before
    # Phase 3 ran, so a decision about them cannot readmit them; they are
    # reported separately rather than dropped in silence.
    base_keys = set(map(tuple, base[["pair_name", "chip_id"]].to_numpy()))
    resolved_keys = set(map(tuple, resolved[["pair_name", "chip_id"]].to_numpy()))
    orphans = resolved[
        [key not in base_keys for key in map(tuple, resolved[["pair_name", "chip_id"]].to_numpy())]
    ].copy()
    del resolved_keys

    table = base.merge(resolved, on=["pair_name", "chip_id"], how="left", validate="1:1")
    table = table.rename(
        columns={
            "resolved_action": "phase3_action",
            "scene_mode": "phase3_scene_mode",
            "source_category": "phase3_source_category",
            "committed_at": "phase3_committed_at",
            "revision": "phase3_revision",
        }
    )
    table["phase3_reviewed"] = table["pair_name"].isin(reviewed_pairs)

    # Every chip of a reviewed scene must carry a decision: within a committed
    # scene the export is complete, so a null here means the join lost a row.
    gap = table["phase3_reviewed"] & table["phase3_action"].isna()
    if gap.any():
        raise AssertionError("%d chips in reviewed scenes lack a decision" % int(gap.sum()))

    # Scene-level QC lineage. Left join: every base pair must have a verdict,
    # since round 1 reviewed the whole corpus.
    table = table.merge(qc, on="pair_name", how="left", validate="m:1")
    if table["qc_scene_category"].isna().any():
        missing = int(table["qc_scene_category"].isna().sum())
        raise AssertionError("%d chips belong to a pair with no QC verdict" % missing)

    # The two sources must agree on which scenes went to round 3: the reviewed
    # set is exactly the round-2 water-correction set. If these ever diverge,
    # one of the inputs is from a different run and the join is meaningless.
    water = set(table.loc[table["qc_scene_category"] == "water-correction", "pair_name"])
    if water != set(table.loc[table["phase3_reviewed"], "pair_name"]):
        raise AssertionError("round-2 water-correction scenes != Phase 3 reviewed scenes")

    # Which round-3 rejections were scene-wide rather than per-chip. A chip
    # here was not judged on its own merits, so the two are not the same
    # evidence even though both resolve to `reject`.
    table["phase3_scene_reject"] = table["pair_name"].isin(scene_rejects)
    if (table["phase3_scene_reject"] & (table["phase3_action"] != "reject")).any():
        raise AssertionError("a scene-reject chip does not resolve to reject")

    # v3 decision columns. Everything starts `pending`: no chip is included
    # until a stage says so, so the unreviewed majority stays visible in every
    # count instead of being silently assumed good.
    table["v3_status"] = "pending"
    table["v3_label_variant"] = pd.NA
    table["v3_stage"] = "base"
    table["v3_reason"] = pd.NA
    table["v3_updated_at"] = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")

    return table, orphans


def summarise(table: pd.DataFrame, orphans: pd.DataFrame) -> None:
    total = len(table)
    reviewed = table["phase3_reviewed"]
    print("BASE")
    print("  chips                     %9d" % total)
    print("  pairs                     %9d" % table["pair_name"].nunique())
    print()
    print("SCENE QC (chips, by round-2 scene verdict)")
    scenes = table.drop_duplicates("pair_name")
    for category, n in table["qc_scene_category"].value_counts().items():
        pairs = int((scenes["qc_scene_category"] == category).sum())
        print("  %-18s  %9d chips  %5d pairs" % (category, n, pairs))
    print("  SCL=11 released     %9d chips  %5d pairs"
          % (int(table["scl11_released"].sum()), int(scenes["scl11_released"].sum())))
    print()
    print("  round 1 -> round 2 (pairs)")
    flow = pd.crosstab(scenes["qc_round1_category"], scenes["qc_scene_category"])
    for line in flow.to_string().splitlines():
        print("    " + line)
    print()
    print("PHASE 3 COVERAGE")
    print("  chips in reviewed scenes  %9d  (%.1f%%)" % (int(reviewed.sum()), 100 * reviewed.mean()))
    print("  chips with no decision    %9d  (%.1f%%)" % (int((~reviewed).sum()), 100 * (~reviewed).mean()))
    print("  reviewed scenes           %9d" % table.loc[reviewed, "pair_name"].nunique())
    print()
    print("PHASE 3 ACTION, over memmap-selected chips only")
    for action, n in table["phase3_action"].value_counts().items():
        print("  %-24s  %9d" % (action, n))
    scene_reject = table["phase3_scene_reject"]
    print()
    print("  of the rejects, scene-wide        %9d  (%d scenes)"
          % (int(scene_reject.sum()), table.loc[scene_reject, "pair_name"].nunique()))
    print("  of the rejects, per-chip          %9d"
          % int(((table["phase3_action"] == "reject") & ~scene_reject).sum()))
    print()
    print("  reviewed but NOT memmap-selected  %9d" % len(orphans))
    print("    (excluded before Phase 3; a decision cannot readmit them)")
    print()
    print("SPLITS (chip-based / pair-based)")
    for column in ("chipbased_split", "pairbased_split"):
        counts = table[column].value_counts().to_dict()
        print("  %-16s %s" % (column, {k: int(v) for k, v in sorted(counts.items())}))
    print()
    print("V3 STATE (seeded)")
    print("  %s" % table["v3_status"].value_counts().to_dict())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true",
                        help="write the parquet; without it, only summarise")
    args = parser.parse_args()

    base = load_base()
    if len(base) != EXPECTED_CHIPS or base["pair_name"].nunique() != EXPECTED_PAIRS:
        raise AssertionError(
            "base is %d chips / %d pairs, expected %d / %d"
            % (len(base), base["pair_name"].nunique(), EXPECTED_CHIPS, EXPECTED_PAIRS)
        )

    resolved = load_phase3()
    table, orphans = build(base, resolved, load_qc_rounds(),
                           detect_scene_rejects(resolved))
    summarise(table, orphans)

    if not args.write:
        print("\n(dry run; pass --write to save)")
        return 0

    paths.ensure_out()
    target = paths.MANIFESTS / "dataset_v3_base.parquet"
    table.to_parquet(target, index=False, compression="zstd")
    orphan_target = paths.MANIFESTS / "dataset_v3_reviewed_not_selected.parquet"
    orphans.to_parquet(orphan_target, index=False, compression="zstd")
    print("\nwrote %s  (%d rows, %.1f MiB)"
          % (target, len(table), target.stat().st_size / 2**20))
    print("wrote %s  (%d rows)" % (orphan_target, len(orphans)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
