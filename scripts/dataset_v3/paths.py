#!/usr/bin/env python3
"""Filesystem roots for the dataset_v3 build.

Every path used by this package resolves through here, so the same code runs
on the Mac and on the VM without edits. Override any root with the matching
``SEGWATER_V3_*`` environment variable.

Nothing in this module touches the filesystem at import time. Call
:func:`require` when a path must exist, so a missing input fails with a
readable message instead of a confusing error deep in a build.
"""

from __future__ import annotations

import os
from pathlib import Path


def _root(env: str, default: Path) -> Path:
    value = os.environ.get(env)
    return Path(value).expanduser() if value else default


# Repository layout. REPO is derived from this file's location, never hard-coded.
REPO = Path(__file__).resolve().parents[2]
CODE = REPO / "scripts" / "dataset_v3"

# Heavy artifacts. Gitignored; safe to fill with rasters.
OUT = _root("SEGWATER_V3_OUT", REPO / "outputs" / "dataset_v3")
MANIFESTS = OUT / "manifests"
LOGS = OUT / "logs"
QC = OUT / "qc"

# Documentation lives in the nested `docs` git repository.
DOCS = REPO / "docs" / "dataset_v3"

# ---------------------------------------------------------------------------
# Staged inputs. The defaults are the VM layout, where the build runs; set the
# environment variables to point at a local copy.
# ---------------------------------------------------------------------------

# Phase 3 reviewer project.
REVIEWER = _root("SEGWATER_V3_REVIEWER", Path.home() / "SEGWATER_CHIP_REVIEWER_663")
REVIEWER_PROJECT = REVIEWER / "segwater-chip-reviewer-663"
REVIEWER_DB = REVIEWER_PROJECT / "data" / "segwater_chip_decisions.sqlite3"
REVIEWER_ASSETS = REVIEWER_PROJECT / "assets"
REVIEWER_SNAPSHOT = (
    REVIEWER
    / "SEGWATER_CHIP_REVIEWER_663_PHASE3_COMPLETE_METADATA"
    / "segwater_chip_decisions_verified_snapshot.sqlite3"
)

# The resolved Phase 3 export: one row per chip, with a `resolved_action`
# column. Prefer this over resolving the sparse database by hand.
#
# Shipped as "segwater_chip_decisions (2).csv"; renamed on 2026-08-31 to drop
# the browser-download suffix. Content is untouched -- SHA-256 04866ada... still
# matches KEY_FILES_SHA256.txt, which records the file under its old name.
RESOLVED_DIR = REVIEWER / "Visual_Quality_Control" / "3rd_round_ee_harmonized"
RESOLVED_CSV = RESOLVED_DIR / "segwater_chip_decisions.csv"
RESOLVED_JSON = RESOLVED_DIR / "segwater_chip_decisions.json"

# Level 2 labels.
LABELS_L2 = _root(
    "SEGWATER_V3_LABELS", Path.home() / "SEGWATER_V2_LABELS_S2_CSPLUS_L2"
) / "SEGWATER_V2_LABELS_S2_CSPLUS_Level_2"
LABELS_OUT = LABELS_L2 / "out"                    # 3,385 parent labels
LABELS_B8 = LABELS_L2 / "b8_lt400_water_663"      # 663 B8<400 alternatives
LABELS_PROVENANCE = LABELS_L2 / "provenance"
LABELS_SCRIPTS = LABELS_L2 / "scripts"
LABELS_LEDGER = LABELS_L2 / "ledger.csv"

# Raw corpora and the chip database.
RAW = _root("SEGWATER_V3_RAW", Path("/mnt/local_ssd/segwater_v2_raw"))
S2_SR = RAW / "sentinel2_sr_harmonized_ee"
S2_CSPLUS = RAW / "sentinel2_cloudscore_plus_chipext"
S2_CPROB = RAW / "sentinel2_s2cloudless_cloudprob_chipext"
DATABASE = RAW / "DATABASE"
CHIPS_SCL_LOST = RAW / "chips_SCL_LOST"

# The two dataset generations are DIFFERENT CORPORA, not versions of one.
DB_FIRST_ROUND = DATABASE / "FIRST_ROUND_PARQUET"
DB_SLIM = DATABASE / "SLIM_PARQUET"
DB_CHIP_MEMMAP_INDEX = DATABASE / "Sen12Coast_CHIP_MEMMAP_INDEX.parquet"


def require(path: Path, what: str) -> Path:
    """Return ``path``, or raise with a message naming what was missing."""
    if not path.exists():
        raise FileNotFoundError(f"{what} not found: {path}")
    return path


def ensure_out() -> None:
    """Create the writable output tree. Safe to call repeatedly."""
    for directory in (OUT, MANIFESTS, LOGS, QC):
        directory.mkdir(parents=True, exist_ok=True)


def reviewer_bundle(pair_id: str) -> Path:
    """Path to one reviewer bundle.

    Bundles are addressed by pair id. The database's ``asset_path`` column
    holds absolute paths from the machine the review was done on and does not
    survive the move; do not read it.
    """
    return REVIEWER_ASSETS / f"{pair_id}.sqlite"


if __name__ == "__main__":
    rows = [
        ("REPO", REPO), ("OUT", OUT), ("DOCS", DOCS),
        ("REVIEWER_DB", REVIEWER_DB), ("RESOLVED_CSV", RESOLVED_CSV),
        ("LABELS_OUT", LABELS_OUT), ("LABELS_B8", LABELS_B8),
        ("S2_SR", S2_SR), ("DATABASE", DATABASE),
    ]
    width = max(len(name) for name, _ in rows)
    for name, path in rows:
        print(f"{name:<{width}}  {'ok ' if path.exists() else 'MISSING'}  {path}")
