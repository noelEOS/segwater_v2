"""Unit tests for the egress guard.

The policy functions (``classify`` / ``check``) are pure -- they take
``(size, path)`` pairs -- so every rule is exercised here with no VM, no ssh and
no filesystem.

These pin the incident of 2026-07-30: a bare ``rsync -az`` of an SDS results dir
pulled 2.5 GB of ``.gpkg`` shoreline vectors off the VM. The HARD RULE
("only KB-scale CSVs leave", docs/RUNBOOK_INDEX.md) already forbade it, but the
rule lived only in prose across three documents. The first test below is that
incident, reproduced.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts" / "evaluation" / "vm"))

from pull_results import (  # noqa: E402
    DEFAULT_ALLOWED,
    EgressRefused,
    check,
    classify,
    human,
    summarize_rejected,
)

MB = 1024 * 1024
GB = 1024 * MB


def sds_tree():
    """An SDS results dir as it actually is: a few CSVs beside ~14k gpkg."""
    e = [(4_000, "narrabeen/raw/%s_sweep/sweep_metrics.csv" % i) for i in range(18)]
    e += [(220_000, "narrabeen/raw/s%d/shoreline_%d.gpkg" % (i % 18, i))
          for i in range(14_094)]
    return e


def test_the_2026_07_30_incident_is_refused():
    """2.5 GB of .gpkg must not leave the VM, even though none of it is a TIFF."""
    with pytest.raises(EgressRefused) as exc:
        check(sds_tree())
    msg = str(exc.value)
    assert ".gpkg" in msg
    assert "whitelist" in msg
    # the message must name the rule so the reader can go find it
    assert "RUNBOOK_INDEX" in msg


def test_csv_json_tree_passes():
    entries = [(4_000, "demak_gate/summary.csv"),
               (900, "narrabeen/sweep_summary.json"),
               (12_000, "FINDINGS.md")]
    ok, rejected, total = check(entries)
    assert rejected == []
    assert len(ok) == 3
    assert total == 16_900


def test_override_permits_but_still_reports():
    """--i-know-this-is-heavy proceeds, but the violation is still surfaced."""
    ok, rejected, _ = check(sds_tree(), override=True)
    assert len(rejected) == 14_094          # still classified as violations
    assert len(ok) == 18                    # the CSVs still come across
    by_ext = summarize_rejected(rejected)
    assert ".gpkg" in by_ext and by_ext[".gpkg"][0] == 14_094


@pytest.mark.parametrize("name", [
    "run/scene_probability_water.tif",       # the original heavy-TIFF case
    "shorelines.gpkg",                       # the case that actually happened
    "aoi.shp",
    "vectors.geojson",
    "archive.tar.zst",
    "model_last.pth",                        # a checkpoint must never egress
    "notes",                                 # no extension => cannot be vouched for
])
def test_non_whitelisted_extensions_are_rejected(name):
    ok, rejected = classify([(10, name)])
    assert ok == [] and len(rejected) == 1


def test_extension_match_is_case_insensitive():
    """`.CSV` is a CSV; `.GPKG` is still refused."""
    ok, rejected = classify([(10, "a.CSV"), (10, "b.GPKG")])
    assert [p for _, p in ok] == ["a.CSV"]
    assert [p for _, p in rejected] == ["b.GPKG"]


def test_size_cap_refuses_even_an_all_csv_tree():
    """A whitelisted extension is not a licence for unbounded volume."""
    entries = [(60 * MB, "a.csv"), (60 * MB, "b.csv")]
    with pytest.raises(EgressRefused) as exc:
        check(entries)
    assert "cap" in str(exc.value)
    # ...and the override still lets it through
    ok, _, total = check(entries, override=True)
    assert total == 120 * MB


def test_cap_is_measured_on_allowed_files_only():
    """Rejected bytes must not count toward the cap; they are simply not pulled."""
    entries = [(4_000, "a.csv"), (3 * GB, "big.gpkg")]
    with pytest.raises(EgressRefused):
        check(entries)
    ok, rejected, total = check(entries, override=True)
    assert total == 4_000                    # not 3 GB + 4 kB
    assert len(rejected) == 1


def test_custom_allow_list_is_honoured():
    entries = [(10, "a.gpkg"), (10, "b.csv")]
    ok, rejected = classify(entries, allowed=(".gpkg", ".csv"))
    assert len(ok) == 2 and rejected == []


def test_default_allowlist_is_text_artifacts_only():
    """Guard against someone widening the default to a raster/vector format."""
    for ext in DEFAULT_ALLOWED:
        assert ext in {".csv", ".json", ".md", ".txt", ".yaml", ".yml"}, ext


def test_human_readable_sizes():
    assert human(512) == "512 B"
    assert human(2 * MB).startswith("2.0 MB")
    assert human(3 * GB).startswith("3.0 GB")
