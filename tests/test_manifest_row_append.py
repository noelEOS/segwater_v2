"""Regression PINS for src/utils/inference_outputs.py -- documenting limits, not fixing them.

`append_manifest_row` and `prepare_output_paths` are on the production inference
hot path. Two of their behaviours were named as hazards during the 2026-07
ship-campaign retrospective:

  1. **Header duplication is possible under concurrency.** The header decision is
     `write_header = not manifest_path.exists()` taken OUTSIDE the append-mode
     open, so two processes that both observe no file both write a header. The
     sequential case -- the only one that occurs in practice -- is correct: one
     header, N rows.

  2. **The overwrite guard is per-SCENE, not per-RUN.** `prepare_output_paths`
     raises `FileExistsError` when `{root}/{run_id}/{scene_id}/` exists, so a
     second sweep into an existing run dir happily *adds* scenes to it and only
     refuses the scenes that already ran. There is no run-level "this run dir is
     already populated" gate.

**This file deliberately does NOT change src/.** Decided during the hardening
pass (plan "Decided NOT to do"): the concurrent scenario needs the same
`run_name` AND the same UTC second on two writers, which is practically
unreachable for the sweep driver (one process per group, distinct run names);
rewriting the append path risks the numeric-output-stability guarantee for no
reachable bug. The remedy shipped instead is `completion.py` -- the run manifest
is not the completion signal; the `*_probability_water.tif` count is.

So these tests pin what IS. If a future change makes any of them fail, that is
the signal to update the pin *deliberately*, having read the note above -- not
to "fix the test".

Import note: `src.utils.inference_outputs` imports rasterio, torch and omegaconf
at module scope; the test env has only omegaconf and numpy. The functions under
test (`append_manifest_row`, `to_json_safe`, `prepare_output_paths`) never call
into rasterio or torch, so the module is loaded by PATH with minimal stubs in
`sys.modules` for the two missing deps. That keeps the REAL code under test
(no reimplementation) with no editable install and no heavy deps.
"""

from __future__ import annotations

import csv
import importlib.util
import sys
import types
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO / "src" / "utils" / "inference_outputs.py"


# --------------------------------------------------------------------------- #
# module loading (stub only what the tested functions never touch)
# --------------------------------------------------------------------------- #
def _load_inference_outputs():
    """Import src/utils/inference_outputs.py by path with rasterio/torch stubbed.

    The stubs are installed only if the real package is absent, so a fuller env
    (the VM's inference env) exercises the genuine imports.
    """
    added = []
    if "rasterio" not in sys.modules:
        try:
            import rasterio  # noqa: F401
        except ImportError:
            stub = types.ModuleType("rasterio")
            stub.open = lambda *a, **k: (_ for _ in ()).throw(
                RuntimeError("rasterio stub: read_raster_metadata is out of scope here")
            )
            sys.modules["rasterio"] = stub
            added.append("rasterio")
    if "torch" not in sys.modules:
        try:
            import torch  # noqa: F401
        except ImportError:
            stub = types.ModuleType("torch")

            class _Tensor:  # to_json_safe does isinstance(value, torch.Tensor)
                pass

            stub.Tensor = _Tensor
            sys.modules["torch"] = stub
            added.append("torch")

    spec = importlib.util.spec_from_file_location(
        "_inference_outputs_under_test", MODULE_PATH
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


io_mod = _load_inference_outputs()

FIELDNAMES = [
    "run_id", "scene_id", "status", "input_path", "probability_memmap",
    "probability_geotiff", "binary_mask_geotiff", "shoreline_geojson",
    "metadata_json", "width", "height", "crs", "threshold", "checkpoint_path",
    "created_utc", "elapsed_minutes",
]


def a_row(scene: str = "S1_scene_a", **over):
    row = {
        "run_id": "demak_gate_x",
        "scene_id": scene,
        "status": "success",
        "input_path": f"/data/{scene}.tif",
        "width": 1000,
        "height": 2000,
        "crs": "EPSG:32649",
        "threshold": 0.5,
        "checkpoint_path": "outputs/mx630_stage2/s42/model_last.pth",
        "created_utc": "2026-07-30T00:00:00Z",
        "elapsed_minutes": "1.2345",
    }
    row.update(over)
    return row


def read_lines(p: Path) -> list[str]:
    return p.read_text(encoding="utf-8").splitlines()


# --------------------------------------------------------------------------- #
# 1. sequential appends -- the reachable case, and it is CORRECT
# --------------------------------------------------------------------------- #
def test_sequential_appends_write_exactly_one_header(tmp_path):
    m = tmp_path / "run_manifest.csv"
    for i in range(4):
        io_mod.append_manifest_row(m, a_row(f"scene_{i}"))

    lines = read_lines(m)
    assert lines[0] == ",".join(FIELDNAMES)
    assert sum(1 for ln in lines if ln.startswith("run_id,")) == 1, (
        "sequential appends must produce exactly one header"
    )
    rows = list(csv.DictReader(m.open(newline="", encoding="utf-8")))
    assert [r["scene_id"] for r in rows] == [f"scene_{i}" for i in range(4)]


def test_append_creates_parent_dirs(tmp_path):
    m = tmp_path / "nested" / "deeper" / "run_manifest.csv"
    io_mod.append_manifest_row(m, a_row())
    assert m.exists()


def test_column_set_is_fixed_and_extra_keys_are_dropped_silently(tmp_path):
    """PIN: unknown keys vanish without warning; missing keys become "".

    `writerow({field: safe_row.get(field, "") for field in fieldnames})` means a
    typo'd key (`probability_geotif`) is silently lost rather than raising -- so
    a column can go quietly empty across a whole run. Pinned, not fixed.
    """
    m = tmp_path / "run_manifest.csv"
    io_mod.append_manifest_row(
        m, a_row(probability_geotif="/typo/path.tif", nonsense_key=123)
    )
    rows = list(csv.DictReader(m.open(newline="", encoding="utf-8")))
    assert list(rows[0]) == FIELDNAMES, "column set must be exactly the fixed list"
    assert rows[0]["probability_geotiff"] == "", "typo'd key silently dropped (pinned)"
    assert "nonsense_key" not in rows[0]


def test_values_are_json_safed_paths_become_strings(tmp_path):
    m = tmp_path / "run_manifest.csv"
    io_mod.append_manifest_row(
        m, a_row(probability_geotiff=Path("/runs/x/scene/scene_probability_water.tif"))
    )
    rows = list(csv.DictReader(m.open(newline="", encoding="utf-8")))
    assert rows[0]["probability_geotiff"] == "/runs/x/scene/scene_probability_water.tif"


# --------------------------------------------------------------------------- #
# 2. the header race -- PINNED AS A KNOWN LIMITATION (see module docstring)
# --------------------------------------------------------------------------- #
def test_header_decision_is_taken_before_the_append_open(tmp_path, monkeypatch):
    """PIN the race MECHANISM: two writers that both see no file both add a header.

    Simulated by making the FIRST writer's existence probe report "absent" for a
    file that a concurrent writer has already created (exactly what a
    lost/stale stat looks like from the loser of the race). This is the real
    `append_manifest_row` code path -- only `Path.exists` is perturbed.

    Consequence pinned here: the resulting CSV has an embedded header row, so a
    naive `len(rows)` over-counts by one and `csv.DictReader` yields a row whose
    every field equals its own column name. NOT FIXED -- src/ is the inference
    hot path and the scenario needs two writers on the same run_name in the same
    second. Use completion.py (raster count), not the manifest, as the
    completion signal.
    """
    m = tmp_path / "run_manifest.csv"
    io_mod.append_manifest_row(m, a_row("scene_0"))  # writer A: header + row

    real_exists = Path.exists

    def blind_exists(self, *a, **k):
        if self == m:
            return False  # writer B never saw A's file
        return real_exists(self, *a, **k)

    monkeypatch.setattr(Path, "exists", blind_exists)
    io_mod.append_manifest_row(m, a_row("scene_1"))  # writer B: header AGAIN + row
    monkeypatch.undo()

    lines = read_lines(m)
    assert sum(1 for ln in lines if ln.startswith("run_id,")) == 2, (
        "premise broken: the header decision is no longer racy -- if this is a "
        "deliberate fix, update this pin and the module docstring"
    )

    rows = list(csv.DictReader(m.open(newline="", encoding="utf-8")))
    assert len(rows) == 3, "3 data rows: scene_0, the stray header, scene_1"
    stray = rows[1]
    assert stray["scene_id"] == "scene_id", (
        "the duplicated header parses as a data row whose values are the column names"
    )
    assert [r["scene_id"] for r in rows] == ["scene_0", "scene_id", "scene_1"]


def test_no_atomicity_or_locking_is_claimed(tmp_path):
    """PIN: append is a plain `open(..., 'a')` -- no lock, no .tmp+replace.

    Deliberate (plan: "Decided NOT to do -- lockfile/flock"). The hardening
    remedy for shared artifacts is naming.atomic_write, which is used for the
    DERIVED tables (build_ship_manifest, consolidate_ship_results,
    collate_sds_arms2) -- not for this per-scene append, which must stay
    append-only so a crashed sweep keeps the scenes it did finish.
    """
    src = MODULE_PATH.read_text()
    fn = src.split("def append_manifest_row", 1)[1].split("\ndef ", 1)[0]
    assert 'open("a"' in fn.replace("'", '"'), "still an append-mode open"
    for forbidden in ("flock", "lock", "os.replace", ".tmp"):
        assert forbidden not in fn, (
            f"append_manifest_row gained {forbidden!r} -- if locking/atomicity was "
            "added deliberately, update this pin and the module docstring"
        )


# --------------------------------------------------------------------------- #
# 3. the overwrite guard is per-SCENE, not per-RUN
# --------------------------------------------------------------------------- #
def make_cfg(root_dir: Path, scene_stem: str, run_name: str, overwrite: bool):
    """Build the cfg tree prepare_output_paths reads.

    A real `DictConfig` is used (not a stub) because the function mixes attribute
    access with one `OmegaConf.select(cfg, "inference.post_processing.shoreline.
    output_format")`, and select()'s missing-key semantics are load-bearing.
    """
    from omegaconf import OmegaConf  # noqa: PLC0415

    return OmegaConf.create({
        "inference": {
            "checkpoint_path": "outputs/mx630_stage2/s42/model_step23930_last.pth",
            "data": {"input_image": f"/data/{scene_stem}.tif"},
            "output": {
                "root_dir": str(root_dir),
                "run_name": run_name,
                "overwrite": overwrite,
            },
            "post_processing": {"shoreline": {"output_format": "gpkg"}},
        },
        "model": {"arch": "upernet", "encoder_name": "tu-swin_base_patch4_window7_224"},
    })


def test_existing_scene_dir_raises_when_overwrite_false(tmp_path):
    cfg = make_cfg(tmp_path, "S1_scene_a", "run_x", overwrite=False)
    first = io_mod.prepare_output_paths(cfg)
    assert first.scene_dir.exists()

    with pytest.raises(FileExistsError) as e:
        io_mod.prepare_output_paths(cfg)
    assert "overwrite=False" in str(e.value)


def test_guard_is_per_scene_so_a_new_scene_joins_an_existing_run_dir(tmp_path):
    """PIN: no run-level gate. Scene B lands inside run_x even though run_x is
    already populated by scene A -- i.e. two different sweeps writing the same
    `run_name` MERGE into one run dir instead of being refused.

    This is why run-dir completeness must be checked by counting rasters against
    an expected scene count (completion.EXPECTED_SCENES), not by asking whether
    the run dir exists. Documented, not fixed: legitimate resumed sweeps rely on
    exactly this behaviour.
    """
    cfg_a = make_cfg(tmp_path, "S1_scene_a", "run_x", overwrite=False)
    cfg_b = make_cfg(tmp_path, "S1_scene_b", "run_x", overwrite=False)

    pa = io_mod.prepare_output_paths(cfg_a)
    pb = io_mod.prepare_output_paths(cfg_b)  # must NOT raise

    assert pa.run_dir == pb.run_dir, "same run_name -> same run dir"
    assert pa.scene_dir != pb.scene_dir
    assert sorted(p.name for p in pa.run_dir.iterdir()) == ["S1_scene_a", "S1_scene_b"]


def test_overwrite_true_reuses_the_scene_dir_without_clearing_it(tmp_path):
    """PIN: overwrite=True does not empty the scene dir -- stale products from a
    previous run survive alongside the new ones (mkdir(exist_ok=True) only)."""
    cfg = make_cfg(tmp_path, "S1_scene_a", "run_x", overwrite=True)
    paths = io_mod.prepare_output_paths(cfg)
    stale = paths.scene_dir / "S1_scene_a_probability_water.tif"
    stale.write_bytes(b"stale")

    again = io_mod.prepare_output_paths(cfg)
    assert again.scene_dir == paths.scene_dir
    assert stale.exists(), "overwrite=True leaves prior files in place (pinned)"


def test_run_level_artifacts_are_shared_across_scenes_of_one_run(tmp_path):
    """The manifest/summary/config paths are RUN-level, which is what makes the
    append path (and its header decision) shared state between scene writers."""
    cfg_a = make_cfg(tmp_path, "S1_scene_a", "run_x", overwrite=False)
    cfg_b = make_cfg(tmp_path, "S1_scene_b", "run_x", overwrite=False)
    pa = io_mod.prepare_output_paths(cfg_a)
    pb = io_mod.prepare_output_paths(cfg_b)

    assert pa.run_manifest == pb.run_manifest == pa.run_dir / "run_manifest.csv"
    assert pa.run_summary == pb.run_summary
    assert pa.run_config == pb.run_config


def test_run_id_is_sanitized_from_run_name(tmp_path):
    """Pin the sanitisation, because it decides which run dirs collide: `.` -> `p`
    and any other non-[A-Za-z0-9_-] run collapses to a single `_`."""
    cfg = make_cfg(tmp_path, "S1_scene_a", "run x/thr0.5", overwrite=False)
    paths = io_mod.prepare_output_paths(cfg)
    assert paths.run_id == "run_x_thr0p5"
    assert paths.run_dir.name == "run_x_thr0p5"


def test_probability_raster_name_matches_the_completion_glob():
    """The completion contract counts `*/*_probability_water.tif`; that filename
    is produced HERE. If this test fails, completion.PROBABILITY_GLOB is stale."""
    sys.path.insert(0, str(REPO / "scripts" / "evaluation" / "vm"))
    from completion import PROBABILITY_GLOB  # noqa: PLC0415

    assert PROBABILITY_GLOB.endswith("_probability_water.tif")
    src = MODULE_PATH.read_text()
    assert '_probability_water.tif"' in src, (
        "inference_outputs no longer emits *_probability_water.tif -- "
        "completion.PROBABILITY_GLOB must be updated in lockstep"
    )
