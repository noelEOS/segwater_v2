"""Tests for the SDS scorer's per-site required-flag registry and glob hygiene.

Both units under test live in the nested SDS_Benchmark_slim tree, which is not an
installable package and whose modules import the heavy `eda_coastsat` stack
(geopandas, rasterio, coastsat, utils_modified) at module level. The test env has
none of those, so a plain `import sds_core` cannot work here.

Rather than reimplement the logic (which would test a copy, not the shipped
code), each helper below loads the REAL source file and executes only the
top-level definitions it needs, selected by name via `ast`. The bytes executed
are the shipped bytes; only the unrelated heavy imports are skipped. If the full
`eda_coastsat` env is ever used to run these tests, `_load_names` still executes
the same definitions, so the tests do not diverge.
"""

import ast
import os

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SDS_DIR = os.path.join(REPO_ROOT, "SDS_Benchmark_slim", "scripts", "sds")
SDS_CORE = os.path.join(SDS_DIR, "sds_core.py")
RUN_FROM_RASTERS = os.path.join(SDS_DIR, "run_sds_from_rasters.py")


def _load_names(path, names):
    """Exec the named top-level defs/assignments from `path` in a fresh namespace.

    Keeps the real source as the thing under test while skipping module-level
    imports of the heavy SDS stack, which is unavailable in the test env.
    """
    with open(path, "r") as f:
        src = f.read()
    tree = ast.parse(src, filename=path)
    wanted = set(names)
    keep = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if node.name in wanted:
                keep.append(node)
        elif isinstance(node, ast.Assign):
            targets = [t.id for t in node.targets if isinstance(t, ast.Name)]
            if wanted.intersection(targets):
                keep.append(node)
    found = set()
    for node in keep:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            found.add(node.name)
        else:
            found.update(t.id for t in node.targets if isinstance(t, ast.Name))
    missing = wanted - found
    assert not missing, f"not found at top level of {path}: {sorted(missing)}"
    ns = {"os": os, "__name__": "_sds_under_test"}
    exec(compile(ast.Module(body=keep, type_ignores=[]), path, "exec"), ns)
    return ns


@pytest.fixture(scope="module")
def core():
    return _load_names(
        SDS_CORE,
        ["SITE_REQUIRED_FLAGS", "required_flags", "check_required_flags"])


@pytest.fixture(scope="module")
def rasters_mod():
    return _load_names(RUN_FROM_RASTERS, ["drop_appledouble"])


# --- registry contents -----------------------------------------------------
def test_registry_covers_the_four_benchmark_sites(core):
    assert set(core["SITE_REQUIRED_FLAGS"]) == {
        "NARRABEEN", "DUCK", "TORREYPINES", "TRUCVERT"}


def test_required_flags_values(core):
    rf = core["required_flags"]
    assert rf("NARRABEEN") == {}
    assert rf("DUCK") == {"keep_top_k": 999}
    assert rf("TORREYPINES") == {"use_min_chainage_length": False}
    assert rf("TRUCVERT") == {"use_min_chainage_length": False}


def test_required_flags_is_case_insensitive(core):
    assert core["required_flags"]("duck") == {"keep_top_k": 999}


def test_required_flags_returns_a_copy(core):
    """Mutating the returned dict must not corrupt the registry."""
    rf = core["required_flags"]
    got = rf("DUCK")
    got["keep_top_k"] = 1
    assert rf("DUCK") == {"keep_top_k": 999}


def test_required_flags_unknown_site_lists_known_sites(core):
    with pytest.raises(ValueError) as exc:
        core["required_flags"]("NARABEEN")  # typo
    msg = str(exc.value)
    assert "NARABEEN" in msg
    for site in ("NARRABEEN", "DUCK", "TORREYPINES", "TRUCVERT"):
        assert site in msg


# --- check_required_flags --------------------------------------------------
DEFAULTS = {"keep_top_k": 1, "use_min_chainage_length": True}


def test_narrabeen_clean_at_defaults(core):
    assert core["check_required_flags"]("NARRABEEN", **DEFAULTS) == []


def test_duck_violated_at_default_keep_top_k(core):
    v = core["check_required_flags"]("DUCK", **DEFAULTS)
    assert len(v) == 1
    assert "keep_top_k" in v[0]
    assert "999" in v[0]
    assert "--keep-top-k 999" in v[0]


def test_duck_satisfied_at_999(core):
    assert core["check_required_flags"](
        "DUCK", keep_top_k=999, use_min_chainage_length=True) == []


@pytest.mark.parametrize("site", ["TRUCVERT", "TORREYPINES"])
def test_min_chainage_sites_violated_when_rule_on(core, site):
    v = core["check_required_flags"](
        site, keep_top_k=1, use_min_chainage_length=True)
    assert len(v) == 1
    assert "use_min_chainage_length" in v[0]
    assert "--no-min-chainage-length" in v[0]
    assert site in v[0]


@pytest.mark.parametrize("site", ["TRUCVERT", "TORREYPINES"])
def test_min_chainage_sites_satisfied_when_rule_off(core, site):
    assert core["check_required_flags"](
        site, keep_top_k=1, use_min_chainage_length=False) == []


def test_check_required_flags_unknown_site_raises(core):
    with pytest.raises(ValueError):
        core["check_required_flags"]("ATLANTIS", **DEFAULTS)


def test_extra_non_required_flags_are_not_flagged(core):
    """Only the site's requirements are checked; other values are free."""
    assert core["check_required_flags"](
        "NARRABEEN", keep_top_k=999, use_min_chainage_length=False) == []


# --- AppleDouble filter ----------------------------------------------------
def test_drop_appledouble_removes_stubs_only(rasters_mod):
    drop = rasters_mod["drop_appledouble"]
    paths = [
        "/d/S1_20190101_probability_water.tif",
        "/d/._S1_20190101_probability_water.tif",
        "/d/sub/._S1_20190202_probability_water.tif",
        "/d/sub/S1_20190202_probability_water.tif",
    ]
    kept, n = drop(paths)
    assert n == 2
    assert kept == ["/d/S1_20190101_probability_water.tif",
                    "/d/sub/S1_20190202_probability_water.tif"]


def test_drop_appledouble_noop_on_clean_list(rasters_mod):
    drop = rasters_mod["drop_appledouble"]
    paths = ["/d/a.tif", "/d/b.tif"]
    kept, n = drop(paths)
    assert n == 0
    assert kept == paths


def test_drop_appledouble_preserves_order_and_underscore_names(rasters_mod):
    """A leading single underscore or an interior '._' is NOT a stub."""
    drop = rasters_mod["drop_appledouble"]
    paths = ["/d/_a.tif", "/d/b._c.tif", "/d/._d.tif"]
    kept, n = drop(paths)
    assert n == 1
    assert kept == ["/d/_a.tif", "/d/b._c.tif"]


def test_drop_appledouble_empty(rasters_mod):
    assert rasters_mod["drop_appledouble"]([]) == ([], 0)


# --- exit-code semantics on failed scene extraction ------------------------
# main() cannot be imported here (matplotlib/pandas-heavy module import), so the
# contract is exercised on the exact code shape main() uses: failures collected
# per threshold tag -> SystemExit after all writes; no failures -> falls through.
def _finish(failed_by_tag):
    """Mirror of main()'s terminal block (see run_sds_from_rasters.py)."""
    n_failed_total = sum(len(v) for v in failed_by_tag.values())
    if failed_by_tag:
        raise SystemExit(
            "FAILED: %d scene extraction(s) failed across %d threshold(s); "
            "see failed_scenes_by_threshold in sweep_summary.json"
            % (n_failed_total, len(failed_by_tag)))
    return 0


def test_terminal_block_source_matches_the_mirror():
    """Guard the mirror above against drift in the shipped source."""
    with open(RUN_FROM_RASTERS, "r") as f:
        src = f.read()
    assert 'raise SystemExit(\n            "FAILED: %d scene extraction(s) ' in src
    assert '"see failed_scenes_by_threshold in sweep_summary.json"' in src
    assert "% (n_failed_total, len(failed_by_tag)))" in src
    # the raise must come after sweep_summary.json is written
    assert src.index('"sweep_summary.json"') < src.index('raise SystemExit(\n')


def test_clean_run_exits_zero():
    assert _finish({}) == 0


def test_failed_run_exits_nonzero_with_counts():
    with pytest.raises(SystemExit) as exc:
        _finish({"0p50": ["sceneA"], "0p60": ["sceneA", "sceneB"]})
    # SystemExit carrying a string => non-zero exit status with that message
    assert exc.value.code != 0
    msg = str(exc.value.code)
    assert msg.startswith("FAILED: 3 scene extraction(s) failed across 2 threshold(s)")
    assert "failed_scenes_by_threshold" in msg


def test_consume_collects_failures_from_both_paths():
    """_consume is the single fold point for the serial and parallel paths."""
    with open(RUN_FROM_RASTERS, "r") as f:
        src = f.read()
    # serial branch and pool branch both funnel through _consume
    assert "_consume(_process_threshold(cfg))" in src
    assert "_consume(fut.result())" in src
    # and _consume records failures before the early return on error
    consume = src[src.index("def _consume("):src.index("# --- run the per-threshold")]
    assert consume.index("failed_by_tag[tag] = failed") < consume.index(
        'if result["error"] is not None:')
