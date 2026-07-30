"""Schema contract for the tracked Hampyeong scoring specs.

``score_pairbased_hampyeong.py`` is one config-driven scorer whose *entire*
variation lives in a YAML spec. ``load_spec()`` validates only three things
(``expected_valid_pixels``, ``output``, non-empty ``entries``); every other key
it needs is read later with plain ``[...]`` indexing, deep inside ``audit()`` or
the per-date loop. A spec missing ``slug`` therefore does not fail at load — it
fails with a bare ``KeyError`` after the provenance audit has already printed
"passed", or (worse, for ``training_data``/``lineage_slug`` which have
``.get()`` defaults) produces a mislabelled row that looks fine.

These tests parse every tracked spec and assert the full set of keys the scorer
actually dereferences, with the types it would crash on. They are a
*static* check: they never touch run dirs, rasters or the VM.

Derived by reading score_pairbased_hampyeong.py (load_spec L92-111, audit
L138-163, _tag L166-169, main L172-257) — NOT from documentation.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
VM = REPO / "scripts" / "evaluation" / "vm"
sys.path.insert(0, str(VM))

# ------------------------------------------------------------------ discovery #
# The scorer's specs are not confined to specs/: ship/ keeps its campaign spec
# beside its generators, and analysis/configs/ holds the 5-arm two-lineage spec.
# Discover by SHAPE (a mapping with `entries` + `expected_valid_pixels`) over the
# tracked YAML in those three dirs, so a new spec is covered the day it lands
# instead of the day someone remembers to extend a hardcoded list.
SPEC_SEARCH_DIRS = [
    VM / "specs",
    VM / "ship",
    VM / "analysis" / "configs",
]


def _looks_like_scoring_spec(doc: object) -> bool:
    return (
        isinstance(doc, dict)
        and "entries" in doc
        and "expected_valid_pixels" in doc
    )


def discover_specs() -> list[Path]:
    out = []
    for d in SPEC_SEARCH_DIRS:
        for f in sorted(d.glob("*.yaml")):
            try:
                doc = yaml.safe_load(f.read_text())
            except yaml.YAMLError:
                continue  # not-a-spec YAML is covered by test_every_yaml_parses
            if _looks_like_scoring_spec(doc):
                out.append(f)
    return out


SPECS = discover_specs()
SPEC_IDS = [p.name for p in SPECS]


def test_specs_were_discovered():
    """Pin the premise: if the layout moves, this file must not silently pass."""
    assert len(SPECS) >= 6, f"expected the tracked scoring specs, found {SPEC_IDS}"
    assert "spec_hampyeong_ship.yaml" in SPEC_IDS
    assert "hampyeong_swin.yaml" in SPEC_IDS


@pytest.mark.parametrize("path", SPECS, ids=SPEC_IDS)
def test_spec_parses_as_a_mapping(path: Path):
    doc = yaml.safe_load(path.read_text())
    assert isinstance(doc, dict), "a spec must be a YAML mapping"


# ------------------------------------------------------------ load_spec gate #
@pytest.mark.parametrize("path", SPECS, ids=SPEC_IDS)
def test_load_spec_accepts_it(path: Path):
    """The three checks load_spec enforces itself, exercised through load_spec.

    Imported by path so no editable install is needed; the scorer's own heavy
    imports (numpy/pandas/yaml) are present in the test env, and the torch/
    rasterio-dependent modules are imported lazily inside main(), after the
    repo path is prepended — so importing the module is cheap and safe.
    """
    from score_pairbased_hampyeong import load_spec  # noqa: PLC0415

    spec = load_spec(path)
    # load_spec fills these; a missing default here means output columns move.
    for k in ("repo", "nas_root", "pair_root", "variant_column",
              "lineage_slug", "training_data"):
        assert k in spec, f"load_spec must default {k}"


@pytest.mark.parametrize("path", SPECS, ids=SPEC_IDS)
def test_required_top_level_keys_and_types(path: Path):
    doc = yaml.safe_load(path.read_text())

    # int(spec["expected_valid_pixels"]) -- must not be a list/dict/None.
    assert "expected_valid_pixels" in doc
    evp = doc["expected_valid_pixels"]
    assert isinstance(evp, int) and not isinstance(evp, bool), (
        f"expected_valid_pixels must be an int, got {type(evp).__name__}"
    )
    assert evp > 0

    # Path(spec["output"]) -- and the scorer refuses to clobber it, so it must
    # be a real single path string, not a list of candidates.
    assert isinstance(doc["output"], str) and doc["output"], "output must be a path str"
    assert doc["output"].endswith(".csv"), "output is a CSV path"

    assert isinstance(doc["entries"], list) and doc["entries"], "entries non-empty list"

    # Optional-with-default keys: absent is fine (load_spec fills them), but a
    # present value of the wrong type crashes later (str()/bool()/Path()).
    for key, typ in (("repo", str), ("nas_root", str), ("pair_root", str),
                     ("lineage_slug", str), ("training_data", str)):
        if key in doc:
            assert isinstance(doc[key], typ), f"{key} must be {typ.__name__}"
    if "variant_column" in doc:
        assert isinstance(doc["variant_column"], bool), "variant_column must be a bool"


@pytest.mark.parametrize("path", SPECS, ids=SPEC_IDS)
def test_every_entry_carries_the_keys_the_scorer_dereferences(path: Path):
    """`label`/`slug`/`seed`/`run_dir`/`checkpoint` are indexed, not `.get()`.

    A missing one is a KeyError raised AFTER `audit()` prints "Provenance audit
    passed" (slug/seed) or during it (label/run_dir/checkpoint) -- never at load.
    """
    doc = yaml.safe_load(path.read_text())
    for i, e in enumerate(doc["entries"]):
        where = f"{path.name} entry[{i}]"
        assert isinstance(e, dict), f"{where}: entry must be a mapping"

        for key in ("label", "slug", "run_dir", "checkpoint"):
            assert key in e, f"{where}: missing required key {key!r}"
            assert isinstance(e[key], str) and e[key], f"{where}: {key} must be a non-empty str"

        # seed goes into an f-string and a `seed` output column; int by convention
        # in every tracked spec (the scorer would accept a str, but a mixed-type
        # column silently breaks downstream groupbys).
        assert "seed" in e, f"{where}: missing seed"
        assert isinstance(e["seed"], int) and not isinstance(e["seed"], bool), (
            f"{where}: seed must be an int, got {type(e['seed']).__name__}"
        )
        assert e["seed"] in (19, 42, 58), f"{where}: unexpected seed {e['seed']}"

        # Per-entry overrides: optional, but must be str when present.
        for key in ("lineage_slug", "training_data", "variant"):
            if key in e:
                assert isinstance(e[key], str) and e[key], f"{where}: {key} must be a non-empty str"

        # `checkpoint` is compared verbatim to run_config's checkpoint_path,
        # which is repo-RELATIVE in every audited run; an absolute path here
        # can never match and the audit fails with a confusing diff.
        assert not e["checkpoint"].startswith("/"), (
            f"{where}: checkpoint must be relative to `repo`"
        )


@pytest.mark.parametrize("path", SPECS, ids=SPEC_IDS)
def test_variant_column_and_per_entry_variant_agree(path: Path):
    """`variant` only reaches the output when variant_column is true.

    _tag() and the model-name builder both gate on `variant_column`; an entry
    carrying `variant` under a spec without the flag loses that distinction
    silently, which is exactly the best-vs-last collapse the gate exists to
    prevent. The reverse (flag on, no variant) is legal (`.get(..., "")`).
    """
    doc = yaml.safe_load(path.read_text())
    variant_column = bool(doc.get("variant_column", False))
    if not variant_column:
        offenders = [i for i, e in enumerate(doc["entries"]) if e.get("variant")]
        assert not offenders, (
            f"{path.name}: entries {offenders} set `variant` but variant_column is "
            "false -- the variant would be dropped from `model` and `_tag`"
        )


@pytest.mark.parametrize("path", SPECS, ids=SPEC_IDS)
def test_entries_are_mutually_distinguishable(path: Path):
    """The scorer requires distinct checkpoints; the output rows must also be
    distinct in (variant, model, seed) or two arms collapse into one label."""
    doc = yaml.safe_load(path.read_text())
    variant_column = bool(doc.get("variant_column", False))
    file_lineage = doc.get("lineage_slug", "pairbased")

    ckpts = [e["checkpoint"] for e in doc["entries"]]
    dupes = {c for c in ckpts if ckpts.count(c) > 1}
    assert not dupes, f"{path.name}: audit() would raise on shared checkpoints: {dupes}"

    keys = []
    for e in doc["entries"]:
        lineage = e.get("lineage_slug", file_lineage)
        model = f"{e['slug']}_s{e['seed']}_{lineage}"
        if variant_column and e.get("variant"):
            model += f"_{e['variant']}"
        keys.append(model)
    dupe_keys = {k for k in keys if keys.count(k) > 1}
    assert not dupe_keys, f"{path.name}: duplicate `model` labels: {dupe_keys}"


@pytest.mark.parametrize("path", SPECS, ids=SPEC_IDS)
def test_glob_run_dirs_are_anchorable(path: Path):
    """A `glob:` run_dir must contain a wildcard for runsel to anchor on.

    `resolve_glob_spec` substitutes the UTC stamp at the first `*`; a `glob:`
    value with no `*` is a silent no-match (0 hits -> RunDirError) at run time.
    """
    from runsel import resolve_glob_spec  # noqa: PLC0415

    doc = yaml.safe_load(path.read_text())
    for i, e in enumerate(doc["entries"]):
        rd = e["run_dir"]
        if not rd.startswith("glob:"):
            continue
        pattern = rd[len("glob:"):]
        assert "*" in pattern, f"{path.name} entry[{i}]: `glob:` with no wildcard"
        # And it must be resolvable against an empty tree without raising
        # (0 hits is fine here; a malformed pattern is not).
        assert resolve_glob_spec(REPO / "tests", pattern) == []


def test_every_tracked_yaml_in_the_spec_dirs_parses():
    """Nothing under the spec dirs may be unparseable YAML, spec or not.

    The non-spec files here are inference-sweep / eval configs read by other
    tools; a YAML syntax error in any of them is a VM-side failure that a Mac
    test can catch for free.
    """
    bad = []
    for d in SPEC_SEARCH_DIRS:
        for f in sorted(d.glob("*.yaml")):
            try:
                yaml.safe_load(f.read_text())
            except yaml.YAMLError as exc:
                bad.append(f"{f.name}: {exc}")
    assert not bad, "unparseable YAML:\n" + "\n".join(bad)
