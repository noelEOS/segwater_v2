"""The evaluation-side S1 re-quantizer, and the config contract around it.

The training corpus is served on a quantized dB lattice; the evaluation rasters
are continuous dB. The normalization constants describe the lattice, so
inference has to cross it before the z-score. These tests pin the arithmetic to
the two sources of truth -- the GEE export op that produced the corpus and the
memmap builder that decoded it -- rather than to a transcription of either.

Export (gee-search-agent ``export/products.py``), from linear power::

    code = clamp(round((10*log10(linear) + offset_db) / step_db), 1, 65535)

Decode (``scripts/dataset_v3/build_memmaps_v3.py``)::

    dB = code * STEP_DB + FLOOR_DB          # 0.05, -55.0

Inference inputs are already dB, so the re-quantizer enters the export chain
after the ``10*log10`` and must invert the decode exactly.
"""

from __future__ import annotations

import ast
import glob
from pathlib import Path

import numpy as np
import pytest
import yaml
from omegaconf import OmegaConf

REPO = Path(__file__).resolve().parents[1]

# The corpus constants, from docs/dataset_v3/NORM_CONSTANTS_V3.md.
STEP_DB = 0.05
FLOOR_DB = -55.0
FLOOR_CODE = 1                      # code 0 is reserved for nodata
FLOOR_VALUE = FLOOR_CODE * STEP_DB + FLOOR_DB        # -54.95 dB
V3_MEANS = [-15.435106, -24.856323]
V3_STDS = [6.517790, 9.117346]


def _load_functions():
    """Execute the two builders straight out of run_inference.py.

    The module imports torch at load time, which the test env has no reason to
    carry, so the functions are lifted by AST rather than imported. Testing a
    copy of the arithmetic would defeat the point of the test.
    """
    src = (REPO / "scripts" / "run_inference.py").read_text()
    want = {"build_s1_encoding", "build_inference_transform"}
    ns: dict = {"np": np}
    for node in ast.parse(src).body:
        if isinstance(node, ast.FunctionDef) and node.name in want:
            exec(compile(ast.Module([node], []), "<run_inference>", "exec"), ns)
    missing = want - ns.keys()
    assert not missing, f"not found in run_inference.py: {sorted(missing)}"
    return ns


FUNCS = _load_functions()


def _cfg(encoding):
    cfg = OmegaConf.load(REPO / "configs" / "inference.yaml")
    cfg.inference.data.s1_encoding = encoding
    return cfg


def _requantize(encoding="q55s005"):
    return FUNCS["build_s1_encoding"](_cfg(encoding))


# --- the arithmetic -------------------------------------------------------

def test_inverts_the_training_decode_exactly():
    """Every decoded training value must survive the round trip untouched.

    This is the check that ties inference to the corpus: if it holds for all
    codes, the re-quantizer maps continuous dB onto exactly the lattice the
    constants were computed over.
    """
    codes = np.arange(FLOOR_CODE, 20_000, dtype=np.float32)
    decoded = codes * np.float32(STEP_DB) + np.float32(FLOOR_DB)
    assert np.array_equal(_requantize()(decoded), decoded)


def test_sub_floor_pins_to_code_one_not_nodata():
    """The export clamps at code 1; code 0 means nodata and must never appear.

    Evaluation rasters go below the floor (VH observed at -63.5 dB) where the
    corpus cannot, so this clamp is the load-bearing half of the transform.
    """
    below = np.array([FLOOR_VALUE - 1e-3, -55.0, -63.457, -90.0], dtype=np.float32)
    out = _requantize()(below)
    assert np.allclose(out, FLOOR_VALUE)
    # Restating the floor as a code proves 0 is unreachable from real data.
    codes = np.round((out + 55.0) / STEP_DB)
    assert np.array_equal(codes, np.full(below.shape, FLOOR_CODE))


def test_bright_pixels_pass_through():
    """The upper clamp is 65535 (+3221 dB here), so it never binds.

    The corpus carries codes above +25 dB -- build_histograms.py counts them and
    NORM_CONSTANTS_V3.md reports 50,794 such VV pixels. +25 is where the
    histogram folds, not where the data stops, so inference must not clip there.
    """
    bright = np.array([25.0, 30.0, 38.912, 100.0], dtype=np.float32)
    out = _requantize()(bright)
    assert np.all(np.abs(out - bright) <= STEP_DB / 2 + 1e-5)
    assert out.max() > 25.0


def test_snapping_rounds_and_does_not_truncate():
    """`.round()`, not truncation.

    The superseded export used toUint16 and truncated, which put code 1 over
    [-40, -37) and inflated the apparent sub-floor population. Rounding keeps a
    code centred on its value, bounding the error at half a step.
    """
    rng = np.random.default_rng(0)
    x = rng.uniform(-54.0, 20.0, 10_000).astype(np.float32)
    err = np.abs(_requantize()(x) - x)
    # Half a step, plus float32 slack: the lattice value is reconstructed as
    # code * 0.05 - 55, and neither operand is exact in binary32.
    assert err.max() <= STEP_DB / 2 + 1e-5
    # A truncating implementation is biased low; a rounding one is not.
    assert abs(float(np.mean(_requantize()(x) - x))) < STEP_DB / 20


# --- the config contract --------------------------------------------------

def test_encoding_key_is_mandatory():
    """A config that omits the key must fail, never default to a lineage.

    Checkpoints record no lineage, so a default here would silently pick one --
    which is how the constants drifted before.
    """
    cfg = OmegaConf.load(REPO / "configs" / "inference.yaml")
    with pytest.raises(Exception):
        _ = cfg.inference.data.s1_encoding


def test_unknown_encoding_is_rejected():
    with pytest.raises(ValueError, match="s1_encoding"):
        FUNCS["build_s1_encoding"](_cfg("q55"))


def test_none_passes_values_through():
    assert FUNCS["build_s1_encoding"](_cfg("none")) is None


def test_shipped_constants_are_the_v3_set_in_both_copies():
    """means/stds and the padding fills are one unit and must agree.

    padding.channel_fill_values is a second copy of the means; if it lags, edge
    padding stops normalizing to ~zero and nothing reports it.
    """
    d = OmegaConf.load(REPO / "configs" / "inference.yaml").inference.data
    assert np.allclose(list(d.normalization.means), V3_MEANS)
    assert np.allclose(list(d.normalization.stds), V3_STDS)
    assert np.allclose(list(d.padding.channel_fill_values), V3_MEANS)


def test_transform_applies_requantization_before_normalizing():
    """Order matters: the constants describe post-quantization values."""
    sub = np.array([[[-63.457]], [[-63.457]]], dtype=np.float32)
    z = FUNCS["build_inference_transform"](_cfg("q55s005"))(sub)
    expected = (np.float32(FLOOR_VALUE) - np.array(V3_MEANS, dtype=np.float32)) / np.array(
        V3_STDS, dtype=np.float32
    )
    assert np.allclose(z.ravel(), expected, atol=1e-5)
    # Without the re-quantization the same input lands somewhere else entirely.
    z_raw = FUNCS["build_inference_transform"](_cfg("none"))(sub)
    assert not np.allclose(z.ravel(), z_raw.ravel())


def test_every_sweep_config_declares_an_encoding():
    """The key is mandatory, so a tracked sweep config missing it cannot run."""
    files = glob.glob(str(REPO / "scripts/evaluation/vm/configs/**/*.yaml"), recursive=True)
    files += glob.glob(str(REPO / "configs/**/*.yaml"), recursive=True)
    missing = []
    for f in files:
        doc = yaml.safe_load(open(f)) or {}
        if not (isinstance(doc, dict) and "sweep" in doc):
            continue
        overrides = (doc["sweep"] or {}).get("common_overrides") or {}
        if "inference.data.s1_encoding" not in overrides:
            missing.append(Path(f).name)
    assert not missing, f"sweep configs without s1_encoding: {missing}"
