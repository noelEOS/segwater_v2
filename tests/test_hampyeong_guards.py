"""The Hampyeong scorer's provenance guards must survive ``python -O``.

``score_pairbased_hampyeong.py`` used ``assert`` for all eleven of its
provenance/consistency guards. ``assert`` is stripped by ``-O``, so under that
flag the scorer would have skipped every guard and silently mis-scored -- the
exact opposite of MANIFEST.md's "fails loudly, never silently mis-scores".
These tests pin the fix: guards raise :class:`ProvenanceError`, both in normal
execution and under ``-O``.

Stdlib + pytest only (the module imports numpy/pandas/yaml at module scope but
nothing heavier; the repo-relative torch imports happen inside ``main()``).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
KIT = REPO / "scripts" / "evaluation" / "vm"
SCORER = KIT / "score_pairbased_hampyeong.py"
sys.path.insert(0, str(KIT))

from score_pairbased_hampyeong import ProvenanceError, require  # noqa: E402


def test_require_passes_silently():
    assert require(True, "must not raise") is None


def test_require_raises_provenance_error():
    with pytest.raises(ProvenanceError) as e:
        require(False, "x")
    assert str(e.value) == "x"


def test_provenance_error_is_not_assertion_error():
    """If it subclassed AssertionError, existing `except AssertionError`
    handlers elsewhere could swallow it."""
    assert not issubclass(ProvenanceError, AssertionError)


def test_no_asserts_left_in_the_scorer():
    """The guards must not regress to `assert`. (Docstring/comment mentions of
    the word are fine; a statement-initial `assert ` is not.)"""
    offenders = [
        (i, line) for i, line in enumerate(SCORER.read_text().splitlines(), 1)
        if line.lstrip().startswith("assert ")
    ]
    assert offenders == [], "assert-based guards reintroduced: %r" % offenders


def test_clobber_guard_message_is_preserved():
    """The runbook quotes this string; scoring workflows grep for it."""
    assert "refusing to clobber existing" in SCORER.read_text()


def test_guard_survives_dash_O():
    """The point of the change: `-O` strips `assert`, not `raise`."""
    code = (
        "import sys, importlib.util\n"
        "spec = importlib.util.spec_from_file_location('sph', %r)\n"
        "m = importlib.util.module_from_spec(spec)\n"
        "sys.path.insert(0, %r)\n"
        "spec.loader.exec_module(m)\n"
        "assert False, 'asserts are stripped under -O, so this must not fire'\n"
        "try:\n"
        "    m.require(False, 'boom')\n"
        "except m.ProvenanceError as e:\n"
        "    print('RAISED:' + str(e))\n"
        "    sys.exit(0)\n"
        "sys.exit('GUARD SILENTLY SKIPPED UNDER -O')\n"
        % (str(SCORER), str(KIT))
    )
    proc = subprocess.run([sys.executable, "-O", "-c", code],
                          capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    assert "RAISED:boom" in proc.stdout


def test_uncaught_guard_exits_nonzero_under_dash_O():
    """An unhandled guard must abort the process (not merely return)."""
    code = (
        "import sys, importlib.util\n"
        "spec = importlib.util.spec_from_file_location('sph', %r)\n"
        "m = importlib.util.module_from_spec(spec)\n"
        "sys.path.insert(0, %r)\n"
        "spec.loader.exec_module(m)\n"
        "m.require(False, 'refusing to clobber existing /tmp/x.csv')\n"
        % (str(SCORER), str(KIT))
    )
    proc = subprocess.run([sys.executable, "-O", "-c", code],
                          capture_output=True, text=True)
    assert proc.returncode != 0
    assert "ProvenanceError" in proc.stderr
    assert "refusing to clobber existing" in proc.stderr
