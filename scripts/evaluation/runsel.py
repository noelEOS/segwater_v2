"""Re-export of the run-dir resolver that lives in the VM eval kit.

The canonical implementation is ``scripts/evaluation/vm/runsel.py``. It lives
inside ``vm/`` because that directory is copied to a fresh VM as a
self-contained kit (see ``vm/MANIFEST.md``) and must not reach outward for
imports.

Scripts in the wider tree should import from here instead of reaching into
``vm/``, which would invert that dependency:

    sys.path.insert(0, str(REPO / "scripts"))      # existing convention
    from evaluation.runsel import resolve_run_dir

One implementation, two import paths, no duplicated logic.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "vm"))

from runsel import (  # noqa: E402  (path shim must precede the import)
    TIMESTAMP_RE,
    RunDirError,
    resolve_glob_spec,
    resolve_run_dir,
    resolve_run_dirs,
    run_dir_regex,
)

__all__ = [
    "RunDirError",
    "TIMESTAMP_RE",
    "run_dir_regex",
    "resolve_run_dirs",
    "resolve_run_dir",
    "resolve_glob_spec",
]
