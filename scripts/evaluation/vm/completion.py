"""Is an inference run COMPLETE? Count probability rasters — nothing else works.

Every other "completion signal" in a run dir is a trap, and each has already
cost real time (docs/RUNBOOK_multiagent_vm_campaign.md §2):

* ``run_metadata.json`` is an **evaluation** output. It never appears in an
  inference run dir; polling a run dir for it waits forever (~20 min lost in
  the ship campaign).
* ``run_summary.json`` IS written into the run dir — but it is rewritten after
  **every scene**, so its presence means "at least one scene finished", not
  "the run is complete". It carries no scene count to compare against.
* The sweep's exit code is always 0, and ``continue_on_error: true`` is in
  every sweep template, so a sweep can finish SHORT of its scene count and
  still look successful. ``sweep_summary.json`` has the counts but lives in
  the ``sweeps/`` tree, not the run dir.

The one signal that cannot lie is the output itself: count
``*_probability_water.tif`` (specifically — run dirs also hold auxiliary
rasters, so ``*.tif`` overcounts) against the gate's expected scene count.
AppleDouble ``._*`` stubs are excluded explicitly: they are prefix-mangled
(``._X_probability_water.tif``) so a suffix glob alone WOULD count them.

Design contract shared with ``runsel.py`` (deliberate; please keep):

* Errors raise :class:`CompletionError`, never ``assert`` (``python -O``
  strips asserts in exactly the scripts this protects).
* Stdlib-only, so it can be copied with the VM eval kit and imported without
  an editable install.
* :data:`EXPECTED_SCENES` is a Python dict, not a YAML sidecar: a data file
  would add a parse dependency and a path-resolution question for six rows of
  integers that change roughly once per campaign. Shell callers read the
  numbers via ``--print-expected`` instead of keeping a second copy.

CLI::

    python completion.py --gate narrabeen --print-expected     # -> 87
    python completion.py --gate narrabeen --check DIR [DIR...] # exit 1 if any short
    python completion.py --expected 87 --check DIR [DIR...]    # explicit count
"""

from __future__ import annotations

from pathlib import Path

__all__ = [
    "CompletionError",
    "EXPECTED_SCENES",
    "PROBABILITY_GLOB",
    "expected_scenes",
    "count_probability_rasters",
    "is_run_complete",
    "require_run_complete",
    "completion_report",
]

#: Per-scene layout: each scene gets a subdir holding its probability raster.
PROBABILITY_GLOB = "*/*_probability_water.tif"

#: Expected probability-raster count per evaluation gate. THE single source of
#: truth — every hardcoded 6 / 87 / 213 in the kit should read this instead.
#:
#: demak_gate / hampyeong / narrabeen / demak_full gate real decisions.
#: duck / torreypines / trucvert are STAGED counts measured 2026-07-27
#: (staging-dependent, advisory): re-measure if scenes are ever re-staged —
#: see the canonical-frames table in scripts/evaluation/vm/MANIFEST.md.
EXPECTED_SCENES = {
    "demak_gate": 6,       # 6 concurrent S1/S2 pairs
    "hampyeong": 6,        # 6 S1B descending scenes (5 DEM-flood dates + 1)
    "narrabeen": 87,       # ron_147_ts_9_sn_16, staged in-window set
    "duck": 109,           # ron_4_ts_24_sn_19 — staged 2026-07-27, advisory
    "torreypines": 25,     # ron_71 — staged 2026-07-27, advisory (thin!)
    "trucvert": 78,        # ron_8_ts_30_sn_20 — staged 2026-07-27, advisory
    "demak_full": 213,     # full series; analysis window keeps 206 (<= 2024-12-31)
}


class CompletionError(Exception):
    """A run directory is short of (or unknown against) its expected count."""


def expected_scenes(gate: str) -> int:
    """Expected raster count for ``gate``. Raises on an unknown gate name —
    never returns a default, because a guessed count silently passes short runs.
    """
    try:
        return EXPECTED_SCENES[gate]
    except KeyError:
        raise CompletionError(
            "unknown gate %r; known gates: %s"
            % (gate, ", ".join(sorted(EXPECTED_SCENES)))
        ) from None


def count_probability_rasters(run_dir, *, glob: str = PROBABILITY_GLOB) -> int:
    """Count probability rasters under ``run_dir``, excluding AppleDouble
    ``._*`` stubs (rsync from macOS can plant them; they match a suffix glob
    but are junk metadata files, not scenes)."""
    root = Path(run_dir)
    return sum(1 for p in root.glob(glob) if not p.name.startswith("._"))


def is_run_complete(run_dir, expected_scenes: int, *, glob: str = PROBABILITY_GLOB) -> bool:
    """True when ``run_dir`` holds at least ``expected_scenes`` rasters.

    A count ABOVE expectation is not "complete", it is a different problem
    (wrong expectation, or two sweeps writing one dir) — use
    :func:`require_run_complete` to distinguish, this predicate stays boolean.
    """
    return count_probability_rasters(run_dir, glob=glob) == expected_scenes


def require_run_complete(run_dir, expected_scenes: int, *, glob: str = PROBABILITY_GLOB) -> int:
    """Return the raster count, raising :class:`CompletionError` unless it
    equals ``expected_scenes`` exactly. Over-complete is an error too: it means
    the expectation is stale or two sweeps shared the out-dir."""
    n = count_probability_rasters(run_dir, glob=glob)
    if n != expected_scenes:
        raise CompletionError(
            "%s: expected %d probability raster(s), found %d (glob %r)"
            % (Path(run_dir), expected_scenes, n, glob)
        )
    return n


def completion_report(run_dirs, expected_scenes: int, *, glob: str = PROBABILITY_GLOB):
    """``[(run_dir, count, complete), ...]`` sorted by dir name — deterministic
    for logs and shell loops."""
    out = []
    for d in sorted(Path(p) for p in run_dirs):
        n = count_probability_rasters(d, glob=glob)
        out.append((d, n, n == expected_scenes))
    return out


def _main(argv=None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--gate", choices=sorted(EXPECTED_SCENES),
                   help="named gate; its expected count comes from EXPECTED_SCENES")
    g.add_argument("--expected", type=int,
                   help="explicit expected raster count (instead of --gate)")
    ap.add_argument("--print-expected", action="store_true",
                    help="print the expected count and exit (shell consumption)")
    ap.add_argument("--check", nargs="+", metavar="RUN_DIR", default=None,
                    help="run dirs to check; exit 1 if any is not exactly complete")
    ap.add_argument("--glob", default=PROBABILITY_GLOB)
    a = ap.parse_args(argv)

    n_expected = expected_scenes(a.gate) if a.gate else a.expected
    if a.print_expected:
        print(n_expected)
        return 0
    if not a.check:
        ap.error("nothing to do: pass --print-expected or --check DIR...")

    bad = 0
    for d, n, ok in completion_report(a.check, n_expected, glob=a.glob):
        print("%s  %s  %d/%d" % ("OK  " if ok else "SHORT" if n < n_expected else "OVER",
                                 d, n, n_expected))
        bad += 0 if ok else 1
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(_main())
