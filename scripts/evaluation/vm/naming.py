"""Sweep-name collision checking and atomic file writes.

Two small cross-cutting utilities, kept out of ``runsel.py`` so that module
stays pure run-dir *resolution* (read-only); this one may write to disk.

**Prefix collisions.** Run dirs are ``<sweep_name>_<UTCstamp>Z_...``, and a
bare prefix glob (``<name>_*``) matches every sweep whose name *extends*
``<name>`` (``demak_full_mx630k_*`` also matches ``demak_full_mx630k_best_...``).
``runsel`` defuses that at *resolution* time; this check defuses it at
*generation* time — before any GPU spend — by refusing to emit a set of sweep
names in which one is a prefix of another. The predicate is exactly
``longer.startswith(shorter + "_")``: names that merely share a prefix token
(``unet_resnet50`` / ``unetplusplus_resnet50``) do NOT collide, because the
extension there is not underscore-separated.

**Atomic writes.** A process killed mid-write leaves a truncated CSV that a
downstream reader parses as a real (short) table — the crashed-scorer variant
of the ship campaign's two-writer incident. Writing to ``<path>.tmp`` in the
same directory and ``os.replace``-ing on success means a partial artifact can
never sit at the final path. Pattern lifted from
``scripts/backup_runs_to_drive.py::write_manifest``.

**Why there is deliberately NO lockfile here.** The two-writer collision of
2026-07-29 (campaign runbook §3) was a *deliberate* second writer — a human
pre-empting an agent — which a lockfile does not prevent; and a lock adds its
own failure mode (stale lock after a kill) that needs an override path and
documentation. The remedies that actually close the hole are:

1. atomic writes (this module) — a partial artifact never looks complete;
2. skip logic keyed on the OUTPUT ARTIFACT (``sweep_metrics.csv``), gated on
   INPUT COMPLETENESS (``completion.py``) — the ``run_ship_sds.sh`` pattern;
3. one writer per output directory as a documented operating rule.

Please do not re-litigate the lockfile without new evidence.

Design contract shared with ``runsel.py``: errors raise
:class:`NameCollisionError`, never ``assert`` (``python -O`` strips asserts);
stdlib-only (pandas is imported lazily and only by :func:`atomic_to_csv`).
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path

__all__ = [
    "NameCollisionError",
    "find_prefix_collisions",
    "require_no_prefix_collisions",
    "atomic_write",
    "atomic_write_text",
    "atomic_to_csv",
]


class NameCollisionError(Exception):
    """One generated name is an underscore-prefix of another."""


def find_prefix_collisions(names) -> list[tuple[str, str]]:
    """Every ``(shorter, longer)`` pair where ``longer.startswith(shorter + "_")``.

    Reproduces the inline check ``gen_ship_configs.py`` shipped with (verified
    against a decoy set there); deterministically ordered for stable output.
    """
    names = list(names)
    bad = [
        (x, y)
        for x in names
        for y in names
        if x != y and y.startswith(x + "_")
    ]
    return sorted(set(bad))


def require_no_prefix_collisions(names, *, what: str = "sweep name") -> None:
    """Raise :class:`NameCollisionError` listing every colliding pair.

    Call this at config-generation time, on the full set of names a campaign
    will create PLUS any pre-existing sibling names the runs root already
    holds — collisions with legacy dirs are just as ambiguous.
    """
    bad = find_prefix_collisions(names)
    if bad:
        raise NameCollisionError(
            "%s prefix collision(s) — a bare '<name>_*' glob would match both:\n%s"
            % (what, "\n".join("  %r is a prefix of %r" % (x, y) for x, y in bad))
        )


@contextmanager
def atomic_write(path, *, mode: str = "w", newline=None, encoding="utf-8",
                 keep_bak: bool = False):
    """Yield a handle on ``<path>.tmp``; ``os.replace`` onto ``path`` on clean
    exit, unlink the tmp on exception. The final path never holds a partial
    file. ``keep_bak=True`` preserves the previous version as ``<path>.bak``.

    ``newline=""`` is what ``csv`` writers want; binary modes pass
    ``encoding=None`` implicitly.
    """
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    kwargs = {} if "b" in mode else {"newline": newline, "encoding": encoding}
    f = open(tmp, mode, **kwargs)
    try:
        yield f
    except BaseException:
        f.close()
        try:
            tmp.unlink()
        except OSError:
            pass
        raise
    else:
        f.close()
        if keep_bak and path.exists():
            os.replace(path, path.with_name(path.name + ".bak"))
        os.replace(tmp, path)


def atomic_write_text(path, text: str, **kw) -> None:
    """Atomic counterpart of ``Path.write_text``."""
    with atomic_write(path, **kw) as f:
        f.write(text)


def atomic_to_csv(df, path, **to_csv_kw) -> None:
    """Atomic counterpart of ``DataFrame.to_csv(path)`` (pandas imported by the
    caller; this just needs the object to have ``.to_csv``)."""
    with atomic_write(path, newline="") as f:
        df.to_csv(f, **to_csv_kw)
