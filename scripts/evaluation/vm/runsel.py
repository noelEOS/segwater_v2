"""Unambiguous inference run-directory resolution.

Inference run dirs are named by ``scripts/run_inference_sweep.py``::

    <sweep_name>_<UTC timestamp>Z_<checkpoint-name>_<preset>_s<stride>

    demak_full_mx630k_20260729T020404Z_mx630k_native224_weighted_224_b0_s32
    demak_full_mx630k_best_20260729T040713Z_mx630k_best_native224_weighted_224_b0_s32

Resolving one of these with a bare prefix glob (``RUNS.glob(name + "_*")``) is
**ambiguous whenever one run's name is a prefix of another's**: the pattern
``demak_full_mx630k_*`` matches BOTH dirs above. That has bitten this repo more
than once, and the failure mode varies by call site -- some assert and crash,
some silently take ``hits[0]`` / ``matches[-1]`` / shell ``| head -1`` and
quietly analyse the wrong run.

The fix is to anchor on the timestamp. The naming contract guarantees the
segment immediately after the sweep name is a UTC stamp:

* ``run_inference_sweep.py`` ``timestamp_id()`` -> ``strftime("%Y%m%dT%H%M%SZ")``
* ``sweep_id = f"{sweep_name}__{timestamp_id()}"``
* ``run_name = sanitize_for_path(f"{sweep_id}__{ckpt}__{preset}")`` (``__`` -> ``_``)

so ``^<name>_\\d{8}T\\d{6}Z_`` matches the intended run and cannot match a
longer sibling, which necessarily puts a non-digit segment where the timestamp
must be.

Design notes (these are deliberate; please keep them):

* **Singular and plural are separate functions**, not one function with a
  ``require_unique`` flag. A flag that returns ``Path | list[Path]`` gives every
  caller a union type to unpack, which is precisely how ``hits[0]`` creeps back
  in. Making the call site state its cardinality is the point.
* **Errors raise :class:`RunDirError`, never ``assert``.** Under ``python -O``
  asserts are stripped, which would silently reinstate the bug in the exact
  scripts this module exists to protect.
* **Error messages list the candidate basenames**, not just a count. When this
  fires on a remote machine, the names are the diagnosis.

This module is stdlib-only (``re`` + ``pathlib``) so it can be copied with the
VM eval kit and imported without an editable install. See
``scripts/evaluation/vm/MANIFEST.md``.
"""

from __future__ import annotations

import re
from pathlib import Path

__all__ = [
    "RunDirError",
    "TIMESTAMP_RE",
    "TIMESTAMP_PATTERN",
    "run_dir_regex",
    "resolve_run_dirs",
    "resolve_run_dir",
    "resolve_glob_spec",
]

#: The UTC stamp emitted by ``run_inference_sweep.timestamp_id()``.
TIMESTAMP_RE = r"\d{8}T\d{6}Z"

#: Compiled form, for callers that need to ``search`` a dir name for its stamp
#: (positional ``name.split("_")[i]`` indexing silently breaks when a preset
#: changes the underscore count — search for the stamp instead).
TIMESTAMP_PATTERN = re.compile(TIMESTAMP_RE)


class RunDirError(Exception):
    """A run directory could not be resolved to the expected cardinality."""


def _fmt(paths) -> str:
    """Render candidates for an error message, one per line."""
    if not paths:
        return "  (none)"
    return "\n".join("  " + Path(p).name for p in paths)


def run_dir_regex(name: str, *, timestamp: str | None = None) -> re.Pattern:
    """Regex matching run dirs produced by the sweep named ``name``.

    The name is anchored at the start and must be followed by a UTC timestamp,
    so a longer sibling name (``<name>_best_...``) cannot match.

    ``timestamp`` pins one exact stamp (e.g. ``"20260729T020404Z"``) when a
    name alone is ambiguous because the same sweep was run more than once.
    """
    stamp = re.escape(timestamp) if timestamp else TIMESTAMP_RE
    return re.compile(r"^" + re.escape(name) + r"_" + stamp + r"(?:_|$)")


def resolve_run_dirs(
    runs_root,
    name: str,
    *,
    timestamp: str | None = None,
    suffix: str | None = None,
    expect: int | None = None,
) -> list[Path]:
    """Return every run dir under ``runs_root`` belonging to sweep ``name``.

    One sweep name legitimately maps to several dirs -- typically one per
    stride (``..._b0_s8`` / ``_s32`` / ``_s112``). Pass ``expect`` to assert the
    count you require; leave it ``None`` to accept any number.

    ``suffix`` additionally requires the dir name to end with the given string.
    Front-anchoring does not subsume this: it is how callers separate tokens
    where one is a prefix of another (``unet_resnet50`` vs
    ``unetplusplus_resnet50``), which occurs mid-name rather than at the front.

    Raises :class:`RunDirError` if ``expect`` is given and not met.
    """
    root = Path(runs_root)
    pat = run_dir_regex(name, timestamp=timestamp)
    hits = sorted(
        d for d in root.glob(name + "_*")
        if d.is_dir()
        and pat.match(d.name)
        and (suffix is None or d.name.endswith(suffix))
    )
    if expect is not None and len(hits) != expect:
        detail = "" if suffix is None else " suffix=%r" % suffix
        if timestamp:
            detail += " timestamp=%r" % timestamp
        raise RunDirError(
            "%s: expected %d run dir(s), found %d under %s%s\n%s"
            % (name, expect, len(hits), root, detail, _fmt(hits))
        )
    return hits


def resolve_run_dir(
    runs_root,
    name: str,
    *,
    timestamp: str | None = None,
    suffix: str | None = None,
) -> Path:
    """Return the single run dir for sweep ``name``.

    Raises :class:`RunDirError` unless exactly one matches. When several do, the
    message lists them so the caller can disambiguate with ``timestamp=`` or
    ``suffix=``.
    """
    return resolve_run_dirs(
        runs_root, name, timestamp=timestamp, suffix=suffix, expect=1
    )[0]


def _anchor_pattern(pattern: str) -> str:
    """Translate a shell-ish run-dir glob into an anchored regex.

    The **first** ``*`` is where the timestamp lives, so it becomes
    ``<stamp>(?:_.*)?`` -- the stamp itself, optionally followed by the rest of
    the middle (checkpoint name, preset, ...). Any later ``*`` is an ordinary
    ``.*``.

    Getting this rule right matters. Substituting the first ``*`` with a bare
    ``<stamp>`` looks equivalent and works for patterns whose ``*`` stands for
    exactly the timestamp -- but it silently breaks the ones that let a single
    ``*`` span the timestamp *and* the middle, e.g.::

        hampyeong_mx630k_s42_last_*_s32

    where the naive rule would demand ``..._last_<stamp>_s32`` and match
    nothing. Both shapes are in use, so the trailing ``(?:_.*)?`` is required.
    """
    head, star, rest = pattern.partition("*")
    if not star:
        return r"^" + re.escape(pattern) + r"$"
    return (
        r"^"
        + re.escape(head)
        + TIMESTAMP_RE
        + r"(?:_.*)?"
        + ".*".join(re.escape(p) for p in rest.split("*"))
        + r"$"
    )


def resolve_glob_spec(root, pattern: str, *, timestamp: str | None = None) -> list[Path]:
    """Resolve a spec-file glob (the ``glob:`` values in the Hampyeong specs).

    The pattern is matched against paths relative to ``root`` and is
    **auto-anchored**: the first ``*`` must begin with a UTC timestamp. This is
    strictly a tightening -- it can only reject a directory that violates the
    naming contract -- so existing spec patterns keep resolving to the same
    dirs while prefix siblings stop being candidates.

    ``timestamp`` pins one exact stamp instead of accepting any.
    """
    root = Path(root)
    stamp = re.escape(timestamp) if timestamp else TIMESTAMP_RE
    rx = re.compile(_anchor_pattern(pattern).replace(TIMESTAMP_RE, stamp, 1))
    return sorted(
        p for p in root.glob(pattern)
        if rx.match(p.relative_to(root).as_posix())
    )
