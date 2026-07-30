"""Checkpoint role classification and selection — one implementation.

Before this module, the STEP/MIOU regexes, the best.pth candidate-pool
exclusions, and the selection rules were re-implemented (divergently) in
``ensure_best_ckpts.py``, ``relink_best_ckpts.py``, ``eval_stratified_ladder.py``
and ``build_swa_checkpoint.py`` — plus a fifth copy with NO exclusions at all
(``make_best_ckpt_path.py``, deleted) that would happily relink ``best.pth`` to
a ``_last.pth`` or SWA file.

**The two "best" rules are deliberately distinct — do not merge them:**

* :func:`pick_best_filename_rule` — the AS-SHIPPED rule: highest 4-dp mIoU
  parsed from the FILENAME, tie-break highest step, final fallback on name.
  This is what every downstream eval and inference run actually loaded; it is
  reproduced, never improved.
* :func:`pick_best_float_rule` — the trainer's argmax of the full-float
  ``val_miou`` stored INSIDE the checkpoint, ties to the later step. Used as a
  CROSS-CHECK; disagreements (known case: Swin-Base s42) are reported, not
  resolved.

**Resolution hazards this module encodes (each from a real incident):**

* ``best`` resolves via the ``best.pth`` SYMLINK only. Never glob
  ``*_pmwiou*.pth`` — each seed dir holds THREE such files, so a glob picks by
  sort order (campaign runbook §5).
* A checkpoint file copied into the wrong seed dir passes any path check; only
  the ``_<seed>_`` token in its FILENAME catches it
  (:func:`require_seed_token`).
* Two arms silently sharing weights is the one failure whose output is
  indistinguishable from success — hash and compare
  (:func:`assert_distinct_weights`) BEFORE spending GPU time.

Design contract shared with ``runsel.py``: errors raise
:class:`CkptSelError`, never ``assert``; error messages name the candidates;
stdlib-only (``re``, ``hashlib``, ``pathlib``) — functions that need a LOADED
checkpoint take the already-loaded dict, they never import torch.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

__all__ = [
    "CkptSelError",
    "STEP_RE",
    "MIOU_RE",
    "SWA_RE",
    "step_of",
    "miou_of",
    "step_of_checkpoint",
    "role_of",
    "is_best_candidate",
    "best_pool",
    "pick_best_filename_rule",
    "pick_best_float_rule",
    "resolve_best",
    "resolve_last",
    "require_seed_token",
    "sha256",
    "assert_distinct_weights",
]

STEP_RE = re.compile(r"step(\d+)")
MIOU_RE = re.compile(r"miou([0-9]*\.?[0-9]+)")
#: SWA checkpoints (``build_swa_checkpoint.py``: ``_swa{K}_step{min}-{max}.pth``)
#: are a derived, COMPETING checkpoint-selection arm, not a candidate for
#: best.pth — keeping them out of the pool is what makes best-vs-SWA an
#: independent contrast. They also carry ``val_miou=None``, which the float
#: cross-check must survive.
SWA_RE = re.compile(r"_swa\d+_step")


class CkptSelError(Exception):
    """A checkpoint could not be resolved/validated as requested."""


def step_of(path) -> int:
    """Training step parsed from the filename; ``-1`` when absent."""
    m = STEP_RE.search(Path(path).name)
    return int(m.group(1)) if m else -1


def miou_of(path) -> float:
    """Filename mIoU (4-dp as written by the trainer); ``-1.0`` when absent."""
    m = MIOU_RE.search(Path(path).name)
    return float(m.group(1)) if m else -1.0


def step_of_checkpoint(path, ckpt: dict) -> int:
    """Step from the LOADED checkpoint payload, falling back to the filename.

    This is ``relink_best_ckpts``' rule and it is deliberately different from
    :func:`step_of`: derived checkpoints may carry a payload step that the
    filename does not (or vice versa), and the trainer rule trusts the payload.
    """
    step = ckpt.get("step")
    if step is not None:
        return int(step)
    return step_of(path)


def role_of(name: str) -> str:
    """Classify a checkpoint filename: ``last`` / ``snap`` / ``best`` / ``other``.

    Order matters: ``_last`` / ``_snap_`` before the generic ``_miou`` (a
    snapshot name also contains ``miou``).
    """
    if name.endswith("_last.pth"):
        return "last"
    if "_snap_" in name:
        return "snap"
    if "_miou" in name:
        return "best"  # a retained top-k checkpoint
    return "other"


def is_best_candidate(path) -> bool:
    """Pool membership for best.pth selection.

    Excluded, each for a reason:

    * ``best.pth`` itself — the artifact being created, not an ingredient;
    * ``_snap_`` snapshots — model-only mid-training states for SWA building;
    * SWA outputs (:data:`SWA_RE`) — a competing selection arm (see above);
    * ``*_last.pth`` — the competing deploy-final rule's pick.
    """
    name = Path(path).name
    return (
        name != "best.pth"
        and "_snap_" not in name
        and not SWA_RE.search(name)
        and not name.endswith("_last.pth")
    )


def best_pool(seed_dir) -> list[Path]:
    """Sorted best.pth candidate pool of a seed dir; raises if empty."""
    seed_dir = Path(seed_dir)
    pool = [p for p in sorted(seed_dir.glob("*.pth")) if is_best_candidate(p)]
    if not pool:
        raise CkptSelError("no step checkpoints (best-candidates) in %s" % seed_dir)
    return pool


def pick_best_filename_rule(pths) -> Path:
    """AS-SHIPPED rule: highest 4-dp filename mIoU; tie-break highest step;
    final fallback on name. Byte-for-byte the historical selection — reproduce,
    never improve."""
    def key(p: Path):
        return (miou_of(p), step_of(p), p.name)
    return max(pths, key=key)


def pick_best_float_rule(scored) -> Path:
    """Trainer rule over pre-scored candidates: argmax of full-float
    ``val_miou``, ties to the later step.

    ``scored`` is an iterable of ``(val_miou: float, step: int, path)`` — the
    caller loads the checkpoints and decides how to treat a missing
    ``val_miou`` (``ensure_best_ckpts`` maps it to ``-inf``;
    ``relink_best_ckpts`` refuses). Tie behavior beyond ``(miou, step)`` is
    "last in input order wins", preserved from both originals.
    """
    scored = list(scored)
    if not scored:
        raise CkptSelError("pick_best_float_rule: empty candidate list")
    return sorted(scored, key=lambda t: (t[0], t[1]))[-1][2]


def resolve_best(seed_dir) -> Path:
    """The ``best.pth`` symlink's target. Raises if the symlink is missing or
    points at a ``_last.pth`` (always a wiring error: 'best' and 'last' are
    competing arms and must stay distinct)."""
    seed_dir = Path(seed_dir)
    link = seed_dir / "best.pth"
    if not link.exists():
        raise CkptSelError(
            "%s: no best.pth symlink — create it with "
            "scripts/evaluation/ensure_best_ckpts.py; do NOT glob *_pmwiou*.pth "
            "(several per seed dir; a glob picks by sort order)" % seed_dir
        )
    p = link.resolve()
    if p.name.endswith("_last.pth"):
        raise CkptSelError(
            "%s: best.pth points at a _last checkpoint (%s)" % (seed_dir, p.name)
        )
    return p


def resolve_last(seed_dir) -> Path:
    """The single ``*_last.pth`` of a seed dir; raises on 0 or >1, listing
    the candidates."""
    seed_dir = Path(seed_dir)
    hits = sorted(seed_dir.glob("*_last.pth"))
    if len(hits) != 1:
        raise CkptSelError(
            "%s: expected 1 *_last.pth, found %d: %s"
            % (seed_dir, len(hits), [h.name for h in hits])
        )
    return hits[0]


def require_seed_token(path, seed: str) -> None:
    """The resolved FILENAME must carry ``_<seed>_``. A checkpoint copied into
    the wrong seed dir passes every path-based check but fails this one."""
    name = Path(path).name
    if "_%s_" % seed not in name:
        raise CkptSelError(
            "%s: filename does not carry _%s_ — mis-copied checkpoint?" % (name, seed)
        )


def sha256(path) -> str:
    """Streaming sha256 of a checkpoint file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def assert_distinct_weights(arms: dict) -> dict:
    """``{arm_label: checkpoint_path}`` -> ``{arm_label: sha256}``, raising if
    any two arms share a digest. Two arms silently sharing weights produce
    output indistinguishable from success — this is the pre-GPU guard."""
    digests: dict[str, str] = {}
    seen: dict[str, str] = {}  # digest -> first arm label
    for label, path in arms.items():
        d = sha256(path)
        if d in seen:
            raise CkptSelError(
                "DUPLICATE WEIGHTS: %s and %s share sha256 %s\n  %s\n  %s"
                % (label, seen[d], d[:16], Path(path).name,
                   Path(arms[seen[d]]).name)
            )
        seen[d] = label
        digests[label] = d
    return digests
