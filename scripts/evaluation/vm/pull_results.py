"""Pull result files off the VM, refusing anything that is not a small text artifact.

THE RULE THIS ENFORCES
----------------------
``docs/RUNBOOK_INDEX.md``: "**No heavy-raster egress.** Score VM-side; only
KB-scale CSVs leave." Also ``scripts/evaluation/vm/MANIFEST.md`` ("HARD RULE — no
heavy-TIFF egress") and ``docs/RUNBOOK_sds_vm_eval.md``. Ingress (Mac -> VM) is
free and unrestricted; this script is only about the return leg.

Note the rule is a **whitelist** — "only KB-scale CSVs leave" — not a blacklist of
TIFFs. That distinction is the whole point: on 2026-07-30 a bare ``rsync -az`` of
an SDS results dir pulled **2.5 GB of ``.gpkg`` shoreline vectors**, which are not
TIFFs and not CSVs. The rule as written already forbade it; the rule as
*remembered* did not. Hence this file.

WHY CODE AND NOT MORE PROSE
---------------------------
The rule was stated in three separate documents and still broken. The eval kit's
MANIFEST has a table titled "Rules now enforced by code (and where)" precisely
because prose rules kept failing — ``runsel`` for run-dir resolution,
``completion`` for scene counts, ``ckptsel`` for checkpoints, ``naming`` for
collisions. Egress was the one HARD RULE in that kit with **no enforcing call
site**, so it is the one that got broken. This closes that gap.

WHAT IT DOES
------------
Enumerates the remote tree first (``find``, no transfer), classifies every file,
and **refuses before moving a byte** if anything is outside the whitelist or the
total exceeds the size cap. The refusal names the offending extensions and their
byte totals, so the fix is obvious. Only then does it rsync, with the whitelist
as explicit ``--include``/``--exclude`` filters, so even a mid-flight change on
the VM cannot smuggle a raster through.

Usage:
    python scripts/evaluation/vm/pull_results.py \\
        --host gpu-rtx-hpo-west.us-west1-a.spring-ember-503606 \\
        --remote '~/workspace/results/ship_decision_cnxt_2026-07/' \\
        --local  experiments/mx630_stage2/ship_decision_cnxt_2026-07/

    # inspect without transferring
    ... --dry-run
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

# Small text artifacts only. `.md` is here because campaign READMEs/FINDINGS are
# legitimately part of a results tree. Everything else -- rasters (.tif), vectors
# (.gpkg/.shp/.geojson), archives, checkpoints -- is refused.
DEFAULT_ALLOWED = (".csv", ".json", ".md", ".txt", ".yaml", ".yml")

# Total-bytes ceiling for one pull. A results tree of CSVs is single-digit MB;
# 100 MB is far above that and far below anything raster-shaped.
DEFAULT_MAX_BYTES = 100 * 1024 * 1024


class EgressRefused(RuntimeError):
    """Raised instead of transferring when the tree violates the rule."""


def classify(entries, allowed=DEFAULT_ALLOWED):
    """Split ``(size, path)`` pairs into (ok, rejected) by extension.

    Pure and filesystem-free so the tests can exercise the policy directly.
    Extension match is case-insensitive; a file with no extension is rejected
    (it cannot be shown to be a small text artifact).
    """
    ok, rejected = [], []
    allowed_l = {a.lower() for a in allowed}
    for size, path in entries:
        if Path(path).suffix.lower() in allowed_l:
            ok.append((size, path))
        else:
            rejected.append((size, path))
    return ok, rejected


def summarize_rejected(rejected):
    """``{ext: (count, bytes)}`` — what the refusal message reports."""
    agg = defaultdict(lambda: [0, 0])
    for size, path in rejected:
        ext = Path(path).suffix.lower() or "<no extension>"
        agg[ext][0] += 1
        agg[ext][1] += size
    return {k: tuple(v) for k, v in agg.items()}


def check(entries, *, allowed=DEFAULT_ALLOWED, max_bytes=DEFAULT_MAX_BYTES,
          override=False, skip_non_whitelisted=False):
    """Apply the egress policy. Returns the allowed entries or raises.

    ``override`` corresponds to ``--i-know-this-is-heavy``: it permits the
    transfer but does NOT silence the report, so an override is always visible
    in the log of whoever ran it.

    ``skip_non_whitelisted`` corresponds to ``--skip-non-whitelisted``: the
    normal case for a results tree that legitimately holds heavy artifacts
    alongside the CSVs (an SDS sweep dir is CSVs + thousands of ``.gpkg``). It
    filters them out instead of refusing, which is SAFE -- rsync is already
    driven by the whitelist, so the heavy files were never going to be pulled
    either way. It does NOT relax the size cap, which still applies to the
    whitelisted bytes. Distinct from ``override``, which actually permits heavy
    egress.
    """
    ok, rejected = classify(entries, allowed)
    total_ok = sum(s for s, _ in ok)
    problems = []

    if rejected and not skip_non_whitelisted:
        by_ext = summarize_rejected(rejected)
        detail = ", ".join(
            "%s x%d (%s)" % (ext, n, human(b))
            for ext, (n, b) in sorted(by_ext.items(), key=lambda kv: -kv[1][1]))
        problems.append(
            "%d file(s) outside the whitelist %s: %s"
            % (len(rejected), sorted(allowed), detail))

    if total_ok > max_bytes:
        problems.append("allowed files total %s, over the %s cap"
                        % (human(total_ok), human(max_bytes)))

    if problems and not override:
        raise EgressRefused(
            "REFUSING TO PULL — no heavy-raster egress (docs/RUNBOOK_INDEX.md):\n"
            + "\n".join("  - " + p for p in problems)
            + "\n  Score VM-side and bring back only KB-scale CSV/JSON.\n"
              "  If this is genuinely intended, re-run with "
              "--i-know-this-is-heavy.")
    return ok, rejected, total_ok


def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024 or unit == "TB":
            return "%.1f %s" % (n, unit) if unit != "B" else "%d B" % n
        n /= 1024.0
    return "%.1f TB" % n


def remote_listing(host: str, remote: str) -> list[tuple[int, str]]:
    """``(bytes, relpath)`` for every regular file under ``remote``.

    Uses ``find -printf`` so one round trip enumerates the tree; nothing is
    transferred. AppleDouble ``._*`` stubs are skipped -- they are metadata, not
    results, and they inflate counts elsewhere in this kit.
    """
    cmd = ("cd %s 2>/dev/null && find . -type f ! -name '._*' -printf '%%s\\t%%p\\n'"
           % _shquote(remote))
    out = subprocess.run(["ssh", "-o", "ConnectTimeout=30", host, cmd],
                         capture_output=True, text=True)
    if out.returncode != 0:
        raise SystemExit("could not list %s:%s\n%s" % (host, remote, out.stderr.strip()))
    entries = []
    for line in out.stdout.splitlines():
        if not line.strip():
            continue
        size, _, path = line.partition("\t")
        try:
            entries.append((int(size), path.lstrip("./")))
        except ValueError:
            continue
    return entries


def _shquote(s: str) -> str:
    """Quote for the remote shell while leaving a leading ``~`` expandable."""
    if s.startswith("~"):
        head, sep, tail = s.partition("/")
        return head + sep + "'" + tail.replace("'", "'\\''") + "'" if sep else head
    return "'" + s.replace("'", "'\\''") + "'"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", required=True)
    ap.add_argument("--remote", required=True, help="remote dir (trailing / implied)")
    ap.add_argument("--local", required=True, type=Path)
    ap.add_argument("--allow", nargs="+", default=list(DEFAULT_ALLOWED),
                    help="permitted extensions (default: %(default)s)")
    ap.add_argument("--max-bytes", type=int, default=DEFAULT_MAX_BYTES)
    ap.add_argument("--dry-run", action="store_true",
                    help="report what would transfer; move nothing")
    ap.add_argument("--skip-non-whitelisted", action="store_true",
                    help="filter out non-whitelisted files instead of refusing. "
                         "Safe: rsync is whitelist-driven, so they were never "
                         "going to transfer. Use for a results tree that holds "
                         "heavy artifacts beside the CSVs (e.g. SDS sweeps). "
                         "The size cap still applies to the whitelisted bytes.")
    ap.add_argument("--i-know-this-is-heavy", action="store_true",
                    dest="override",
                    help="actually permit heavy egress despite a violation "
                         "(still reported). NOT the same as "
                         "--skip-non-whitelisted.")
    a = ap.parse_args()

    entries = remote_listing(a.host, a.remote)
    if not entries:
        raise SystemExit("no files found under %s:%s" % (a.host, a.remote))

    allowed = tuple(x if x.startswith(".") else "." + x for x in a.allow)
    try:
        ok, rejected, total_ok = check(
            entries, allowed=allowed, max_bytes=a.max_bytes,
            override=a.override, skip_non_whitelisted=a.skip_non_whitelisted)
    except EgressRefused as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(2)

    print("remote tree : %d file(s), %s total"
          % (len(entries), human(sum(s for s, _ in entries))))
    print("would pull  : %d file(s), %s" % (len(ok), human(total_ok)))
    if rejected:
        why = "skipped" if a.skip_non_whitelisted else "override in effect"
        print("EXCLUDED    : %d file(s), %s  (%s)"
              % (len(rejected), human(sum(s for s, _ in rejected)), why))
        for ext, (n, b) in sorted(summarize_rejected(rejected).items(),
                                  key=lambda kv: -kv[1][1]):
            print("    %-14s x%-6d %s" % (ext, n, human(b)))
    if a.dry_run:
        print("(--dry-run: nothing transferred)")
        return

    # Whitelist as rsync filters too: belt and braces, so a file appearing on the
    # VM between the listing and the transfer still cannot come across.
    filters = []
    for ext in allowed:
        filters += ["--include", "*%s" % ext]
    cmd = (["rsync", "-az", "--prune-empty-dirs", "--include", "*/"]
           + filters + ["--exclude", "*",
                        "%s:%s" % (a.host, a.remote.rstrip("/") + "/"),
                        str(a.local) + "/"])
    a.local.mkdir(parents=True, exist_ok=True)
    r = subprocess.run(cmd)
    if r.returncode != 0:
        raise SystemExit("rsync failed with %d" % r.returncode)
    print("pulled -> %s" % a.local)


if __name__ == "__main__":
    main()
