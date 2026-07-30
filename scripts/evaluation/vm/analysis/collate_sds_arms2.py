"""Collate Narrabeen SDS sweep metrics across all five mx630 arms.

Reads sweep_metrics.csv from each run dir, tagging arch / lineage / arm / stride
so the `model` identity is legible from the row alone (two arms of DIFFERENT
architectures share the checkpoint filename step23930_last).

Hardening (plan 2.4). The numeric/pandas body is untouched; the *plumbing*
around it is now defensive, because every failure mode this script had was
silent:

* SPEC keys are matched by substring, so an **overlapping pair** (one key a
  substring of another) makes ``len(hit) != 1`` fire and drop BOTH dirs with a
  message that looks like ordinary noise. :func:`check_spec_disjoint` refuses to
  start in that state, and a genuine multi-match at resolution time is now
  FATAL rather than skipped.
* The seed lived only in the ``"%s_s42_%s_%s"`` format string, so re-using this
  script for another seed silently mislabels every row. The seed is now DATA in
  the SPEC tuple, cross-checked against the seed embedded in the key.
* Sweep dirs are required to carry a UTC stamp (``runsel.TIMESTAMP_PATTERN``),
  which rejects hand-made/copied dirs that no run produced.
* Every SKIP goes to stderr and is counted; an empty collection raises a clear
  message instead of ``pd.concat([])``'s opaque ValueError; and the number of
  distinct arms is asserted against ``--expect-arms``.
* The CSV is written through :func:`naming.atomic_to_csv`, so a kill mid-write
  cannot leave a truncated table at the final path.
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import runsel  # noqa: E402
from naming import atomic_to_csv  # noqa: E402

DEFAULT_ROOTS = ["/home/noel/sds_vm_eval_mx630s2", "/home/noel/sds_vm_eval_mx630k",
                 "/home/noel/sds_vm_eval_mx630_arms2"]
DEFAULT_OUT = "/home/noel/sds_narrabeen_mx630_arms_msl.csv"
# (dir-substring) -> (arch, lineage, arm, seed)
SPEC = [("mx630s2_s42_best", ("swinb", "mx630s2", "best", 42)),
        ("mx630s2_s42_last", ("swinb", "mx630s2", "last", 42)),
        ("mx630s2_s42_swa5", ("swinb", "mx630s2", "swa5", 42)),
        ("mx630k_s42_last",  ("cnxv2t", "mx630k", "last", 42)),
        ("mx630k_s42_best",  ("cnxv2t", "mx630k", "best", 42))]

#: The seed as spelled inside a SPEC key (``..._s42_...``).
KEY_SEED_RE = re.compile(r"_s(\d+)_")


class SpecError(Exception):
    """A SPEC table is internally inconsistent (overlap, or seed disagreement)."""


def check_spec_seeds(spec) -> None:
    """Raise :class:`SpecError` if a key's ``_s<N>_`` disagrees with its tuple.

    Typo guard: the seed is carried twice (in the dir-matching key and in the
    tuple that lands in the ``seed`` column), so the two must agree or one of
    them is wrong and the rows are mislabelled.
    """
    bad = []
    for key, val in spec:
        m = KEY_SEED_RE.search(key)
        if m is None:
            bad.append("%r: no _s<N>_ segment to check against seed=%r" % (key, val[3]))
        elif int(m.group(1)) != int(val[3]):
            bad.append("%r embeds seed %s but its tuple says seed=%r"
                       % (key, m.group(1), val[3]))
    if bad:
        raise SpecError("SPEC seed mismatch:\n" + "\n".join("  " + b for b in bad))


def check_spec_disjoint(spec) -> None:
    """Raise :class:`SpecError` if any SPEC key is a substring of another.

    Matching is ``key in base_name``, so an overlapping pair makes every dir
    that matches the longer key match the shorter one too -- two hits, which the
    resolution branch would drop. Refusing at startup turns a silent
    both-arms-missing into an unmissable error.
    """
    keys = [k for k, _ in spec]
    bad = sorted({(a, b) for a in keys for b in keys if a != b and a in b})
    if bad:
        raise SpecError(
            "SPEC keys overlap -- a dir matching the longer key matches both, "
            "so BOTH arms would be dropped:\n"
            + "\n".join("  %r is a substring of %r" % (a, b) for a, b in bad)
        )


def resolve_spec(base_name: str, spec):
    """Return the single SPEC tuple whose key occurs in ``base_name``, else None.

    ``None`` means "no arm claims this dir" -- a normal, countable skip (roots
    hold sweeps from other campaigns). More than one match means the SPEC is
    ambiguous for this dir, which is FATAL: taking either one would silently
    label the rows with a coin flip.
    """
    hits = [(k, v) for k, v in spec if k in base_name]
    if not hits:
        return None
    if len(hits) > 1:
        raise SystemExit(
            "FATAL: %d SPEC keys match run dir %r: %s\n"
            "Two arms cannot claim the same dir -- fix SPEC (see "
            "check_spec_disjoint) or narrow the keys."
            % (len(hits), base_name, ", ".join(repr(k) for k, _ in hits))
        )
    return hits[0][1]


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--roots", nargs="+", default=list(DEFAULT_ROOTS),
                   help="sweep-output roots to scan for '*_sweep' dirs")
    p.add_argument("--out", default=DEFAULT_OUT, help="output CSV path")
    p.add_argument("--expect-arms", type=int, default=len(SPEC),
                   help="required number of distinct `model` values after collation")
    p.add_argument("--site", default="NARRABEEN", help="value for the `site` column")
    p.add_argument("--reference", default="MSL",
                   help="tidal datum, value for the `reference` column")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    check_spec_seeds(SPEC)
    check_spec_disjoint(SPEC)

    rows = []
    SITE, REFERENCE = args.site, args.reference
    n_skip_nospec = n_skip_nostamp = n_skip_nostride = n_skip_nometrics = 0
    for root in args.roots:
      for d in sorted(glob.glob(os.path.join(root, "*_sweep"))):
        base = os.path.basename(d)
        if not runsel.TIMESTAMP_PATTERN.search(base):
            print("SKIP (no UTC stamp): %s" % base, file=sys.stderr)
            n_skip_nostamp += 1; continue
        spec_hit = resolve_spec(base, SPEC)
        if spec_hit is None:
            print("SKIP (0 spec matches): %s" % base, file=sys.stderr)
            n_skip_nospec += 1; continue
        arch, lineage, arm, SEED = spec_hit
        m = re.search(r"_b0_s(\d+)_sweep$", base)
        if not m:
            print("SKIP (no stride): %s" % base, file=sys.stderr)
            n_skip_nostride += 1; continue
        f = os.path.join(d, "sweep_metrics.csv")
        if not os.path.exists(f):
            print("SKIP (no metrics): %s" % base, file=sys.stderr)
            n_skip_nometrics += 1; continue
        t = pd.read_csv(f)
        # --- pandas body: VERBATIM from the pre-hardening script. The only
        # change is that the three literals it hard-coded (42 / "NARRABEEN" /
        # "MSL") are now the bound names SEED / SITE / REFERENCE. Keep it that
        # way: the column set and order here IS the output schema.
        t["arch"], t["lineage"], t["arm"] = arch, lineage, arm
        t["model"] = "%s_s%s_%s_%s" % (arch, SEED, lineage, arm)
        t["stride"] = int(m.group(1))
        t["site"], t["reference"], t["seed"] = SITE, REFERENCE, SEED
        t["run_dir"] = base
        rows.append(t)

    n_skip = n_skip_nostamp + n_skip_nospec + n_skip_nostride + n_skip_nometrics
    print("scanned %d root(s); collected %d run dir(s); skipped %d "
          "(no-stamp %d, no-spec-match %d, no-stride %d, no-metrics %d)"
          % (len(args.roots), len(rows), n_skip, n_skip_nostamp, n_skip_nospec,
             n_skip_nostride, n_skip_nometrics), file=sys.stderr)
    if not rows:
        raise SystemExit(
            "no sweep_metrics.csv collected from %d root(s): %s\n"
            "Nothing to collate -- check the roots exist and hold "
            "'*_sweep' dirs whose names contain a SPEC key, a UTC stamp and a "
            "'_b0_s<stride>_sweep' tail. Any SKIP lines above name what was "
            "dropped and why." % (len(args.roots), ", ".join(args.roots))
        )

    df = pd.concat(rows, ignore_index=True)
    found = sorted(df["model"].unique())
    if len(found) != args.expect_arms:
        raise SystemExit(
            "expected %d distinct arm(s) (--expect-arms), found %d: %s\n"
            "Any SKIP lines above name the dirs that were dropped and why."
            % (args.expect_arms, len(found), ", ".join(found))
        )

    out = args.out
    atomic_to_csv(df, out, index=False)
    print("wrote %s (%d rows; %d model x stride cells)" % (out, len(df), df.groupby(["model","stride"]).ngroups))
    print(df.groupby(["model","stride"]).size().unstack(fill_value=0).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
