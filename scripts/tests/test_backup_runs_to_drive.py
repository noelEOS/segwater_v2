#!/usr/bin/env python
"""Standalone tests for scripts/backup_runs_to_drive.py (no pytest dependency).

Run:  python scripts/tests/test_backup_runs_to_drive.py

Everything happens in a throwaway temp dir; no project data is touched. Requires the
`zstd` binary (same requirement as the script itself).

Covers, in particular, the 2026-07-19 manifest-truncation bug: a run scanning a SUBSET
of source roots must MERGE into the existing manifest, not rewrite it to that subset.
"""
from __future__ import annotations

import csv
import hashlib
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPT = os.path.join(HERE, "..", "backup_runs_to_drive.py")

PASS = 0
FAIL = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  PASS  {name}")
    else:
        FAIL += 1
        print(f"  FAIL  {name}  {detail}")


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_manifest(staging: str) -> dict[str, dict]:
    with open(os.path.join(staging, "MANIFEST.csv"), newline="") as f:
        return {r["archive_relpath"]: r for r in csv.DictReader(f)}


def run_backup(source_roots: list[str], staging: str, rel_base: str,
               extra: list[str] | None = None) -> subprocess.CompletedProcess:
    cmd = [sys.executable, SCRIPT]
    for sr in source_roots:
        cmd += ["--source-root", sr]
    cmd += ["--staging-dir", staging, "--rel-base", rel_base, "--zstd-level", "3"]
    cmd += extra or []
    return subprocess.run(cmd, capture_output=True, text=True)


def make_run_dir(base: str, name: str, files: dict[str, bytes]) -> str:
    d = os.path.join(base, name)
    os.makedirs(d, exist_ok=True)
    for rel, content in files.items():
        p = os.path.join(d, rel)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "wb") as f:
            f.write(content)
    return d


def main() -> int:
    tmp = tempfile.mkdtemp(prefix="backup_script_test_")
    try:
        src = os.path.join(tmp, "srcdata")
        root_a = os.path.join(src, "rootA")
        root_b = os.path.join(src, "rootB")
        staging = os.path.join(tmp, "staging")
        for r in (root_a, root_b):
            os.makedirs(r)

        # synthetic run-dirs: 2 under rootA, 2 under rootB (one demak-named), 1 ENSEMBLE
        make_run_dir(root_a, "dev_sweep_alpha_s19_20260101T000000Z_m1_b0_s8",
                     {"metrics.csv": b"a,1\n", "sub/out.tif": b"TIFA" * 100})
        make_run_dir(root_a, "dev_sweep_beta_s42_20260101T000000Z_m1_b0_s8",
                     {"metrics.csv": b"b,2\n", "junk.memmap": b"M" * 50,
                      ".DS_Store": b"x", "img.tif.aux.xml": b"<x/>"})
        make_run_dir(root_b, "dev_sweep_gamma_s58_20260101T000000Z_m2_b0_s32",
                     {"metrics.csv": b"c,3\n"})
        make_run_dir(root_b, "dev_sweep_all_demak_s19_20260101T000000Z_m2_b0_s32",
                     {"metrics.csv": b"d,4\n"})
        make_run_dir(root_b, "ENSEMBLE_pair_cnx_swin",
                     {"metrics.csv": b"e,5\n"})

        # ---------------- 1. initial full run over both roots -------------------------
        print("[1] initial full run (both roots, keepset, demak excluded)")
        p = run_backup([root_a, root_b], staging, src,
                       ["--apply-keepset", "--exclude-regex", "_demak_"])
        check("exit 0", p.returncode == 0, p.stderr[-500:])
        m1 = read_manifest(staging)
        check("4 rows (demak excluded)", len(m1) == 4, str(sorted(m1)))
        check("rootA alpha present", any("alpha" in k for k in m1))
        check("rootB gamma present", any("gamma" in k for k in m1))
        check("ENSEMBLE unit archived", any("ENSEMBLE_pair" in k for k in m1))
        check("demak excluded", not any("demak" in k for k in m1))
        beta_key = next(k for k in m1 if "beta" in k)
        # keepset + cruft: beta should contain ONLY metrics.csv (memmap/DS_Store/aux dropped)
        check("keepset+cruft honored (beta n_files==1)", m1[beta_key]["n_files"] == "1",
              m1[beta_key]["n_files"])

        # ---------------- 2. THE BUG: partial-root run must merge ---------------------
        print("[2] partial run (rootA only) must preserve rootB rows")
        shas_before = {k: v["tar_sha256"] for k, v in m1.items()}
        p = run_backup([root_a], staging, src, ["--apply-keepset"])
        check("exit 0", p.returncode == 0, p.stderr[-500:])
        m2 = read_manifest(staging)
        check("row set unchanged after partial run", set(m2) == set(m1),
              f"lost={set(m1)-set(m2)} gained={set(m2)-set(m1)}")
        check("rootB shas untouched",
              all(m2[k]["tar_sha256"] == shas_before[k] for k in m2 if "gamma" in k or "ENSEMBLE" in k))
        check("carried_over logged", "carried_over" in p.stdout, p.stdout[-300:])
        check(".bak created", os.path.exists(os.path.join(staging, "MANIFEST.csv.bak")))

        # ---------------- 3. determinism / idempotent force-repack --------------------
        print("[3] force-repack over rootA: shas must not move for unchanged dirs")
        p = run_backup([root_a], staging, src, ["--apply-keepset", "--force-repack"])
        check("exit 0", p.returncode == 0, p.stderr[-500:])
        m3 = read_manifest(staging)
        check("all shas identical after force-repack",
              all(m3[k]["tar_sha256"] == shas_before[k] for k in m1 if k in m3),
              "determinism broken")
        check("to_upload=0 for unchanged force-repack", "to_upload=0" in p.stdout,
              p.stdout[-300:])

        # ---------------- 4. content change is detected -------------------------------
        print("[4] content change -> new sha + to_upload")
        alpha_dir = os.path.join(root_a, "dev_sweep_alpha_s19_20260101T000000Z_m1_b0_s8")
        with open(os.path.join(alpha_dir, "extra_metrics.json"), "wb") as f:
            f.write(b'{"new": true}')
        p = run_backup([root_a], staging, src, ["--apply-keepset", "--force-repack"])
        m4 = read_manifest(staging)
        alpha_key = next(k for k in m4 if "alpha" in k)
        check("alpha sha changed", m4[alpha_key]["tar_sha256"] != shas_before[alpha_key])
        check("alpha n_files 2->3", m4[alpha_key]["n_files"] == "3",
              m4[alpha_key]["n_files"])
        check("to_upload=1", "to_upload=1" in p.stdout, p.stdout[-300:])
        check("beta sha unchanged", m4[next(k for k in m4 if 'beta' in k)]["tar_sha256"]
              == shas_before[beta_key])

        # ---------------- 5. surgical --include-regex ---------------------------------
        print("[5] --include-regex targets one dir; everything else carried over")
        with open(os.path.join(alpha_dir, "extra2.json"), "wb") as f:
            f.write(b"{}")
        p = run_backup([root_a, root_b], staging, src,
                       ["--apply-keepset", "--exclude-regex", "_demak_",
                        "--include-regex", "dev_sweep_alpha", "--force-repack"])
        check("exit 0", p.returncode == 0, p.stderr[-500:])
        check("only 1 unit discovered", "discovered 1 run-dir units" in p.stdout,
              p.stdout[:400])
        m5 = read_manifest(staging)
        check("row set fully preserved", set(m5) == set(m4),
              f"lost={set(m4)-set(m5)}")
        check("alpha updated (n_files 4)", m5[alpha_key]["n_files"] == "4",
              m5[alpha_key]["n_files"])
        check("non-target shas untouched",
              all(m5[k]["tar_sha256"] == m4[k]["tar_sha256"] for k in m5 if k != alpha_key))

        # ---------------- 6. missing source dir: warn-keep vs --prune-missing ---------
        print("[6] deleted run-dir: kept+warned by default, dropped with --prune-missing")
        gamma_dir = os.path.join(root_b, "dev_sweep_gamma_s58_20260101T000000Z_m2_b0_s32")
        shutil.rmtree(gamma_dir)
        p = run_backup([root_a, root_b], staging, src,
                       ["--apply-keepset", "--exclude-regex", "_demak_"])
        m6 = read_manifest(staging)
        gamma_key = next(k for k in m5 if "gamma" in k)
        check("gamma row KEPT without flag", gamma_key in m6)
        check("stale warning printed", "STALE" in p.stdout and "WARNING" in p.stdout,
              p.stdout[-500:])
        p = run_backup([root_a, root_b], staging, src,
                       ["--apply-keepset", "--exclude-regex", "_demak_",
                        "--prune-missing"])
        m6b = read_manifest(staging)
        check("gamma row dropped with --prune-missing", gamma_key not in m6b)
        check("only gamma dropped", set(m6) - set(m6b) == {gamma_key},
              str(set(m6) - set(m6b)))
        gamma_archive = os.path.join(staging, gamma_key)
        check("gamma ARCHIVE file not deleted", os.path.exists(gamma_archive))
        # demak row: excluded-by-regex but present on disk -> must survive pruning
        # (it was never in the manifest here; assert no accidental additions instead)
        check("no rows added by pruning run", set(m6b) <= set(m6))

        # ---------------- 6b. F1 regression: rel-base == source-root + prune ----------
        # A run with --rel-base equal to a scanned source root must NOT claim (and
        # under --prune-missing must NOT prune) rows belonging to other, unscanned
        # roots — even though such rows carry '..' components when written under this
        # rel-base, and even though the old '.'-prefix catch-all matched everything.
        print("[6b] F1 regression: rel-base==source-root must not prune other roots")
        f1tmp = os.path.join(tmp, "f1")
        f1_x = os.path.join(f1tmp, "rootX")
        f1_y = os.path.join(f1tmp, "rootY")
        f1_stage = os.path.join(f1tmp, "staging")
        os.makedirs(f1_x); os.makedirs(f1_y)
        make_run_dir(f1_x, "dev_sweep_x1_s19_20260101T000000Z_m1_b0_s8",
                     {"metrics.csv": b"x,1\n"})
        make_run_dir(f1_y, "dev_sweep_y1_s19_20260101T000000Z_m1_b0_s8",
                     {"metrics.csv": b"y,1\n"})
        # full pack of both roots with rel-base = rootX (rootY keys get '..' components)
        p = run_backup([f1_x, f1_y], f1_stage, f1_x)
        check("f1 setup exit 0", p.returncode == 0, p.stderr[-500:])
        mf1 = read_manifest(f1_stage)
        check("f1 setup has both rows", len(mf1) == 2, str(sorted(mf1)))
        y_key = next(k for k in mf1 if "y1" in k)
        # delete rootY's dir, then re-scan ONLY rootX with rel-base rootX + prune
        shutil.rmtree(os.path.join(f1_y, "dev_sweep_y1_s19_20260101T000000Z_m1_b0_s8"))
        p = run_backup([f1_x], f1_stage, f1_x, ["--prune-missing"])
        check("f1 partial-prune exit 0", p.returncode == 0, p.stderr[-500:])
        mf1b = read_manifest(f1_stage)
        check("F1: unscanned root's row survives prune", y_key in mf1b,
              f"rows={sorted(mf1b)}")
        # and the same deletion IS pruned when rootY is actually scanned
        p = run_backup([f1_x, f1_y], f1_stage, f1_x, ["--prune-missing"])
        mf1c = read_manifest(f1_stage)
        check("F1: pruned once its root IS scanned", y_key not in mf1c)

        # ---------------- 7. atomicity: .tmp never left behind -------------------------
        print("[7] no stray tmp file")
        check("no MANIFEST.csv.tmp", not os.path.exists(os.path.join(staging, "MANIFEST.csv.tmp")))

        # ---------------- 8. archives byte-identical across rebuilds ------------------
        print("[8] archive files byte-identical across force-repacks (beta)")
        beta_archive = os.path.join(staging, beta_key)
        sha_a = sha256_file(beta_archive)
        p = run_backup([root_a], staging, src, ["--apply-keepset", "--force-repack"])
        sha_b = sha256_file(beta_archive)
        check("beta .tar.zst byte-identical", sha_a == sha_b)

        print(f"\n{PASS} passed, {FAIL} failed")
        return 1 if FAIL else 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_backup_selftest() -> None:
    """Pytest wrapper so the harness above is COLLECTED coverage, not folklore.

    This file predates `testpaths = tests scripts/tests` (pytest.ini) and is a
    `main()`-style harness: pytest collected ZERO tests from it, so 40-odd
    assertions — including the 2026-07-19 manifest-truncation regression — only
    ran when someone remembered to invoke the file by hand. The wrapper adds no
    new coverage; it makes the existing coverage automatic.

    The harness prints its own PASS/FAIL lines and returns 1 on any failure;
    run pytest with `-s` (or read the captured stdout on failure) to see which
    check failed. Globals are reset so a second call in one session is clean.
    """
    global PASS, FAIL
    PASS = FAIL = 0
    if shutil.which("zstd") is None:
        import pytest
        pytest.skip("`zstd` binary not on PATH (same requirement as the script)")
    rc = main()
    assert rc == 0, f"{FAIL} of {PASS + FAIL} harness checks failed (see stdout)"


if __name__ == "__main__":
    sys.exit(main())
