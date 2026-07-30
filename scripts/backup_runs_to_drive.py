#!/usr/bin/env python
"""Pack per-run-directory tar.zst archives + a manifest, for an incremental Drive backup.

Why this shape (see docs/BACKUP_RUNS_ARCHIVE.md for the full rationale):
  The natural unit of change in this project is the RUN-DIRECTORY
  (`dev_sweep_all_<site>_<seed>_<ts>_<arch>_<stride>/`). Every reconciliation we do
  (seed-miswiring patches, botched-s42 removal, seedfix re-runs) adds/replaces/removes
  WHOLE run-dirs, never individual files. Run-dirs are immutable once written
  (timestamp in the name) and byte-identical when the run is the same.

  So we back up ONE `<run-dir>.tar.zst` per run-dir instead of millions of tiny
  tif/json files. That gives:
    - fast Drive sync (hundreds of medium files, not millions of small ones)
    - granular updates (fix one run -> replace one tarball)
    - cheap change detection (a name+sha lookup in MANIFEST.csv, not a content crawl)

  This complements docs/BACKUP_SOP.md: the SOP's keep-set rules still decide WHAT files
  go in each archive (tif + json + run metadata; memmap + gpkg dropped). This script is
  the PACKAGING + INCREMENTAL-SYNC layer on top of that.

Determinism: archives are built with Python's tarfile with normalized member metadata
  (sorted entries, clamped mtime, stripped uid/gid/uname/gname, fixed mode bits) and
  piped through `zstd`. The SAME run-dir contents therefore always produce a
  byte-identical archive, so the sha256 recorded in the manifest is a stable identity
  for the run — re-running the packer does not create phantom "changed" rows.

Superseded-data policy (decided 2026-07-15): the Drive backup holds only the
  CURRENT-GOOD set. Botched / pre-fix data is already relocated to local NAS backups
  (e.g. swin_base_s42_botched_seed_backup/) and is NOT uploaded here. When a run is
  superseded, its archive is removed from Drive and its manifest row dropped; history
  lives on the local drives, not on Drive.

This script does NOT delete anything and does NOT push to Drive unless you pass
  --rclone-remote AND --do-upload. Default is a local dry build + manifest so you can
  inspect before syncing.

Manifest semantics (fixed 2026-07-19; previously a partial --source-root run would
  REWRITE the manifest to only the scanned roots' units, orphaning every other row):
  MANIFEST.csv is now MERGED, never truncated. Prior rows are grouped as:
    - rediscovered this run           -> refreshed (packed or skipped as usual)
    - outside the scanned roots       -> carried over untouched
    - under a scanned root, filtered
      out this run but still on disk  -> carried over untouched
    - under a scanned root, source
      dir GONE from disk              -> kept with a WARNING (eviction stays a
                                         deliberate act); pass --prune-missing to drop
  The previous manifest is preserved as MANIFEST.csv.bak and the new one is written
  atomically (tmp + rename), so an interrupted run cannot leave a partial manifest.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import os
import re
import subprocess
import sys
import tarfile
from dataclasses import dataclass, asdict
from datetime import datetime, timezone

# ---- keep-set / cruft rules (mirror docs/BACKUP_SOP.md) --------------------------------

# Regenerable OS / GIS sidecars that accumulate on local copies. Excluded so archives are
# reproducible across machines (we saw these create phantom diffs all through the audit).
CRUFT_BASENAMES = {".DS_Store"}
CRUFT_SUFFIXES = (".tif.aux.xml",)

# Large/redundant artifacts dropped per the keep-set SOP (reconstructable from the tif).
# Kept OPT-IN via --apply-keepset so this script can also archive arbitrary trees verbatim.
KEEPSET_DROP_SUFFIXES = (".memmap", "_shoreline.gpkg", "_probability_water.memmap")

# A run-directory is a leaf dir whose name starts with this and which is not itself a
# parent of other run-dirs. We detect them by name prefix + "contains files".
RUN_DIR_PREFIX = "dev_sweep_"
# Also treat ENSEMBLE_* and sweeps/* leaf dirs as archivable units.
EXTRA_UNIT_PREFIXES = ("ENSEMBLE_",)

# Clamp all mtimes to a fixed epoch so archives are byte-reproducible.
FIXED_MTIME = 946684800  # 2000-01-01T00:00:00Z, arbitrary but stable


def log(msg: str) -> None:
    print(f"[{datetime.now(timezone.utc).isoformat(timespec='seconds')}] {msg}", flush=True)


def is_cruft(name: str) -> bool:
    base = os.path.basename(name)
    return base in CRUFT_BASENAMES or name.endswith(CRUFT_SUFFIXES)


def parse_run_fields(run_name: str) -> dict:
    """Best-effort seed/arch/stride extraction from a run-dir name (for manifest columns)."""
    seed = arch = stride = ""
    m = re.search(r"_(s\d+)(?:-[\w-]+)?_\d{8}T", run_name)
    if m:
        seed = m.group(1)
    m = re.search(r"_b0_(s\d+)(?:_|$)", run_name)
    if m:
        stride = m.group(1)
    m = re.search(r"_(upernet_[\w.-]+?|deeplabv3plus_resnet50|unet_resnet50|unetplusplus_resnet50|"
                  r"dpt_vit_b_16|segformer_mit_b4)_native", run_name)
    if m:
        arch = m.group(1)
    return {"seed": seed, "arch": arch, "stride": stride}


def find_run_dirs(root: str, exclude_re: "re.Pattern | None" = None,
                  include_re: "re.Pattern | None" = None) -> list[str]:
    """Return absolute paths of archivable leaf units under root.

    A unit = a directory whose basename starts with RUN_DIR_PREFIX or an EXTRA prefix.
    We do NOT descend into a unit once found (its whole subtree becomes one archive).

    If exclude_re is given, any unit whose ABSOLUTE PATH matches is skipped (used e.g. to
    keep the SDS-benchmark backup free of Indonesia/demak dirs).
    If include_re is given, ONLY units whose ABSOLUTE PATH matches are kept (surgical
    targeting; rows for the non-matching units are carried over in the manifest merge).
    """
    units: list[str] = []
    root = os.path.abspath(root)
    for dirpath, dirnames, _files in os.walk(root):
        base = os.path.basename(dirpath)
        if base.startswith(RUN_DIR_PREFIX) or base.startswith(EXTRA_UNIT_PREFIXES):
            dirnames[:] = []  # do not recurse into a unit
            if exclude_re and exclude_re.search(dirpath):
                continue
            if include_re and not include_re.search(dirpath):
                continue
            units.append(dirpath)
    return sorted(units)


def collect_members(unit_dir: str, apply_keepset: bool) -> list[tuple[str, str]]:
    """Return sorted (abs_path, arcname) tuples for files to include in the archive."""
    members: list[tuple[str, str]] = []
    unit_parent = os.path.dirname(unit_dir)
    for dp, _dn, files in os.walk(unit_dir):
        for fn in sorted(files):
            full = os.path.join(dp, fn)
            if is_cruft(full):
                continue
            if apply_keepset and full.endswith(KEEPSET_DROP_SUFFIXES):
                continue
            arc = os.path.relpath(full, unit_parent)  # archive holds <run-dir>/<...>
            members.append((full, arc))
    members.sort(key=lambda t: t[1])
    return members


def build_archive(unit_dir: str, out_path: str, apply_keepset: bool,
                  zstd_level: int) -> tuple[str, int, int]:
    """Build a deterministic tar.zst for one unit. Returns (sha256, bytes, n_files)."""
    members = collect_members(unit_dir, apply_keepset)

    # Build the tar into memory-through-pipe: python tarfile -> zstd stdin -> file.
    hasher = hashlib.sha256()
    total_bytes = 0
    zstd = subprocess.Popen(
        ["zstd", f"-{zstd_level}", "-q", "-o", out_path, "-f"],
        stdin=subprocess.PIPE,
    )

    class _Tee(io.RawIOBase):
        """Write tar stream to zstd stdin while sha256-ing the *uncompressed* tar bytes."""
        def writable(self):
            return True

        def write(self, b):
            hasher.update(b)
            zstd.stdin.write(b)
            return len(b)

    tee = _Tee()
    with tarfile.open(fileobj=tee, mode="w|", format=tarfile.PAX_FORMAT) as tar:
        for full, arc in members:
            ti = tar.gettarinfo(full, arcname=arc)
            # normalize for reproducibility
            ti.mtime = FIXED_MTIME
            ti.uid = ti.gid = 0
            ti.uname = ti.gname = ""
            ti.mode = 0o644 if ti.isfile() else 0o755
            if ti.isfile():
                total_bytes += ti.size
                with open(full, "rb") as fh:
                    tar.addfile(ti, fh)
            else:
                tar.addfile(ti)
    zstd.stdin.close()
    rc = zstd.wait()
    if rc != 0:
        raise RuntimeError(f"zstd failed rc={rc} for {unit_dir}")
    return hasher.hexdigest(), total_bytes, len(members)


@dataclass
class ManifestRow:
    archive_relpath: str      # e.g. outputs/inference/runs_trucvert/<run>.tar.zst
    run_name: str
    seed: str
    arch: str
    stride: str
    tar_sha256: str           # sha256 of the *uncompressed deterministic tar stream*
    uncompressed_bytes: int
    n_files: int
    archive_bytes: int        # size of the .tar.zst on disk
    packed_utc: str


def load_manifest(path: str) -> dict[str, dict]:
    if not os.path.exists(path):
        return {}
    with open(path, newline="") as f:
        return {r["archive_relpath"]: r for r in csv.DictReader(f)}


def row_from_prior(prior: dict) -> ManifestRow:
    """Rehydrate a ManifestRow from a raw manifest CSV dict (all values str)."""
    return ManifestRow(
        archive_relpath=prior["archive_relpath"],
        run_name=prior.get("run_name", ""),
        seed=prior.get("seed", ""), arch=prior.get("arch", ""),
        stride=prior.get("stride", ""), tar_sha256=prior["tar_sha256"],
        uncompressed_bytes=int(prior["uncompressed_bytes"]),
        n_files=int(prior["n_files"]), archive_bytes=int(prior["archive_bytes"]),
        packed_utc=prior["packed_utc"])


def write_manifest(path: str, rows: list[ManifestRow]) -> None:
    """Write the manifest atomically, keeping the previous version as .bak."""
    cols = list(asdict(rows[0]).keys()) if rows else [f.name for f in ManifestRow.__dataclass_fields__.values()]
    tmp = path + ".tmp"
    with open(tmp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in sorted(rows, key=lambda x: x.archive_relpath):
            w.writerow(asdict(r))
    if os.path.exists(path):
        os.replace(path, path + ".bak")
    os.replace(tmp, path)


def merge_manifest_rows(prev: dict[str, dict], new_rows: list[ManifestRow],
                        scan_roots: list[tuple[str, str]],
                        prune_missing: bool) -> tuple[list[ManifestRow], int, list[str], list[str]]:
    """Merge this run's rows with prior manifest rows instead of truncating.

    scan_roots: list of (source_root_abs, base) for each scanned source root, where base
    is the rel-base used this run to resolve a row's unit dir back to a disk path.

    Ownership is decided by RESOLVED ABSOLUTE PATH, not by string-prefix matching of
    archive_relpath: a prior row is "under" a scanned root iff normpath(join(base,
    rel_dir)) lies inside that root. This makes the classification immune to relpath
    quirks — a rel-base equal to a source root ('.' prefixes), '..' components in keys
    written under a different rel-base, and root names that are string prefixes of each
    other (SDS_SWEEP vs SDS_SWEEP_DUCK) all resolve correctly.

    Returns (all_rows, n_carried, stale_kept_keys, pruned_keys), where n_carried counts
    every prior row carried over untouched (outside-scope + filtered-but-present):
      - rows rediscovered this run: taken from new_rows
      - rows outside every scanned root: carried over untouched
      - rows under a scanned root whose unit dir still exists on disk (merely filtered
        out this run, e.g. by --include-regex/--exclude-regex): carried over untouched
      - rows under a scanned root whose unit dir is GONE from disk: kept with a warning,
        or dropped when prune_missing is True (eviction stays a deliberate act)
    """
    new_keys = {r.archive_relpath for r in new_rows}
    merged = list(new_rows)
    carried = 0
    stale_kept: list[str] = []
    pruned: list[str] = []
    for key, prior in prev.items():
        if key in new_keys:
            continue
        rel_dir = key[:-len(".tar.zst")] if key.endswith(".tar.zst") else key
        unit_abs = None
        for root_abs, base in scan_roots:
            cand = os.path.normpath(os.path.join(base, rel_dir))
            if cand == root_abs or cand.startswith(root_abs + os.sep):
                unit_abs = cand
                break
        if unit_abs is None:
            merged.append(row_from_prior(prior))
            carried += 1
            continue
        if os.path.isdir(unit_abs):
            merged.append(row_from_prior(prior))
            carried += 1
        elif prune_missing:
            pruned.append(key)
        else:
            merged.append(row_from_prior(prior))
            stale_kept.append(key)
    return merged, carried, stale_kept, pruned


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source-root", required=True, action="append",
                    help="Root to scan for run-dirs (repeatable). e.g. "
                         ".../outputs/inference  or  .../experiments/demak_semarang")
    ap.add_argument("--staging-dir", required=True,
                    help="Local dir where .tar.zst archives + MANIFEST.csv are built.")
    ap.add_argument("--rel-base", default=None,
                    help="Path prefix stripped from source dirs to form archive relpaths. "
                         "Default: the source-root's parent, so the top folder name is kept.")
    ap.add_argument("--apply-keepset", action="store_true",
                    help="Drop memmap/gpkg per docs/BACKUP_SOP.md keep-set rules.")
    ap.add_argument("--exclude-regex", default=None,
                    help="Skip any run-dir whose absolute path matches this regex. "
                         "e.g. for the SDS-benchmark backup: "
                         "'(runs_demak|_demak_|_demak-)' to keep Indonesia/demak dirs out "
                         "(inference + BOTH sweep name forms: dev_sweep_all_demak_* AND "
                         "dev_sweep_all_s1_demak_*).")
    ap.add_argument("--include-regex", default=None,
                    help="Surgical targeting: ONLY pack run-dirs whose absolute path "
                         "matches this regex. All other prior manifest rows are carried "
                         "over untouched (the manifest is merged, never truncated). "
                         "Combine with --force-repack to re-pack specific named dirs.")
    ap.add_argument("--prune-missing", action="store_true",
                    help="Drop manifest rows whose run-dir is under a scanned "
                         "--source-root but no longer exists on disk. Without this flag "
                         "such rows are KEPT and warned about (eviction is deliberate).")
    ap.add_argument("--zstd-level", type=int, default=19)
    ap.add_argument("--rclone-remote", default=None,
                    help="rclone remote path, e.g. drive:segwater_v2_backup/runs_archive")
    ap.add_argument("--do-upload", action="store_true",
                    help="Actually run `rclone copy` of new/changed archives + manifest. "
                         "Requires --rclone-remote. Without this, nothing leaves this machine.")
    ap.add_argument("--force-repack", action="store_true",
                    help="Rebuild archives even if an unchanged manifest row exists.")
    args = ap.parse_args()

    os.makedirs(args.staging_dir, exist_ok=True)
    manifest_path = os.path.join(args.staging_dir, "MANIFEST.csv")
    prev = load_manifest(manifest_path)

    exclude_re = re.compile(args.exclude_regex) if args.exclude_regex else None
    include_re = re.compile(args.include_regex) if args.include_regex else None

    # discover all units across all source roots
    units: list[tuple[str, str]] = []  # (unit_dir, rel_base)
    scan_roots: list[tuple[str, str]] = []  # (source_root_abs, rel_base) per root
    for sr in args.source_root:
        sr = os.path.abspath(sr)
        base = args.rel_base or os.path.dirname(sr)
        scan_roots.append((sr, base))
        for u in find_run_dirs(sr, exclude_re, include_re):
            units.append((u, base))
    log(f"discovered {len(units)} run-dir units across {len(args.source_root)} root(s)"
        + (f" (exclude_regex={args.exclude_regex!r})" if exclude_re else "")
        + (f" (include_regex={args.include_regex!r})" if include_re else ""))

    rows: list[ManifestRow] = []
    packed = skipped = 0
    to_upload: list[str] = []

    for unit_dir, base in units:
        run_name = os.path.basename(unit_dir)
        rel_dir = os.path.relpath(unit_dir, base)          # <parent>/<run-dir>
        archive_rel = rel_dir + ".tar.zst"                 # manifest key
        out_path = os.path.join(args.staging_dir, archive_rel)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        # incremental skip: if a prior row exists, the archive file is present, and we're
        # not force-repacking, trust it (immutable run-dirs -> same name == same content).
        prior = prev.get(archive_rel)
        if prior and os.path.exists(out_path) and not args.force_repack:
            rows.append(ManifestRow(
                archive_relpath=archive_rel, run_name=run_name,
                seed=prior.get("seed", ""), arch=prior.get("arch", ""),
                stride=prior.get("stride", ""), tar_sha256=prior["tar_sha256"],
                uncompressed_bytes=int(prior["uncompressed_bytes"]),
                n_files=int(prior["n_files"]), archive_bytes=int(prior["archive_bytes"]),
                packed_utc=prior["packed_utc"]))
            skipped += 1
            continue

        sha, ubytes, nfiles = build_archive(unit_dir, out_path, args.apply_keepset,
                                            args.zstd_level)
        fields = parse_run_fields(run_name)
        row = ManifestRow(
            archive_relpath=archive_rel, run_name=run_name,
            seed=fields["seed"], arch=fields["arch"], stride=fields["stride"],
            tar_sha256=sha, uncompressed_bytes=ubytes, n_files=nfiles,
            archive_bytes=os.path.getsize(out_path),
            packed_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"))
        rows.append(row)
        packed += 1

        # upload only if new archive OR sha changed vs prior manifest
        if not prior or prior.get("tar_sha256") != sha:
            to_upload.append(archive_rel)
        log(f"packed {archive_rel}  ({nfiles} files, {ubytes/1e6:.1f} MB -> "
            f"{row.archive_bytes/1e6:.1f} MB, sha {sha[:12]})")

    all_rows, n_carried, stale_kept, pruned = merge_manifest_rows(
        prev, rows, scan_roots, args.prune_missing)
    write_manifest(manifest_path, all_rows)
    log(f"done packing. packed={packed} skipped(unchanged)={skipped} "
        f"carried_over(not scanned this run)={n_carried} "
        f"total_manifest_rows={len(all_rows)}  to_upload={len(to_upload)}")
    if stale_kept:
        log(f"WARNING: {len(stale_kept)} manifest row(s) point at run-dirs that no "
            f"longer exist on disk under the scanned roots. They were KEPT "
            f"(pass --prune-missing to drop them):")
        for key in stale_kept[:20]:
            log(f"    STALE {key}")
        if len(stale_kept) > 20:
            log(f"    ... and {len(stale_kept)-20} more")
    if pruned:
        log(f"pruned {len(pruned)} manifest row(s) for run-dirs gone from disk "
            f"(--prune-missing); their archive files were NOT deleted:")
        for key in pruned[:20]:
            log(f"    PRUNED {key}")
        if len(pruned) > 20:
            log(f"    ... and {len(pruned)-20} more")
    bak_note = " (previous version at MANIFEST.csv.bak)" if os.path.exists(manifest_path + ".bak") else ""
    log(f"manifest -> {manifest_path}{bak_note}")

    if args.do_upload:
        if not args.rclone_remote:
            log("ERROR: --do-upload requires --rclone-remote"); return 2
        log(f"UPLOAD: syncing {len(to_upload)} changed archives + manifest to "
            f"{args.rclone_remote}")
        # copy each changed archive (preserves relative path), then the manifest.
        for rel in to_upload:
            src = os.path.join(args.staging_dir, rel)
            dst = f"{args.rclone_remote}/{os.path.dirname(rel)}".rstrip("/")
            subprocess.run(["rclone", "copy", "--checksum", src, dst], check=True)
        subprocess.run(["rclone", "copy", "--checksum", manifest_path,
                        args.rclone_remote], check=True)
        log("UPLOAD complete.")
    else:
        log("dry build only (no upload). Pass --rclone-remote --do-upload to sync.")
        if to_upload:
            log(f"would upload {len(to_upload)} archives:")
            for rel in to_upload[:20]:
                log(f"    {rel}")
            if len(to_upload) > 20:
                log(f"    ... and {len(to_upload)-20} more")

    return 0


if __name__ == "__main__":
    sys.exit(main())
