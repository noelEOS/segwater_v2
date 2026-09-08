#!/usr/bin/env bash
# Pull the v3 memmaps from Drive, ONE FILE AT A TIME, using multi-threaded
# streams to saturate the link.
#
# Why not --transfers N: there are only two large files, so file-level
# parallelism gives at most 2 streams. --multi-thread-streams splits a SINGLE
# file across N connections, which is the flag that matters here.
#
# --multi-thread-cutoff 128M ensures the big files take the multi-thread path.
# Drive throttles per-connection, so 16 streams is where the bandwidth comes
# from, not from a bigger chunk size.
set -u
DST=~/memmaps_v3
mkdir -p "$DST"
SRC=gdrive:Segwater_v2_RAW_DATASET/memmaps_v3

common=(--multi-thread-streams 16 --multi-thread-cutoff 128M
        --transfers 1 --checkers 8 --drive-chunk-size 256M
        --stats 60s --stats-one-line --log-level INFO)

for f in val.memmap train.memmap; do
  echo "=== $f start $(date -u +%H:%M:%S)"
  rclone copy "$SRC/$f" "$DST/" "${common[@]}" --log-file ~/pull_${f}.log
  echo "=== $f exit $? $(date -u +%H:%M:%S)  $(du -h --apparent-size $DST/$f 2>/dev/null | cut -f1)"
done
echo "MEMMAP PULL DONE $(date -u +%H:%M:%S)"
df -h / | tail -1
