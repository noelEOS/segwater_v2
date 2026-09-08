#!/usr/bin/env bash
# Pull val.memmap (80.1 GB) to the EVAL VM, completing the train+val pair.
# Same multi-thread settings that gave ~540-750 MB/s: --multi-thread-streams
# splits ONE file across N connections; --transfers would not help at n=1.
set -u
rclone copy gdrive:Segwater_v2_RAW_DATASET/memmaps_v3/val.memmap ~/memmaps_v3/ \
  --multi-thread-streams 16 --multi-thread-cutoff 128M \
  --transfers 1 --checkers 8 --drive-chunk-size 256M \
  --stats 60s --stats-one-line --log-level INFO \
  --log-file ~/pull_val_only.log
echo "RCLONE EXIT $? $(date -u +%H:%M:%S)"
ls -la ~/memmaps_v3/
df -h / | tail -1
