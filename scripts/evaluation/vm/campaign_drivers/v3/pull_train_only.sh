#!/usr/bin/env bash
# Pull ONLY train.memmap (278.5 GB) from Drive to this VM.
#
# val.memmap is deliberately NOT included: train alone leaves ~50 GB free here,
# and train+val (358.6 GB) does not fit in the 329 GB available. Nothing is
# being deleted to make room -- that is a pending decision.
#
# --multi-thread-streams 16 splits the single large file across 16 connections.
# --transfers would NOT help: there is one file, so file-level parallelism is 1.
# Measured ~540 MB/s this way on the sister VM vs ~180 MB/s without.
set -u
mkdir -p ~/memmaps_v3
rclone copy gdrive:Segwater_v2_RAW_DATASET/memmaps_v3/train.memmap ~/memmaps_v3/ \
  --multi-thread-streams 16 --multi-thread-cutoff 128M \
  --transfers 1 --checkers 8 --drive-chunk-size 256M \
  --stats 60s --stats-one-line --log-level INFO \
  --log-file ~/pull_train_only.log
echo "RCLONE EXIT $? $(date -u +%H:%M:%S)"
ls -la ~/memmaps_v3/
df -h / | tail -1
