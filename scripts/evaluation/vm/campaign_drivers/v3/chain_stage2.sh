#!/usr/bin/env bash
# Queue stage-2 architectures behind whatever is running now.
# Waits on the UNet chain's sentinel line, not a process pattern -- a pgrep
# pattern can match this script's own launching shell and wait on itself.
#
# Order: cheapest-first by measured checkpoint size (391 MB/ckpt x 19 for the
# resnet50 archs). DPT (ViT-B, ~20 GB/run, ~60 GB for 3 seeds) is deliberately
# NOT queued here -- it alone would not fit alongside the others in the 117 GB
# free, so it needs a disk decision first.
set -u
until grep -q "UNET STAGE2 DONE" ~/logs_stage2/unet_chain.log 2>/dev/null; do sleep 120; done
echo "unet done: $(tail -1 ~/logs_stage2/unet_chain.log)"

for tok in unetpp deeplab; do
  echo "########## $tok $(date -u +%H:%M:%S)"
  bash ~/run_stage2_arch.sh "$tok"
  echo "########## $tok finished; free: $(df -h / | tail -1 | awk '{print $4}')"
done
echo "QUEUE DONE $(date -u +%H:%M:%S)"
