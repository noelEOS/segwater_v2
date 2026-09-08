#!/usr/bin/env bash
# Stage 2 -- UNet-R50 (v3 lineage), HPO trial 59, three seeds, strictly serial.
#
# Recipe verified against the Swin-B s42 stage-2 log on Drive, not just the doc:
# val_check_interval 1084, all four trainer.perf flags on, wandb project
# coastal_water_seg_gold. Deploy arm is *_last.pth.
#
# ulimit -n 65536 BEFORE launching: the Swin-B HPO study died at trial 29 on
# "Too many open files" (~40 descriptors leak per trial) under the 1024 default.
#
# ce_weight is NOT independent -- optimize.py sets ce_weight = 1 - dice_weight,
# so it must be passed explicitly here or the loss is not the tuned one.
set -u
ulimit -n 65536
cd ~/segwater_v2
export PATH=$HOME/miniforge3/envs/torch211_cu128/bin:$PATH
export LD_LIBRARY_PATH=$HOME/miniforge3/envs/torch211_cu128/lib

LR=0.0002546471995934161
WD=0.0491539984753506
LS=0.026221519594578396
DW=0.3835833677509212
CW=0.6164166322490787

echo "ulimit -n = $(ulimit -n)"
ok=0; fail=0
for SEED in 42 19 58; do
  echo "=== unet_resnet50 s${SEED} start $(date -u +%H:%M:%S)"
  python scripts/train.py \
    project_name=coastal_water_seg_gold \
    model=smp model.arch=unet model.encoder_name=resnet50 \
    model.label_smoothing=$LS model.dice_weight=$DW model.ce_weight=$CW \
    trainer.base_learning_rate=$LR trainer.weight_decay=$WD \
    trainer.epochs=10 trainer.val_check_interval=1084 \
    trainer.perf.tf32=true trainer.perf.cudnn_benchmark=true \
    trainer.perf.compile=true trainer.perf.fused_adamw=true \
    data.memmap_root=/home/noel/memmaps_v3 data.dtype=float16 \
    data.strata_index_path=data/v3_strata/full_val_strata.npz \
    seed=$SEED \
    > ~/logs_stage2/stage2_v3_unet_s${SEED}.log 2>&1
  rc=$?; [ $rc -eq 0 ] && ok=$((ok+1)) || fail=$((fail+1))
  echo "=== unet_resnet50 s${SEED} exit $rc $(date -u +%H:%M:%S)"
  df -h / | tail -1
done
echo "UNET STAGE2 DONE $(date -u +%H:%M:%S)  ok=$ok fail=$fail"
