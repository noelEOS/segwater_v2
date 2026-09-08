#!/usr/bin/env bash
# Stage 2 for one v3 architecture, three seeds, strictly serial.
#   usage: run_stage2_arch.sh <token>   (segformer|unetpp|dpt|deeplab)
#
# Params are the Neon HPO winners at FULL precision (read 2026-09-08).
# ce_weight is NOT independent: optimize.py sets ce_weight = 1 - dice_weight,
# so it must be passed or the loss is not the tuned one.
#
# Recipe verified against the Swin-B s42 stage-2 log on Drive: val_check_interval
# 1084, all four trainer.perf flags on, wandb project coastal_water_seg_gold.
# ulimit -n 65536 before launch (descriptor leak killed Swin-B HPO at trial 29).
set -u
TOK="${1:?usage: run_stage2_arch.sh <segformer|unetpp|dpt|deeplab>}"
case "$TOK" in
  segformer) ARCH=segformer;    ENC=mit_b4;                       LR=0.00025910244577562766; WD=5.645099653044466e-05;  LS=0.04052622083682943; DW=0.12766689483301574; CW=0.8723331051669843 ;;
  unetpp)    ARCH=unetplusplus; ENC=resnet50;                     LR=0.0001781268475151213;  WD=5.972484587163602e-06;  LS=0.01510348172375758; DW=0.0035889362006009673; CW=0.996411063799399 ;;
  dpt)       ARCH=dpt;          ENC=tu-vit_base_patch16_224.mae;  LR=7.248330563757252e-05;  WD=2.1268586969564757e-05; LS=0.06199271618710185; DW=0.5036866755937622; CW=0.49631332440623777 ;;
  deeplab)   ARCH=deeplabv3plus;ENC=resnet50;                     LR=0.00028749032648063886; WD=3.0551147740955266e-05; LS=0.09116117727498402; DW=0.2972697508100238; CW=0.7027302491899762 ;;
  *) echo "unknown token $TOK"; exit 2 ;;
esac

ulimit -n 65536
cd ~/segwater_v2
export PATH=$HOME/miniforge3/envs/torch211_cu128/bin:$PATH
export LD_LIBRARY_PATH=$HOME/miniforge3/envs/torch211_cu128/lib
mkdir -p ~/logs_stage2

echo "### $TOK  arch=$ARCH enc=$ENC  ulimit=$(ulimit -n)"
ok=0; fail=0
for SEED in 42 19 58; do
  echo "=== ${TOK} s${SEED} start $(date -u +%H:%M:%S)"
  python scripts/train.py \
    project_name=coastal_water_seg_gold \
    model=smp model.arch=$ARCH model.encoder_name=$ENC \
    model.label_smoothing=$LS model.dice_weight=$DW model.ce_weight=$CW \
    trainer.base_learning_rate=$LR trainer.weight_decay=$WD \
    trainer.epochs=10 trainer.val_check_interval=1084 \
    trainer.perf.tf32=true trainer.perf.cudnn_benchmark=true \
    trainer.perf.compile=true trainer.perf.fused_adamw=true \
    data.memmap_root=/home/noel/memmaps_v3 data.dtype=float16 \
    data.strata_index_path=data/v3_strata/full_val_strata.npz \
    seed=$SEED \
    > ~/logs_stage2/stage2_v3_${TOK}_s${SEED}.log 2>&1
  rc=$?; [ $rc -eq 0 ] && ok=$((ok+1)) || fail=$((fail+1))
  echo "=== ${TOK} s${SEED} exit $rc $(date -u +%H:%M:%S)"
  df -h / | tail -1
done
echo "${TOK} STAGE2 DONE $(date -u +%H:%M:%S)  ok=$ok fail=$fail"
