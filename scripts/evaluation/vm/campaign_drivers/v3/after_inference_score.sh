#!/usr/bin/env bash
# Wait for the SDS inference chain to finish, verify the run dirs are complete,
# then score. Waits on the chain LOG (a sentinel line), not on a process
# pattern: an earlier chain pgrep-ed a pattern its own launching shell matched,
# and waited on itself forever.
set -u
until grep -q "SDS ALL DONE" ~/logs_v3/sds_chain.log 2>/dev/null; do sleep 30; done
echo "inference finished: $(tail -1 ~/logs_v3/sds_chain.log)"

cd ~/segwater_v2
export PATH=$HOME/miniforge3/envs/torch211_cu128_inference/bin:$PATH
export LD_LIBRARY_PATH=$HOME/miniforge3/envs/torch211_cu128_inference/lib

# Completeness gate per site BEFORE scoring: a short run dir yields a plausible
# number on the wrong sample, which is what is hardest to catch downstream.
short=0
for g in narrabeen duck torreypines trucvert; do
  python scripts/evaluation/vm/completion.py --gate "$g" \
    --check outputs/inference/runs/v3_sds_${g}_* || short=1
done
if [ "$short" -ne 0 ]; then
  echo "ABORT: at least one run dir is short -- not scoring."
  exit 1
fi
echo "all run dirs complete; scoring"
bash ~/score_sds_v3.sh
