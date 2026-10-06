#!/bin/bash
# Run detached on the login node: when the three A1 sweep runs (seed 0) have finished, pick the learning rate by the
# pre-declared rule (lowest selection-split log loss, diverged runs excluded) and submit A1 seed 1 at that rate.
# Seed 1 and the sweep's chosen seed-0 run are the two runs the delta margins are computed from (D13).
#   setsid nohup bash scripts/a1_followup.sh > $PROJ/rlcd_expA/logs/a1_followup.log 2>&1 < /dev/null &
source "${REPO:-/projects/u6xe/louisk/open-rlcd}/slurm/env.sh"
RUNS=$PROJ/rlcd_expA/runs
cd "$REPO"
echo "[followup] waiting for the A1 sweep $(date -u +%H:%M:%S)"
while true; do
  n=$(ls $RUNS/a1_lr*_s0/run.json 2>/dev/null | wc -l)
  [ "$n" -ge 3 ] && break
  if [ "$(squeue -u "$USER" -h -n a1-lr5e-5,a1-lr1e-4,a1-lr2e-4 | wc -l)" -eq 0 ]; then
    echo "[followup] no A1 sweep jobs left in the queue and only $n finished runs: stopping"; exit 1
  fi
  sleep 120
done
LR=$($PY scripts/pick_lr.py "$RUNS" a1) || { echo "[followup] pick_lr failed"; exit 1; }
echo "[followup] chosen learning rate: $LR $(date -u +%H:%M:%S)"
sbatch --parsable --job-name="a1-lr${LR}-s1" --output="$PROJ/rlcd_expA/logs/%x-%j.out" slurm/expA_run.sh "a1_lr${LR}_s1" a1 "$LR" 1 \
  --grad_share_steps 2,100,300,600
echo "[followup] submitted seed 1"
