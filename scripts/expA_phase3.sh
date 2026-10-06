#!/bin/bash
# Phase 3 launcher for Exp A, run detached on the login node AFTER PREREGISTRATION.md has been committed.
#   setsid nohup bash scripts/expA_phase3.sh PREREG_COMMIT_SHA > $PROJ/rlcd_expA/logs/phase3_driver.log 2>&1 < /dev/null &
# Wave 1 (no dependencies): A1 seed 2, the A4 learning-rate sweep (seed 0), A5, A6 and A6r (3 seeds each) at the A1 rate.
# Wave 2 (after the A4 sweep, by the same selection-split rule): A4 seeds 1-2, A3 seeds 0-2 at the A4 rate, and the
# exploratory A3 with beta = 0 (E1). A1 seeds 0 and 1 are the sweep's chosen run and the delta seed already run.
# Jobs are single-GPU Slurm jobs of slurm/expA_run.sh; the driver only submits and picks rates.
PRE=${1:?usage: expA_phase3.sh PREREG_COMMIT_SHA}
source "${REPO:-/projects/u6xe/louisk/open-rlcd}/slurm/env.sh"
RUNS=$PROJ/rlcd_expA/runs; LOGS=$PROJ/rlcd_expA/logs
cd "$REPO"
echo "[phase3] pre-registration commit $PRE; started $(date -u +%H:%M:%S)"
LR1=$($PY -c "import json; print(json.load(open('$RUNS/sweep_a1.json'))['chosen'])")
echo "[phase3] A1 rate (from sweep_a1.json): $LR1"

sub() {  # NAME ARM LR SEED TIME [extra runner args]
  local name=$1 arm=$2 lr=$3 seed=$4 time=$5; shift 5
  local id
  id=$(sbatch --parsable --job-name="$name" --time="$time" --output="$LOGS/%x-%j.out" \
    slurm/expA_run.sh "$name" "$arm" "$lr" "$seed" --grad_share_steps 2,100,300,600 "$@") \
    || { echo "[phase3] FAILED to submit $name"; return 1; }
  echo "[phase3] submitted $name job $id"
}

sub "a1_lr${LR1}_s2" a1 "$LR1" 2 03:30:00
for LR in 5e-5 1e-4 2e-4; do sub "a4_lr${LR}_s0" a4 "$LR" 0 03:30:00; done
sub "a5_lr${LR1}_s0" a5 "$LR1" 0 03:30:00
for S in 0 1 2; do
  sub "a6_lr${LR1}_s$S" a6 "$LR1" "$S" 04:30:00
  sub "a6r_lr${LR1}_s$S" a6r "$LR1" "$S" 04:30:00
done

echo "[phase3] waiting for the A4 sweep $(date -u +%H:%M:%S)"
while true; do
  n=$(ls $RUNS/a4_lr*_s0/run.json 2>/dev/null | wc -l)
  [ "$n" -ge 3 ] && break
  if [ "$(squeue -u "$USER" -h -n a4_lr5e-5_s0,a4_lr1e-4_s0,a4_lr2e-4_s0 | wc -l)" -eq 0 ]; then
    echo "[phase3] no A4 sweep jobs left in the queue and only $n finished runs: stopping"; exit 1
  fi
  sleep 120
done
LR4=$($PY scripts/pick_lr.py "$RUNS" a4) || { echo "[phase3] pick_lr a4 failed"; exit 1; }
echo "[phase3] A4 rate (also used for A3): $LR4 $(date -u +%H:%M:%S)"
for S in 1 2; do sub "a4_lr${LR4}_s$S" a4 "$LR4" "$S" 03:30:00; done
for S in 0 1 2; do sub "a3_lr${LR4}_s$S" a3 "$LR4" "$S" 03:30:00; done
sub "a3beta0_lr${LR4}_s0" a3 "$LR4" 0 03:30:00 --beta 0
echo "[phase3] all runs submitted $(date -u +%H:%M:%S)"
