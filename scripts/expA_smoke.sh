#!/bin/bash
# Sequential Exp A smoke runs inside an allocation (one GPU): every arm, 1 seed, 10% of the steps.
#   srun --jobid=<dev session> --overlap --gpus=1 bash scripts/expA_smoke.sh [LR] [ARMS...]
# Writes runs/smoke_<arm>; each run also evaluates cal / select / dev and measures per-type gradient shares.
set -uo pipefail
source "${REPO:-/projects/u6xe/louisk/open-rlcd}/slurm/env.sh"
LR=${1:-1e-4}; shift || true
ARMS=${*:-a1 a5 a3 a4 a6 a6r}
cd "$REPO"
for ARM in $ARMS; do
  EXTRA=""
  case $ARM in a6|a6r) EXTRA="--pool_frac 0.1 --n_per_round 150";; esac
  echo "=== smoke $ARM $(date -u +%H:%M:%S)"
  $PY -m rlcd.train_labels --arm $ARM --lr $LR --seed 0 --steps_frac 0.1 --grad_share_steps 2,20,40 $EXTRA \
    --out "$PROJ/rlcd_expA/runs/smoke_$ARM" --data "$PROJ/rlcd_expA/data" \
    --reference "$PROJ/rlcd_expA/ref/ref_logp.pkl" > "$PROJ/rlcd_expA/logs/smoke_$ARM.log" 2>&1 \
    && echo "ok $ARM" || echo "FAILED $ARM (see logs/smoke_$ARM.log)"
done
