#!/bin/bash
# Launch Exp A smoke arms concurrently as steps of an existing allocation, detached from the caller's session so an
# SSH drop cannot kill them (a plain `nohup srun ... &` inside an ssh command did not survive one).
#   setsid nohup bash scripts/expA_wave.sh JOBID ARM [ARM...] > /dev/null 2>&1 < /dev/null &
JOB=$1; shift
source "${REPO:-/projects/u6xe/louisk/open-rlcd}/slurm/env.sh"
for ARM in "$@"; do
  EXTRA=""
  case $ARM in a6|a6r) EXTRA="--pool_frac 0.1 --n_per_round 150";; esac
  rm -rf "$PROJ/rlcd_expA/runs/smoke_$ARM"
  srun --jobid="$JOB" --overlap --ntasks=1 --gpus=1 --cpus-per-task=8 bash -c "source $REPO/slurm/env.sh; cd $REPO; \
    $PY -m rlcd.train_labels --arm $ARM --lr 1e-4 --seed 0 --steps_frac 0.1 --grad_share_steps 2,10,30,50 $EXTRA \
    --reference $PROJ/rlcd_expA/ref/ref_logp.pkl --out $PROJ/rlcd_expA/runs/smoke_$ARM" \
    > "$PROJ/rlcd_expA/logs/smoke_$ARM.log" 2>&1 &
done
wait
