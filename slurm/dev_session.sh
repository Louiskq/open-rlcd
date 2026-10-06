#!/bin/bash
#SBATCH --job-name=rlcd-dev
#SBATCH --account=brics.u6xe
#SBATCH --partition=workq
#SBATCH --gpus=1
#SBATCH --time=05:00:00
# A held GPU allocation, so commands can be run inside it with `srun --jobid=<id> --overlap ...` without a
# queue wait for each one (queue waits here are hours). It only sleeps; cancel it (`scancel <id>`) as soon as the
# work is done so it does not bill idle time.
#   sbatch --output=$PROJ/logs/rlcd-dev-%j.out slurm/dev_session.sh
echo "[dev] started on $(hostname) at $(date -u)"
sleep 5h
