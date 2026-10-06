#!/bin/bash
#SBATCH --job-name=rlcd-expA
#SBATCH --account=brics.u6xe
#SBATCH --partition=workq
#SBATCH --gpus=1
#SBATCH --time=03:30:00
# One Exp A run (train + evaluate cal / select / dev). One GH200; about 2 h at the full step count.
#   sbatch --output=$PROJ/rlcd_expA/logs/%x-%j.out slurm/expA_run.sh NAME ARM LR SEED [more rlcd.train_labels args]
# e.g. sbatch ... slurm/expA_run.sh a1_lr1e-4_s0 a1 1e-4 0
# NAME is the output directory under $PROJ/rlcd_expA/runs. Needs the data and reference from `rlcd.train_labels
# --compute_reference` (see slurm/env.sh and scripts/prep_expA.py).
set -euo pipefail
source "${REPO:-/projects/u6xe/louisk/open-rlcd}/slurm/env.sh"
NAME=$1; ARM=$2; LR=$3; SEED=$4; shift 4
cd "$REPO"
$PY -c "import triton, sys; v = triton.__version__; print('[env] triton', v); sys.exit(v != '3.7.1')" \
  || { echo "[env] triton must be 3.7.1"; exit 1; }
$PY -m rlcd.train_labels --arm "$ARM" --lr "$LR" --seed "$SEED" \
  --out "$PROJ/rlcd_expA/runs/$NAME" --data "$PROJ/rlcd_expA/data" \
  --reference "$PROJ/rlcd_expA/ref/ref_logp.pkl" "$@"
