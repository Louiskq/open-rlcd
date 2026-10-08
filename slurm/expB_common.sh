#!/bin/bash
#SBATCH --job-name=rlcd-expB-common
#SBATCH --account=brics.u6xe
#SBATCH --partition=workq
#SBATCH --gpus=1
#SBATCH --time=00:40:00
# The common states of the H7 rule: sampled rollouts of B1 seed 0's final checkpoint on the evaluation seeds, with the
# belief questions asked and resolved as in training. Every arm answers exactly these questions.
#   sbatch --output=$PROJ/rlcd_expB/logs/%x-%j.out slurm/expB_common.sh
set -euo pipefail
source "${REPO:-/projects/u6xe/louisk/open-rlcd}/slurm/env.sh"
export SDL_VIDEODRIVER=dummy
cd "$REPO"
mkdir -p "$PROJ/rlcd_expB/common"
$PY -m rlcd.breakout_cli collect-common --ckpt "$PROJ/rlcd_expB/runs/b1_s0/model_last" \
  --out "$PROJ/rlcd_expB/common/common_states.jsonl"
