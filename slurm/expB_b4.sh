#!/bin/bash
#SBATCH --job-name=rlcd-expB-b4
#SBATCH --account=brics.u6xe
#SBATCH --partition=workq
#SBATCH --gpus=1
#SBATCH --time=02:30:00
# B4 for one seed: fit B1 seed S's final policy on the belief questions of its own training rollouts (log score only), then
# answer the common-state questions and B1's selection-seed / evaluation-seed questions (for the temperature and the
# on-policy comparison). Needs B1 seed S finished and $COMMON (the common-state file).
#   sbatch --output=$PROJ/rlcd_expB/logs/%x-%j.out slurm/expB_b4.sh SEED
set -euo pipefail
source "${REPO:-/projects/u6xe/louisk/open-rlcd}/slurm/env.sh"
export SDL_VIDEODRIVER=dummy
S=$1
B1=$PROJ/rlcd_expB/runs/b1_s$S
OUT=$PROJ/rlcd_expB/runs/b4_s$S
COMMON=${COMMON:-$PROJ/rlcd_expB/common/common_states.jsonl}
cd "$REPO"
$PY -m rlcd.breakout_cli fit-beliefs --b1_run "$B1" --out "$OUT" --seed "$S"
$PY -m rlcd.breakout_cli eval --ckpt "$OUT/model_last" --out "$OUT/eval" --seed "$S" --no_games \
  --score "common=$COMMON,sel_b1=$B1/eval_last/onpolicy_select.jsonl,ev_b1=$B1/eval_last/onpolicy_eval.jsonl"
rm -rf "$OUT/model_last"   # 18 GB; the answers are saved, the fit is reproducible from B1 seed S
