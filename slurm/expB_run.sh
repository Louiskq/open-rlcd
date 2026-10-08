#!/bin/bash
#SBATCH --job-name=rlcd-expB
#SBATCH --account=brics.u6xe
#SBATCH --partition=workq
#SBATCH --gpus=1
#SBATCH --time=05:00:00
# One Exp B RL run (B1 / B3 / B5): train, then evaluate the selected and the final checkpoint (games and beliefs).
#   sbatch --output=$PROJ/rlcd_expB/logs/%x-%j.out slurm/expB_run.sh NAME ARM SEED [more `breakout_cli train` args]
# e.g. sbatch ... slurm/expB_run.sh b1_s0 b1 0
# Needs decider's venv (slurm/env.sh). Writes $PROJ/rlcd_expB/runs/NAME.
set -euo pipefail
source "${REPO:-/projects/u6xe/louisk/open-rlcd}/slurm/env.sh"
export SDL_VIDEODRIVER=dummy
NAME=$1; ARM=$2; SEED=$3; shift 3
RUN=$PROJ/rlcd_expB/runs/$NAME
cd "$REPO"
$PY -c "import triton, sys; v = triton.__version__; print('[env] triton', v); sys.exit(v != '3.7.1')" \
  || { echo "[env] triton must be 3.7.1"; exit 1; }
COMMON=${COMMON:-}   # optional: the common-state questions (H7 rule); answered by both checkpoints when the file exists
SCORE=""; [ -n "$COMMON" ] && [ -f "$COMMON" ] && SCORE="--score common=$COMMON"
$PY -m rlcd.breakout_cli train --arm "$ARM" --seed "$SEED" --out "$RUN" "$@"
SEL=$($PY -c "import json; print(json.load(open('$RUN/run.json'))['selected_iter'])")
echo "[expB] selected iteration $SEL"
# the selected checkpoint; the start model itself if training never beat it on the selection seeds
if [ "$SEL" = "0" ]; then CK=$PROJ/models/decider9b-sft; else CK=$RUN/model_best; fi
$PY -m rlcd.breakout_cli eval --ckpt "$CK" --out "$RUN/eval_best" --seed "$SEED" $SCORE
$PY -m rlcd.breakout_cli eval --ckpt "$RUN/model_last" --out "$RUN/eval_last" --seed "$SEED" $SCORE
