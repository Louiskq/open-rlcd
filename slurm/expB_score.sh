#!/bin/bash
#SBATCH --job-name=rlcd-expB-score
#SBATCH --account=brics.u6xe
#SBATCH --partition=workq
#SBATCH --gpus=1
#SBATCH --time=02:00:00
# Answer the common-state questions with the selected and the final checkpoint of finished runs (no new games).
#   sbatch --output=$PROJ/rlcd_expB/logs/%x-%j.out slurm/expB_score.sh RUN_NAME [RUN_NAME ...]
set -euo pipefail
source "${REPO:-/projects/u6xe/louisk/open-rlcd}/slurm/env.sh"
export SDL_VIDEODRIVER=dummy
COMMON=${COMMON:-$PROJ/rlcd_expB/common/common_states.jsonl}
cd "$REPO"
for NAME in "$@"; do
  RUN=$PROJ/rlcd_expB/runs/$NAME
  SEL=$($PY -c "import json; print(json.load(open('$RUN/run.json'))['selected_iter'])")
  if [ "$SEL" = "0" ]; then CK=$PROJ/models/decider9b-sft; else CK=$RUN/model_best; fi
  $PY -m rlcd.breakout_cli eval --ckpt "$CK" --out "$RUN/eval_best" --no_games --score "common=$COMMON"
  $PY -m rlcd.breakout_cli eval --ckpt "$RUN/model_last" --out "$RUN/eval_last" --no_games --score "common=$COMMON"
done
