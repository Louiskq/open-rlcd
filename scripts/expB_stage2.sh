#!/bin/bash
# Exp B stage 2: run AFTER results/expB/delta_game.json has been committed (PREREGISTRATION.md Amendment 2, item 6).
#   setsid nohup bash scripts/expB_stage2.sh DELTA_GAME_COMMIT_SHA > $PROJ/rlcd_expB/logs/stage2.log 2>&1 < /dev/null &
# Submits B3 (3 seeds) and B5 (1 seed), each answering the common-state questions after training, and scores B1's
# checkpoints on the same questions.
PRE=${1:?usage: expB_stage2.sh DELTA_GAME_COMMIT_SHA}
source "${REPO:-/projects/u6xe/louisk/open-rlcd}/slurm/env.sh"
LOGS=$PROJ/rlcd_expB/logs; COMMON=$PROJ/rlcd_expB/common/common_states.jsonl
[ -f "$COMMON" ] || { echo "[stage2] no common states yet"; exit 1; }
cd "$REPO"
echo "[stage2] delta_game commit $PRE; $(date -u +%H:%M:%S)"
for S in 0 1 2; do
  J=$(sbatch --parsable --job-name="b3_s$S" --time=05:00:00 --output="$LOGS/%x-%j.out" --export=ALL,COMMON="$COMMON" slurm/expB_run.sh "b3_s$S" b3 "$S")
  echo "[stage2] submitted b3_s$S job $J"
done
J=$(sbatch --parsable --job-name=b5_s0 --time=05:00:00 --output="$LOGS/%x-%j.out" --export=ALL,COMMON="$COMMON" slurm/expB_run.sh b5_s0 b5 0)
echo "[stage2] submitted b5_s0 job $J"
J=$(sbatch --parsable --job-name=b1-score --output="$LOGS/%x-%j.out" --export=ALL,COMMON="$COMMON" slurm/expB_score.sh b1_s0 b1_s1 b1_s2)
echo "[stage2] submitted B1 common-state scoring job $J"
