#!/bin/bash
# Exp B stage 1, run detached on the login node while the three B1 runs train (they are submitted separately):
#   setsid nohup bash scripts/expB_stage1.sh > $PROJ/rlcd_expB/logs/stage1.log 2>&1 < /dev/null &
# 1. when B1 seed 0 has finished, collect the common states from its final checkpoint (H7 rule (a));
# 2. when B1 seed i has finished and the common states exist, run B4 seed i (fit on B1 seed i's own rollouts).
# It submits nothing that trains B3 or B5: those wait for results/expB/delta_game.json to be committed (stage 2).
source "${REPO:-/projects/u6xe/louisk/open-rlcd}/slurm/env.sh"
RUNS=$PROJ/rlcd_expB/runs; LOGS=$PROJ/rlcd_expB/logs; COMMON=$PROJ/rlcd_expB/common/common_states.jsonl
cd "$REPO"
done_b1() { [ -f "$RUNS/b1_s$1/eval_last/games.json" ]; }
alive() { [ "$(squeue -u "$USER" -h -n "b1_s$1" | wc -l)" -gt 0 ]; }
echo "[stage1] started $(date -u +%H:%M:%S)"
until done_b1 0; do
  alive 0 || { done_b1 0 || { echo "[stage1] b1_s0 is not running and not finished: stopping"; exit 1; }; }
  sleep 120
done
CJ=$(sbatch --parsable --job-name=expB-common --output="$LOGS/%x-%j.out" slurm/expB_common.sh)
echo "[stage1] b1_s0 finished; common-state job $CJ"
until [ -f "$COMMON" ]; do
  [ "$(squeue -u "$USER" -h -n expB-common | wc -l)" -gt 0 ] || [ -f "$COMMON" ] || { echo "[stage1] common job vanished without output"; exit 1; }
  sleep 60
done
echo "[stage1] common states ready: $(wc -l < "$COMMON") questions"
for S in 0 1 2; do
  until done_b1 $S; do
    alive $S || done_b1 $S || { echo "[stage1] b1_s$S failed"; exit 1; }
    sleep 120
  done
  J=$(sbatch --parsable --job-name="b4_s$S" --output="$LOGS/%x-%j.out" --export=ALL,COMMON="$COMMON" slurm/expB_b4.sh "$S")
  echo "[stage1] b1_s$S finished; B4 seed $S job $J"
done
echo "[stage1] done $(date -u +%H:%M:%S)"
