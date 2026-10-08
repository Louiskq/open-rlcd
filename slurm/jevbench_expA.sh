#!/bin/bash
#SBATCH --job-name=jevbench-expA
#SBATCH --account=brics.u6xe
#SBATCH --partition=workq
#SBATCH --gpus=1
#SBATCH --time=04:00:00
# JevBench public items (easy, original, hard) for several Exp A models: merge each LoRA adapter into the start model,
# serve it with decider, run JevBench through its typesafe adapter, keep the summaries, delete the merged weights.
#   sbatch --output=$PROJ/rlcd_expA/logs/%x-%j.out slurm/jevbench_expA.sh RUN_NAME [RUN_NAME ...]
# RUN_NAME is a directory under $PROJ/rlcd_expA/runs (or `a0`, the unmodified start model served at T = 1).
# Results: $PROJ/rlcd_expA/jevbench/RUN_NAME/{summary-all,summary-easy,summary-original,summary-hard}.json
set -uo pipefail
source "${REPO:-/projects/u6xe/louisk/open-rlcd}/slurm/env.sh"
export FLA_TILELANG=0
JEV=$PROJ/jevbench
OUT=$PROJ/rlcd_expA/jevbench
mkdir -p "$OUT"
cd "$REPO"
for NAME in "$@"; do
  [ -f "$OUT/$NAME/summary-hard.json" ] && { echo "[jev] $NAME already done"; continue; }
  M=$PROJ/rlcd_expA/merged/$NAME
  rm -rf "$M"
  if [ "$NAME" = "a0" ]; then
    mkdir -p "$M"; cp -r "$PROJ"/models/decider9b-sft/* "$M"/
    $PY -c "import json,sys; p='$M/decider_config.json'; c=json.load(open(p)); c['temperature']=1.0; json.dump(c, open(p,'w'))"
  else
    $PY scripts/merge_expA_adapter.py --adapter "$PROJ/rlcd_expA/runs/$NAME/lora_adapter.pt" --out "$M" --temperature 1.0 \
      || { echo "[jev] merge failed for $NAME"; continue; }
  fi
  PORT=$((20000 + (SLURM_JOB_ID + RANDOM) % 10000))
  cd "$PROJ/decider"
  DECIDER_MODEL=$M $PY -m uvicorn decider.serve:app --host 127.0.0.1 --port $PORT > "$OUT/serve_$NAME.log" 2>&1 &
  SERVER=$!
  for i in $(seq 1 120); do
    curl -sf http://127.0.0.1:$PORT/health | grep -q '"ok":true' && break
    kill -0 $SERVER 2>/dev/null || { echo "[jev] server crashed for $NAME"; break; }
    sleep 5
  done
  RUN=$OUT/$NAME
  mkdir -p "$RUN"
  cd "$JEV"
  TASKS=datasets/public/easy.jsonl,datasets/public/original.jsonl,datasets/public/hard.jsonl
  $PY -m jevbench.cli run --tasks $TASKS --adapter typesafe --endpoint http://127.0.0.1:$PORT --key-env "" \
    --model "$NAME" --cost-basis self_hosted --reserve-usd 0 \
    --results "$RUN/results.jsonl" --raw-dir "$RUN/raw" --ledger "$RUN/ledger.jsonl" --run-label "$NAME"
  kill $SERVER 2>/dev/null; wait $SERVER 2>/dev/null
  $PY -m jevbench.cli summarize --tasks $TASKS --results "$RUN/results.jsonl" --public-export "$RUN/summary-all.json"
  for T in easy original hard; do
    $PY - "$RUN" "$T" << 'PY'
import json, sys
d, t = sys.argv[1], sys.argv[2]
ids = {(r.get("id") or r.get("task_id")) for r in (json.loads(l) for l in open(f"datasets/public/{t}.jsonl") if l.strip())}
with open(f"{d}/results-{t}.jsonl", "w") as out:
    for l in open(f"{d}/results.jsonl"):
        if l.strip() and json.loads(l)["task_id"] in ids:
            out.write(l)
PY
    $PY -m jevbench.cli summarize --tasks datasets/public/$T.jsonl --results "$RUN/results-$T.jsonl" --public-export "$RUN/summary-$T.json"
  done
  rm -rf "$M"
  echo "[jev] done $NAME"
  cd "$REPO"
done
