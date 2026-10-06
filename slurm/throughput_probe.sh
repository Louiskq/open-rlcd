#!/bin/bash
#SBATCH --job-name=rlcd-probe
#SBATCH --account=brics.u6xe
#SBATCH --partition=workq
#SBATCH --gpus=1
#SBATCH --time=00:35:00
# Throughput probe for the Exp A budget (phase 2 estimate). One GH200, under 30 minutes.
#   sbatch --output=$PROJ/logs/rlcd-probe-%j.out slurm/throughput_probe.sh
#
# A. LoRA training on the Stage 1 SFT checkpoint with the *same settings* as the old arm_scaleAll run
#    (rank 64, accum 4, 16,384 tokens per micro-batch), 40 optimizer steps on the real question pool:
#    current tokens/s and peak memory.
# B. Forward-only scoring of the independent dev set (2,791 questions) at two batch sizes: the cost
#    model for scoring A6's candidate pools.
# Nothing here touches an existing run: all outputs go under runs/_probe_<jobid>, which is removed after.
set -euo pipefail
PROJ=${PROJ:-/projects/u6xe/louisk}
export HF_HOME=$PROJ/hf HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 FLA_TILELANG=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$PROJ/decider"
PY=.venv/bin/python
$PY -c "import triton, sys; v = triton.__version__; print('[env] triton', v); sys.exit(v != '3.7.1')" \
  || { echo "[env] triton must be 3.7.1"; exit 1; }
H=$PROJ/hardgen
M=$PROJ/models/decider9b-sft
OUT=$H/runs/_probe_${SLURM_JOB_ID}
echo "[probe] node $(hostname); gpu $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"

echo "[probe] A: LoRA training, 40 optimizer steps"
t0=$(date +%s)
$PY -m decider.train --model "$M" --data "$H/data/arm_scaleAll.pkl" --out "$OUT/train" \
  --epochs 2 --lr 1e-4 --warmup 20 --max_tokens 16384 --accum 4 --max_options 255 --max_ctx 16384 \
  --none_prob 0.1 --schema_first_prob 0.5 --eval_every 100000 --eval_limit 150 --seed 0 \
  --lora_r 64 --lora_alpha 128 --resume --ckpt_every_min 600 --stop_step 40
echo "[probe] A wall seconds: $(( $(date +%s) - t0 ))"

for BS in 16 48; do
  echo "[probe] B: forward-only scoring of dev_new.pkl, batch size $BS"
  t1=$(date +%s)
  $PY -m decider.evaluate --model "$M" --data "$H/data/dev_new.pkl" --out "$OUT/eval_bs$BS" \
    --engine graph --bs $BS --max_options 255 --max_ctx 16384 --layout state_first
  echo "[probe] B bs=$BS wall seconds: $(( $(date +%s) - t1 ))"
done
nvidia-smi --query-gpu=memory.used --format=csv
rm -rf "$OUT"
echo "[probe] done"
