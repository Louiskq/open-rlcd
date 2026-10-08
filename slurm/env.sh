# source me: the cluster environment for Exp A (decider's venv, this repo and decider on PYTHONPATH).
# Paths are the maintainer's Isambard-AI project; override PROJ for another setup.
export PROJ=${PROJ:-/projects/u6xe/louisk}
export REPO=${REPO:-$PROJ/open-rlcd}
export HF_HOME=$PROJ/hf HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 FLA_TILELANG=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=$REPO:$PROJ/decider
export PY=$PROJ/decider/.venv/bin/python
# Slurm points TMPDIR at /local/user/<uid>, which the prolog had not created (or could not write) on some nodes on
# 2026-10-08, so `import transformers` died with PermissionError inside torch's inductor cache. Use a per-job /tmp dir.
export TMPDIR=/tmp/${USER}_${SLURM_JOB_ID:-0}
mkdir -p "$TMPDIR"
export TORCHINDUCTOR_CACHE_DIR=$TMPDIR/inductor
export TRITON_CACHE_DIR=$TMPDIR/triton
