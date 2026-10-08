# source me: the cluster environment for Exp A (decider's venv, this repo and decider on PYTHONPATH).
# Paths are the maintainer's Isambard-AI project; override PROJ for another setup.
export PROJ=${PROJ:-/projects/u6xe/louisk}
export REPO=${REPO:-$PROJ/open-rlcd}
export HF_HOME=$PROJ/hf HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 FLA_TILELANG=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=$REPO:$PROJ/decider
export PY=$PROJ/decider/.venv/bin/python
