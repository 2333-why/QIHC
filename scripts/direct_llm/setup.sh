#!/usr/bin/env bash
# One-time environment and model setup for the standalone direct-LLM branch.
set -euo pipefail

WORK_ROOT="${WORK_ROOT:-/hdd/wl2}"
REPO_DIR="${REPO_DIR:-${WORK_ROOT}/QIHC}"
GLOBAL_ROOT="${GLOBAL_ROOT:-${WORK_ROOT}}"
MODEL_ID="${MODEL_ID:-Qwen/Qwen3.5-35B-A3B}"
MODEL_SLUG="${MODEL_ID//\//--}"
MODEL_DIR="${MODEL_DIR:-${GLOBAL_ROOT}/models/${MODEL_SLUG}}"
export MODEL_ID MODEL_DIR

source "${REPO_DIR}/scripts/s2e/activate_qihc.sh"

# The Qwen3.5 AutoModelForMultimodalLM path needs a newer Transformers than
# the causal-only project baseline. This applies only to this direct-solve setup.
python -m pip install --upgrade "transformers>=4.57,<5" "accelerate>=0.33"
python -m pip install -e "${REPO_DIR}"

MODEL_ID="${MODEL_ID}" MODEL_DIR="${MODEL_DIR}" \
  bash "${REPO_DIR}/scripts/s2e/prepare_model.sh"

echo "DIRECT-LLM SETUP READY"
echo "model=${MODEL_DIR}"
