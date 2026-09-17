#!/usr/bin/env bash
# Source this file after every login on the online 2x RTX PRO 6000 server.
set -euo pipefail
export WORK_ROOT="${WORK_ROOT:-/hdd/wl2}"
export GLOBAL_ROOT="${GLOBAL_ROOT:-${WORK_ROOT}}"
export CONDA_ROOT="${CONDA_ROOT:-${WORK_ROOT}/miniforge3}"
export CONDA_ENVS_PATH="${CONDA_ENVS_PATH:-${WORK_ROOT}/conda-envs}"
export PIP_CACHE_DIR="${GLOBAL_ROOT}/cache/pip"
export HF_HOME="${GLOBAL_ROOT}/cache/huggingface"
export TRANSFORMERS_CACHE="${HF_HOME}/hub"
export HF_DATASETS_CACHE="${GLOBAL_ROOT}/cache/datasets"
export TORCH_HOME="${GLOBAL_ROOT}/cache/torch"
export MODELSCOPE_CACHE="${GLOBAL_ROOT}/cache/modelscope"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export REPO_DIR="${REPO_DIR:-${WORK_ROOT}/QIHC}"
export RUNTIME_ROOT="${RUNTIME_ROOT:-${WORK_ROOT}/runtime}"
export HGS_DIR="${HGS_DIR:-${RUNTIME_ROOT}/src/HGS-CVRP}"
export MODEL_DIR="${MODEL_DIR:-${GLOBAL_ROOT}/models/Qwen--Qwen2.5-32B-Instruct}"
export CONDARC="${CONDARC:-${REPO_DIR}/configs/condarc-pro6000.yaml}"
mkdir -p "${CONDA_ENVS_PATH}" "${PIP_CACHE_DIR}" "${HF_HOME}" "${HF_DATASETS_CACHE}" "${TORCH_HOME}" "${MODELSCOPE_CACHE}"
if [[ ! -f "${CONDA_ROOT}/etc/profile.d/conda.sh" ]]; then
  echo "Conda is missing at ${CONDA_ROOT}; run scripts/s2e/pro6000_online_setup.sh first." >&2
  return 1 2>/dev/null || exit 1
fi
source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENVS_PATH}/qihc"
cd "${REPO_DIR}"
