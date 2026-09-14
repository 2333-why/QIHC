#!/usr/bin/env bash
# Source this file after every CPU/GPU login: source scripts/s2e/activate_qihc.sh
set -euo pipefail
export WORK_ROOT="${WORK_ROOT:-/inspire/hdd/project/project-public/public/why}"
export GLOBAL_ROOT="${GLOBAL_ROOT:-/inspire/hdd/global_user/yanjunchi-24040/qihc}"
export CONDA_ROOT="${CONDA_ROOT:-${WORK_ROOT}/miniforge3}"
export CONDA_ENVS_PATH="${WORK_ROOT}/conda-envs"
export PIP_CACHE_DIR="${GLOBAL_ROOT}/cache/pip"
export HF_HOME="${GLOBAL_ROOT}/cache/huggingface"
export TRANSFORMERS_CACHE="${HF_HOME}/hub"
export HF_DATASETS_CACHE="${GLOBAL_ROOT}/cache/datasets"
export TORCH_HOME="${GLOBAL_ROOT}/cache/torch"
export REPO_DIR="${WORK_ROOT}/QIHC"
export BUNDLE_ROOT="${GLOBAL_ROOT}/offline_bundle_s2e"
export MODEL_DIR="${MODEL_DIR:-${GLOBAL_ROOT}/models/Qwen--Qwen2.5-32B-Instruct}"
mkdir -p "${CONDA_ENVS_PATH}" "${PIP_CACHE_DIR}" "${HF_HOME}" "${HF_DATASETS_CACHE}" "${TORCH_HOME}"
source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENVS_PATH}/qihc"
cd "${REPO_DIR}"
