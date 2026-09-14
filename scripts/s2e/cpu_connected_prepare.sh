#!/usr/bin/env bash
set -euo pipefail

export WORK_ROOT="${WORK_ROOT:-/inspire/hdd/project/project-public/public/why}"
export GLOBAL_ROOT="${GLOBAL_ROOT:-/inspire/hdd/global_user/yanjunchi-24040/qihc}"
export CONDA_ROOT="${CONDA_ROOT:-${WORK_ROOT}/miniforge3}"
export CONDA_ENVS_PATH="${WORK_ROOT}/conda-envs"
export REPO_DIR="${WORK_ROOT}/QIHC"
export BUNDLE_ROOT="${GLOBAL_ROOT}/offline_bundle_s2e"
export MODEL_ID="${MODEL_ID:-Qwen/Qwen2.5-32B-Instruct}"
export MODEL_SLUG="${MODEL_ID//\//--}"
export MODEL_DIR="${GLOBAL_ROOT}/models/${MODEL_SLUG}"
export TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu124}"
export PIP_CACHE_DIR="${GLOBAL_ROOT}/cache/pip"
export HF_HOME="${GLOBAL_ROOT}/cache/huggingface"
mkdir -p "${WORK_ROOT}" "${CONDA_ENVS_PATH}" "${GLOBAL_ROOT}/models" "${BUNDLE_ROOT}/wheelhouse" "${BUNDLE_ROOT}/src"
source "${CONDA_ROOT}/etc/profile.d/conda.sh"
if [[ ! -d "${CONDA_ENVS_PATH}/qihc" ]]; then conda create -y -p "${CONDA_ENVS_PATH}/qihc" python=3.11 pip; fi
conda activate "${CONDA_ENVS_PATH}/qihc"
conda install -y -c conda-forge cmake ninja cxx-compiler binutils
python -m pip install --upgrade pip wheel setuptools
python -m pip install --index-url "${TORCH_INDEX_URL}" torch
python -m pip install -r "${REPO_DIR}/requirements-training.txt"
python -m pip install -e "${REPO_DIR}"
python -m pip freeze --all | grep -Ev '^(-e |qihc==|qihc @)' > "${BUNDLE_ROOT}/requirements-locked.txt"
python -m pip download --dest "${BUNDLE_ROOT}/wheelhouse" --extra-index-url "${TORCH_INDEX_URL}" -r "${BUNDLE_ROOT}/requirements-locked.txt"
if [[ ! -f "${MODEL_DIR}/config.json" ]]; then
  for attempt in 1 2 3 4 5; do
    hf download "${MODEL_ID}" --local-dir "${MODEL_DIR}" --max-workers "${HF_WORKERS:-4}" && break
    [[ "${attempt}" == 5 ]] && exit 1
    echo "model download retry ${attempt}/5; existing files will be resumed"
  done
fi
if [[ ! -d "${BUNDLE_ROOT}/src/HGS-CVRP/.git" ]]; then git clone --depth 1 https://github.com/vidalt/HGS-CVRP.git "${BUNDLE_ROOT}/src/HGS-CVRP"; fi
git -C "${REPO_DIR}" bundle create "${BUNDLE_ROOT}/qihc.git.bundle" --all
git -C "${REPO_DIR}" rev-parse HEAD > "${BUNDLE_ROOT}/QIHC_COMMIT.txt"
(cd "${BUNDLE_ROOT}" && find wheelhouse src -type f -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS)
echo "READY model=${MODEL_DIR} bundle=${BUNDLE_ROOT}"
