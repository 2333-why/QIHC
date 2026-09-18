#!/usr/bin/env bash
set -euo pipefail

# One-machine, online setup for 2x NVIDIA RTX PRO 6000 Blackwell.
export WORK_ROOT="${WORK_ROOT:-/hdd/wl2}"
export GLOBAL_ROOT="${GLOBAL_ROOT:-${WORK_ROOT}}"
export CONDA_ROOT="${CONDA_ROOT:-${WORK_ROOT}/miniforge3}"
export CONDA_ENVS_PATH="${CONDA_ENVS_PATH:-${WORK_ROOT}/conda-envs}"
export REPO_DIR="${REPO_DIR:-${WORK_ROOT}/QIHC}"
export RUNTIME_ROOT="${RUNTIME_ROOT:-${WORK_ROOT}/runtime}"
export HGS_DIR="${HGS_DIR:-${RUNTIME_ROOT}/src/HGS-CVRP}"
export MODEL_ID="${MODEL_ID:-Qwen/Qwen2.5-32B-Instruct}"
export MODEL_SLUG="${MODEL_ID//\//--}"
export MODEL_DIR="${MODEL_DIR:-${GLOBAL_ROOT}/models/${MODEL_SLUG}}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-${GLOBAL_ROOT}/cache/pip}"
export HF_HOME="${HF_HOME:-${GLOBAL_ROOT}/cache/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${GLOBAL_ROOT}/cache/datasets}"
export TORCH_HOME="${TORCH_HOME:-${GLOBAL_ROOT}/cache/torch}"
export MODELSCOPE_CACHE="${MODELSCOPE_CACHE:-${GLOBAL_ROOT}/cache/modelscope}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-0}"
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-600}"
export HF_HUB_ETAG_TIMEOUT="${HF_HUB_ETAG_TIMEOUT:-120}"
export MODEL_DOWNLOAD_BACKEND="${MODEL_DOWNLOAD_BACKEND:-auto}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export CONDARC="${CONDARC:-${REPO_DIR}/configs/condarc-pro6000.yaml}"

MINIFORGE_VERSION="${MINIFORGE_VERSION:-24.11.3-2}"
MINIFORGE_SHA256="${MINIFORGE_SHA256:-65af53dad30b3fcbd1cb1d4ad62fd3a86221464754844544558aae3a28795189}"
TORCH_VERSION="${TORCH_VERSION:-2.8.0}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"
INSTALLER="${WORK_ROOT}/downloads/Miniforge3-${MINIFORGE_VERSION}-Linux-x86_64.sh"

mkdir -p \
  "${WORK_ROOT}/downloads" "${CONDA_ENVS_PATH}" "${RUNTIME_ROOT}/src" \
  "${GLOBAL_ROOT}/models" "${PIP_CACHE_DIR}" "${HF_HOME}" \
  "${HF_DATASETS_CACHE}" "${TORCH_HOME}" "${MODELSCOPE_CACHE}"

if [[ ! -d "${REPO_DIR}/.git" ]]; then
  echo "QIHC repository not found at ${REPO_DIR}. Clone it there before running setup." >&2
  exit 1
fi

if [[ ! -f "${CONDA_ROOT}/etc/profile.d/conda.sh" ]]; then
  if [[ -f "${INSTALLER}" ]] && ! echo "${MINIFORGE_SHA256}  ${INSTALLER}" | sha256sum --check - >/dev/null 2>&1; then
    rm -f "${INSTALLER}"
  fi
  if [[ ! -f "${INSTALLER}" ]]; then
    curl --fail --location --retry 5 --retry-delay 5 \
      --output "${INSTALLER}" \
      "https://github.com/conda-forge/miniforge/releases/download/${MINIFORGE_VERSION}/Miniforge3-${MINIFORGE_VERSION}-Linux-x86_64.sh"
  fi
  echo "${MINIFORGE_SHA256}  ${INSTALLER}" | sha256sum --check -
  if [[ -e "${CONDA_ROOT}" ]]; then
    mv "${CONDA_ROOT}" "${CONDA_ROOT}.incomplete.$(date +%Y%m%d_%H%M%S)"
  fi
  bash "${INSTALLER}" -b -p "${CONDA_ROOT}"
fi

# shellcheck disable=SC1091
source "${CONDA_ROOT}/etc/profile.d/conda.sh"
if [[ -d "${CONDA_ENVS_PATH}/qihc" && ! -x "${CONDA_ENVS_PATH}/qihc/bin/python" ]]; then
  mv "${CONDA_ENVS_PATH}/qihc" "${CONDA_ENVS_PATH}/qihc.incomplete.$(date +%Y%m%d_%H%M%S)"
fi
if [[ ! -d "${CONDA_ENVS_PATH}/qihc" ]]; then
  conda create -y --override-channels -c conda-forge \
    -p "${CONDA_ENVS_PATH}/qihc" python=3.11 pip
fi
conda activate "${CONDA_ENVS_PATH}/qihc"

conda install -y --override-channels -c conda-forge cmake ninja cxx-compiler binutils
python -m pip install --upgrade pip wheel setuptools
python -m pip install --index-url "${TORCH_INDEX_URL}" "torch==${TORCH_VERSION}"
python -m pip install -r "${REPO_DIR}/requirements-training.txt"
python -m pip install -e "${REPO_DIR}"
python -m pip install modelscope

for attempt in 1 2 3; do
  python "${REPO_DIR}/experiments/nsfc_evidence/download_model_hf.py" \
    --repo "${MODEL_ID}" \
    --local-dir "${MODEL_DIR}" \
    --cache-dir "${HF_HOME}/hub" \
    --backend "${MODEL_DOWNLOAD_BACKEND}" && break
  if [[ "${attempt}" == 3 ]]; then
    echo "Model download failed after three resumable ModelScope/HF-mirror attempts." >&2
    exit 1
  fi
  echo "Model download attempt ${attempt}/3 failed; retrying with existing partial files."
done

if [[ ! -d "${HGS_DIR}/.git" ]]; then
  git clone --depth 1 https://github.com/vidalt/HGS-CVRP.git "${HGS_DIR}"
else
  git -C "${HGS_DIR}" pull --ff-only
fi

AR_BIN="$(command -v x86_64-conda-linux-gnu-ar || command -v ar)"
RANLIB_BIN="$(command -v x86_64-conda-linux-gnu-ranlib || command -v ranlib)"
cmake -S "${HGS_DIR}" -B "${HGS_DIR}/build" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_AR="${AR_BIN}" -DCMAKE_RANLIB="${RANLIB_BIN}"
cmake --build "${HGS_DIR}/build" --parallel "${BUILD_JOBS:-32}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" torchrun --standalone --nproc_per_node=2 \
  "${REPO_DIR}/experiments/check_pro6000_environment.py" --expected-gpus 2
python -m pytest -q "${REPO_DIR}/tests"

echo "ONLINE 2xPRO6000 SETUP PASSED"
echo "environment=${CONDA_ENVS_PATH}/qihc"
echo "model=${MODEL_DIR}"
echo "hgs=${HGS_DIR}/build/hgs"
