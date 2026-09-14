#!/usr/bin/env bash
set -euo pipefail
export WORK_ROOT="${WORK_ROOT:-/inspire/hdd/project/project-public/public/why}"
export GLOBAL_ROOT="${GLOBAL_ROOT:-/inspire/hdd/global_user/yanjunchi-24040/qihc}"
export CONDA_ROOT="${CONDA_ROOT:-${WORK_ROOT}/miniforge3}"
export CONDA_ENVS_PATH="${WORK_ROOT}/conda-envs"
export REPO_DIR="${WORK_ROOT}/QIHC"
export BUNDLE_ROOT="${GLOBAL_ROOT}/offline_bundle_s2e"
source "${CONDA_ROOT}/etc/profile.d/conda.sh"
if [[ ! -d "${CONDA_ENVS_PATH}/qihc" ]]; then conda create -y -p "${CONDA_ENVS_PATH}/qihc" python=3.11 pip; fi
conda activate "${CONDA_ENVS_PATH}/qihc"
(cd "${BUNDLE_ROOT}" && sha256sum --check SHA256SUMS)
python -m pip install --no-index --find-links "${BUNDLE_ROOT}/wheelhouse" -r "${BUNDLE_ROOT}/requirements-locked.txt"
python -m pip install --no-index --find-links "${BUNDLE_ROOT}/wheelhouse" --no-build-isolation --no-deps -e "${REPO_DIR}"
HGS_DIR="${BUNDLE_ROOT}/src/HGS-CVRP"
AR_BIN="$(command -v x86_64-conda-linux-gnu-ar || command -v ar)"
RANLIB_BIN="$(command -v x86_64-conda-linux-gnu-ranlib || command -v ranlib)"
cmake -S "${HGS_DIR}" -B "${HGS_DIR}/build" -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_AR="${AR_BIN}" -DCMAKE_RANLIB="${RANLIB_BIN}"
cmake --build "${HGS_DIR}/build" --parallel "${BUILD_JOBS:-40}"
python - <<'PY'
import torch
assert torch.cuda.is_available(), "CUDA unavailable"
assert torch.cuda.device_count() == 4, torch.cuda.device_count()
for i in range(4): print(i, torch.cuda.get_device_name(i), torch.cuda.get_device_properties(i).total_memory)
PY
python -m pytest -q "${REPO_DIR}/tests"
echo "OFFLINE INSTALL PASSED"
