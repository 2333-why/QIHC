#!/usr/bin/env bash
set -euo pipefail

# Run on the OFFLINE H100 server after transferring and extracting offline_bundle.tar.gz.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BUNDLE_ROOT="${1:?usage: install_offline.sh /absolute/path/offline_bundle}"
VENV_DIR="${2:-${REPO_ROOT}/.venv-formal}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

test -f "${BUNDLE_ROOT}/SHA256SUMS"
(cd "${BUNDLE_ROOT}" && sha256sum --check SHA256SUMS) || {
  echo "Checksum verification failed." >&2
  exit 1
}

nvidia-smi
"${PYTHON_BIN}" -m venv "${VENV_DIR}"
"${VENV_DIR}/bin/python" -m pip install --no-index --find-links "${BUNDLE_ROOT}/wheelhouse" \
  --requirement "${BUNDLE_ROOT}/requirements-locked.txt"
"${VENV_DIR}/bin/python" -m pip install \
  --no-index --find-links "${BUNDLE_ROOT}/wheelhouse" \
  --no-build-isolation --no-deps --editable "${REPO_ROOT}"

cmake -S "${BUNDLE_ROOT}/src/HGS-CVRP" -B "${BUNDLE_ROOT}/src/HGS-CVRP/build" -DCMAKE_BUILD_TYPE=Release
cmake --build "${BUNDLE_ROOT}/src/HGS-CVRP/build" --parallel "$(nproc)"

"${VENV_DIR}/bin/python" - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("gpu_count", torch.cuda.device_count())
assert torch.cuda.is_available()
assert torch.cuda.device_count() == 4
for i in range(torch.cuda.device_count()):
    print(i, torch.cuda.get_device_name(i), torch.cuda.get_device_properties(i).total_memory)
PY

"${VENV_DIR}/bin/python" -m pytest -q "${REPO_ROOT}/tests"

echo "Offline environment installed at ${VENV_DIR}"
