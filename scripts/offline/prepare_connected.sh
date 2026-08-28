#!/usr/bin/env bash
set -euo pipefail

# Run on a NETWORKED Linux x86_64 machine with the same Python minor version as the H100 server.
# Environment overrides:
#   MODEL_ID=Qwen/Qwen2.5-7B-Instruct
#   TORCH_INDEX_URL=https://download.pytorch.org/whl/cu124
#   PYTHON_BIN=python3

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BUNDLE_ROOT="${1:-${REPO_ROOT}/offline_bundle}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
MODEL_ID="${MODEL_ID:-Qwen/Qwen2.5-7B-Instruct}"
MODEL_SLUG="${MODEL_ID//\//--}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu124}"

mkdir -p "${BUNDLE_ROOT}/wheelhouse" "${BUNDLE_ROOT}/models/${MODEL_SLUG}" "${BUNDLE_ROOT}/src" "${BUNDLE_ROOT}/data"

BOOTSTRAP_VENV="${BUNDLE_ROOT}/.download-venv"
"${PYTHON_BIN}" -m venv "${BOOTSTRAP_VENV}"
"${BOOTSTRAP_VENV}/bin/python" -m pip install --upgrade pip
"${BOOTSTRAP_VENV}/bin/python" -m pip install --index-url "${TORCH_INDEX_URL}" torch
"${BOOTSTRAP_VENV}/bin/python" -m pip install --requirement "${REPO_ROOT}/requirements-formal.txt"
"${BOOTSTRAP_VENV}/bin/python" -m pip freeze --all > "${BUNDLE_ROOT}/requirements-locked.txt"
"${BOOTSTRAP_VENV}/bin/python" -m pip download \
  --dest "${BUNDLE_ROOT}/wheelhouse" \
  --extra-index-url "${TORCH_INDEX_URL}" \
  --requirement "${BUNDLE_ROOT}/requirements-locked.txt"
"${BOOTSTRAP_VENV}/bin/hf" download "${MODEL_ID}" \
  --local-dir "${BUNDLE_ROOT}/models/${MODEL_SLUG}"

if [[ ! -d "${BUNDLE_ROOT}/src/HGS-CVRP/.git" ]]; then
  git clone --depth 1 https://github.com/vidalt/HGS-CVRP.git "${BUNDLE_ROOT}/src/HGS-CVRP"
fi
if [[ ! -d "${BUNDLE_ROOT}/src/ARS-Routbench/.git" ]]; then
  git clone --depth 1 https://github.com/Ahalikai/ARS-Routbench.git "${BUNDLE_ROOT}/src/ARS-Routbench"
fi

git -C "${REPO_ROOT}" bundle create "${BUNDLE_ROOT}/qihc.git.bundle" --all
git -C "${REPO_ROOT}" rev-parse HEAD > "${BUNDLE_ROOT}/QIHC_COMMIT.txt"

(cd "${BUNDLE_ROOT}" && \
  find . -type f ! -path './.download-venv/*' ! -name 'SHA256SUMS' -print0 \
    | sort -z \
    | xargs -0 sha256sum > SHA256SUMS)

ARCHIVE_PATH="${BUNDLE_ROOT}.tar.gz"
ARCHIVE_DIR="$(dirname "${ARCHIVE_PATH}")"
ARCHIVE_NAME="$(basename "${ARCHIVE_PATH}")"
tar --exclude='.download-venv' -czf "${ARCHIVE_PATH}" -C "$(dirname "${BUNDLE_ROOT}")" "$(basename "${BUNDLE_ROOT}")"
(cd "${ARCHIVE_DIR}" && sha256sum "${ARCHIVE_NAME}" > "${ARCHIVE_NAME}.sha256")

echo "Offline bundle: ${ARCHIVE_PATH}"
echo "Model directory: ${BUNDLE_ROOT}/models/${MODEL_SLUG}"
