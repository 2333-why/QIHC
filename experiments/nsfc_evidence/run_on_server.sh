#!/usr/bin/env bash
# NSFC evidence suite — run on server with 4× H200
#
# Usage:
#   cd /path/to/QIHC
#   bash experiments/nsfc_evidence/run_on_server.sh
#
# Optional env:
#   MODEL_NAME=Qwen/Qwen2.5-7B-Instruct
#   CUDA_VISIBLE_DEVICES=0,1,2,3
#   PROFILE=server          # or smoke
#   HF_HOME=/data/hf_cache

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

# shellcheck disable=SC1091
source "${SCRIPT_DIR}/cn_mirror_env.sh"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-7B-Instruct}"
export PROFILE="${PROFILE:-server}"
export HF_HOME="${HF_HOME:-${REPO_ROOT}/.cache/huggingface}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

TS="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${REPO_ROOT}/experiments/outputs/nsfc_evidence/run_${TS}"
mkdir -p "${RUN_DIR}"

echo "=============================================="
echo " QIHC NSFC Evidence Suite"
echo " Repo:    ${REPO_ROOT}"
echo " Profile: ${PROFILE}"
echo " Model:   ${MODEL_NAME}"
echo " GPUs:    ${CUDA_VISIBLE_DEVICES}"
echo " Run dir: ${RUN_DIR}"
echo "=============================================="

# Tee full console log
exec > >(tee -a "${RUN_DIR}/console.log") 2>&1

echo "[$(date -Iseconds)] Creating venv if missing..."
if [[ ! -d "${REPO_ROOT}/.venv" ]]; then
  python3 -m venv "${REPO_ROOT}/.venv"
fi
# shellcheck disable=SC1091
source "${REPO_ROOT}/.venv/bin/activate"

echo "[$(date -Iseconds)] Installing package (PyPI: ${PIP_INDEX_URL})..."
pip install -q -U pip "${PIP_INSTALL_ARGS[@]}"
pip install -q "${PIP_INSTALL_ARGS[@]}" -e ".[dev,hf,llm]"

echo "[$(date -Iseconds)] Prefetch HF BBH cache (optional)..."
python experiments/download_bbh_hf.py --limit-per-task 50 || true

echo "[$(date -Iseconds)] Running pytest smoke..."
pytest -q tests/test_vci.py tests/test_bbh.py || echo "WARN: some tests failed"

echo "[$(date -Iseconds)] Starting evidence suite profile=${PROFILE}..."
python experiments/nsfc_evidence/run_evidence_suite.py \
  --profile "${PROFILE}" \
  --run-dir "${RUN_DIR}"

echo "[$(date -Iseconds)] Packaging results..."
ARCHIVE="${RUN_DIR}/nsfc_evidence_bundle.tar.gz"
tar -czf "${ARCHIVE}" -C "${RUN_DIR}" .

echo "=============================================="
echo " DONE"
echo " Results:  ${RUN_DIR}"
echo " Bundle:   ${ARCHIVE}"
echo " Download: scp user@server:${ARCHIVE} ."
echo "=============================================="
