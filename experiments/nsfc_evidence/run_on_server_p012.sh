#!/usr/bin/env bash
# NSFC P0–P2 evidence suite — Pro 6000 (2× GPU) background run
#
# Usage (foreground):
#   cd /hdd/why/QIHC
#   bash experiments/nsfc_evidence/run_on_server_p012.sh
#
# Usage (background):
#   nohup bash experiments/nsfc_evidence/run_on_server_p012.sh \
#     > experiments/outputs/nsfc_evidence/p012_launcher.log 2>&1 &
#   echo $!   # save PID
#
# Optional env:
#   MODEL_NAME=Qwen/Qwen2.5-7B-Instruct
#   MODEL_NAME_14B=Qwen/Qwen2.5-14B-Instruct
#   CUDA_VISIBLE_DEVICES=0,1
#   HF_HOME=/hdd/why/.cache/huggingface
#   SKIP_PYTEST=1

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-7B-Instruct}"
export MODEL_NAME_14B="${MODEL_NAME_14B:-Qwen/Qwen2.5-14B-Instruct}"
export PROFILE="${PROFILE:-p012}"
export HF_HOME="${HF_HOME:-${REPO_ROOT}/.cache/huggingface}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

TS="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${REPO_ROOT}/experiments/outputs/nsfc_evidence/run_p012_${TS}"
mkdir -p "${RUN_DIR}"

echo "=============================================="
echo " QIHC NSFC P0–P2 Suite (profile=${PROFILE})"
echo " Repo:         ${REPO_ROOT}"
echo " Model 7B:     ${MODEL_NAME}"
echo " Model 14B:    ${MODEL_NAME_14B}"
echo " GPUs:         ${CUDA_VISIBLE_DEVICES}"
echo " Run dir:      ${RUN_DIR}"
echo "=============================================="

exec > >(tee -a "${RUN_DIR}/console.log") 2>&1

echo "[$(date -Iseconds)] Host: $(hostname)"
echo "[$(date -Iseconds)] nvidia-smi:"
nvidia-smi || true

echo "[$(date -Iseconds)] Creating venv if missing..."
if [[ ! -d "${REPO_ROOT}/.venv" ]]; then
  python3 -m venv "${REPO_ROOT}/.venv"
fi
# shellcheck disable=SC1091
source "${REPO_ROOT}/.venv/bin/activate"

echo "[$(date -Iseconds)] Installing package..."
pip install -q -U pip
pip install -q -e ".[dev,hf,llm]"

echo "[$(date -Iseconds)] Generating synthetic 200-task dataset..."
python -c "
from qihc.orchestrator.constrained_data import load_synthetic_tasks
load_synthetic_tasks(n_tasks=200, seed=42, regenerate=False)
print('synthetic dataset ready')
"

echo "[$(date -Iseconds)] Prefetch HF BBH cache..."
python experiments/download_bbh_hf.py --limit-per-task 50 || true

if [[ "${SKIP_PYTEST:-0}" != "1" ]]; then
  echo "[$(date -Iseconds)] Running pytest smoke..."
  pytest -q tests/test_vci.py tests/test_bbh.py || echo "WARN: some tests failed"
fi

echo "[$(date -Iseconds)] Starting evidence suite profile=${PROFILE}..."
python experiments/nsfc_evidence/run_evidence_suite.py \
  --profile "${PROFILE}" \
  --run-dir "${RUN_DIR}"

echo "[$(date -Iseconds)] Packaging results..."
ARCHIVE="${RUN_DIR}/nsfc_p012_bundle.tar.gz"
tar -czf "${ARCHIVE}" -C "${RUN_DIR}" . || echo "WARN: tar reported changes (non-fatal)"

echo "=============================================="
echo " DONE"
echo " Results:  ${RUN_DIR}"
echo " Bundle:   ${ARCHIVE}"
echo " Tail log: tail -f ${RUN_DIR}/console.log"
echo "=============================================="
