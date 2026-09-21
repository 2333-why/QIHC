#!/usr/bin/env bash
# Fast, non-destructive preflight for a new operator before launching GPU work.
set -euo pipefail

WORK_ROOT="${WORK_ROOT:-/hdd/wl2}"
REPO_DIR="${REPO_DIR:-${WORK_ROOT}/QIHC}"
source "${REPO_DIR}/scripts/s2e/activate_qihc.sh"

echo "=== repository ==="
git -C "${REPO_DIR}" status --short --branch
git -C "${REPO_DIR}" log -1 --oneline

echo "=== Python and packages ==="
python - <<'PY'
import sys
import torch
import transformers
print("python", sys.version.split()[0])
print("torch", torch.__version__)
print("transformers", transformers.__version__)
print("cuda", torch.version.cuda, "available=", torch.cuda.is_available())
PY

echo "=== GPUs ==="
nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu \
  --format=csv,noheader

echo "=== model files ==="
python "${REPO_DIR}/experiments/nsfc_evidence/download_model_hf.py" \
  --repo "${MODEL_ID:-Qwen/Qwen3-Coder-30B-A3B-Instruct}" \
  --revision "${MODEL_REVISION:-main}" --local-dir "${MODEL_DIR}" --verify-only
test -s "${MODEL_DIR}/qihc_model_manifest.json"
if [[ "${RUN_MODEL_SMOKE:-0}" == "1" ]]; then
  echo "=== real model load/generation smoke ==="
  CUDA_VISIBLE_DEVICES="${SMOKE_GPU:-0}" python \
    "${REPO_DIR}/experiments/smoke_local_llm.py" \
    --model-path "${MODEL_DIR}" --device cuda:0
else
  echo "Model load smoke skipped; run with RUN_MODEL_SMOKE=1 after confirming GPU 0 is free."
fi

echo "=== HGS ==="
HGS_BINARY="${HGS_BINARY:-${HGS_DIR}/build/hgs}"
[[ -x "${HGS_BINARY}" ]] || HGS_BINARY="${HGS_DIR}/build/bin/hgs"
test -x "${HGS_BINARY}"
echo "hgs=${HGS_BINARY}"

echo "=== repository tests ==="
python -m pytest -q "${REPO_DIR}/tests"

echo "HANDOFF PREFLIGHT PASSED"
