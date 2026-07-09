#!/usr/bin/env bash
# QIHC three-law suite (NE1–NE9, Tier 0/1/2) — server background run
#
# Usage (foreground):
#   cd /path/to/QIHC
#   bash experiments/nsfc_evidence/run_on_server_three_law.sh
#
# Usage (background):
#   nohup bash experiments/nsfc_evidence/run_on_server_three_law.sh \
#     > experiments/outputs/nsfc_evidence/three_law_launcher.log 2>&1 &
#   echo $!
#
# Optional env:
#   PROFILE=tier012|smoke|tier0|tier1|tier2|full
#   CUDA_VISIBLE_DEVICES=0,1
#   SKIP_PYTEST=1

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

# shellcheck disable=SC1091
source "${SCRIPT_DIR}/cn_mirror_env.sh"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export PROFILE="${PROFILE:-tier012}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

TS="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${REPO_ROOT}/experiments/outputs/nsfc_evidence/run_three_law_${TS}"
mkdir -p "${RUN_DIR}"

echo "=============================================="
echo " QIHC Three-Law Suite (profile=${PROFILE})"
echo " Repo:    ${REPO_ROOT}"
echo " GPUs:    ${CUDA_VISIBLE_DEVICES}"
echo " Run dir: ${RUN_DIR}"
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

echo "[$(date -Iseconds)] Installing package (PyPI: ${PIP_INDEX_URL})..."
pip install -q -U pip "${PIP_INSTALL_ARGS[@]}"
pip install -q "${PIP_INSTALL_ARGS[@]}" -e ".[dev]"

echo "[$(date -Iseconds)] Quick import check..."
python - <<'PY'
from qihc.theory import (
    mean_field_fixed_point,
    generate_higher_order_instance,
    tv_bound_stale_field,
    optimal_lambda,
)
print("theory imports OK")
PY

if [[ "${SKIP_PYTEST:-0}" != "1" ]]; then
  echo "[$(date -Iseconds)] Running lightweight theory smoke..."
  python experiments/nsfc_evidence/run_ne2_contraction.py --profile smoke \
    --output-dir "${RUN_DIR}/_precheck_ne2" || echo "WARN: precheck ne2 failed"
fi

echo "[$(date -Iseconds)] Starting three-law suite profile=${PROFILE}..."
python experiments/nsfc_evidence/run_three_law_suite.py \
  --profile "${PROFILE}" \
  --run-dir "${RUN_DIR}"

echo "[$(date -Iseconds)] Packaging results..."
ARCHIVE="${RUN_DIR}/nsfc_three_law_bundle.tar.gz"
tar -czf "${ARCHIVE}" -C "${RUN_DIR}" . || echo "WARN: tar reported changes (non-fatal)"

echo "=============================================="
echo " DONE"
echo " Results:  ${RUN_DIR}"
echo " Bundle:   ${ARCHIVE}"
echo " Index:    ${RUN_DIR}/three_law_index.json"
echo " Tail log: tail -f ${RUN_DIR}/console.log"
echo "=============================================="
