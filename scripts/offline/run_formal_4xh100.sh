#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BUNDLE_ROOT="${1:?usage: run_formal_4xh100.sh BUNDLE_ROOT OUTPUT_ROOT [MODEL_DIR]}"
OUTPUT_ROOT="${2:?usage: run_formal_4xh100.sh BUNDLE_ROOT OUTPUT_ROOT [MODEL_DIR]}"
MODEL_DIR="${3:-${BUNDLE_ROOT}/models/Qwen--Qwen2.5-7B-Instruct}"
VENV_DIR="${VENV_DIR:-${REPO_ROOT}/.venv-formal}"
PYTHON="${VENV_DIR}/bin/python"
TORCHRUN="${VENV_DIR}/bin/torchrun"
HGS_BUILD_DIR="${BUNDLE_ROOT}/src/HGS-CVRP/build"
if [[ -x "${HGS_BUILD_DIR}/hgs" ]]; then
  HGS_BINARY="${HGS_BUILD_DIR}/hgs"
elif [[ -x "${HGS_BUILD_DIR}/bin/hgs" ]]; then
  HGS_BINARY="${HGS_BUILD_DIR}/bin/hgs"
else
  echo "HGS executable not found under ${HGS_BUILD_DIR}" >&2
  exit 1
fi
CVRP_DIR="${BUNDLE_ROOT}/src/HGS-CVRP/Instances/CVRP"
CPU_WORKERS="${CPU_WORKERS:-40}"
STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_ROOT="${OUTPUT_ROOT}/formal_${STAMP}"

mkdir -p "${RUN_ROOT}"
nvidia-smi -L | tee "${RUN_ROOT}/nvidia-smi.txt"
lscpu | tee "${RUN_ROOT}/lscpu.txt"
free -h | tee "${RUN_ROOT}/memory.txt"

"${PYTHON}" "${REPO_ROOT}/experiments/prepare_nlcvrp_jsonl.py" \
  "${CVRP_DIR}" "${RUN_ROOT}/nl_cvrp.jsonl" --limit 100

# Track B0: evaluate whether the LLM really converts natural-language constraints to Constraint IR.
"${TORCHRUN}" --standalone --nproc_per_node=4 \
  "${REPO_ROOT}/experiments/evaluate_constraint_ir.py" \
  --data "${RUN_ROOT}/nl_cvrp.jsonl" \
  --model-path "${MODEL_DIR}" \
  --output "${RUN_ROOT}/constraint_ir" \
  --limit 100 --temperature 0

# Track A1: CPU baselines use 40 processes by default on the 80-core host.
"${TORCHRUN}" --standalone --nproc_per_node="${CPU_WORKERS}" \
  "${REPO_ROOT}/experiments/run_cvrp_lns.py" \
  --dataset cvrplib \
  --data "${CVRP_DIR}" \
  --output "${RUN_ROOT}/track_a_baselines" \
  --limit 100 \
  --search-seeds 0 1 2 3 4 \
  --methods greedy ortools hgs \
  --hgs-binary "${HGS_BINARY}" \
  --baseline-time-limit 30 \
  --sampler numpy

# Track A2: QIHC methods use one process per H100.
"${TORCHRUN}" --standalone --nproc_per_node=4 \
  "${REPO_ROOT}/experiments/run_cvrp_lns.py" \
  --dataset cvrplib \
  --data "${CVRP_DIR}" \
  --output "${RUN_ROOT}/track_a_qihc" \
  --limit 100 \
  --search-seeds 0 1 2 3 4 \
  --methods random knn llm \
  --sampler torch \
  --iterations 100 --patience 30 \
  --destroy-size 12 --routes-per-customer 4 \
  --sampling-steps 1000 --num-chains 2048 --top-samples 32 \
  --model-path "${MODEL_DIR}"

"${PYTHON}" "${REPO_ROOT}/experiments/merge_cvrp_results.py" \
  --inputs "${RUN_ROOT}/track_a_baselines" "${RUN_ROOT}/track_a_qihc" \
  --output "${RUN_ROOT}/track_a_combined"
"${PYTHON}" "${REPO_ROOT}/experiments/analyze_cvrp_results.py" \
  "${RUN_ROOT}/track_a_combined" --reference knn

# Track B1: CPU baselines for controlled natural-language constraints.
"${TORCHRUN}" --standalone --nproc_per_node="${CPU_WORKERS}" \
  "${REPO_ROOT}/experiments/run_cvrp_lns.py" \
  --dataset jsonl \
  --data "${RUN_ROOT}/nl_cvrp.jsonl" \
  --output "${RUN_ROOT}/track_b_baselines" \
  --limit 100 \
  --search-seeds 0 1 2 3 4 \
  --methods greedy ortools \
  --baseline-time-limit 30 \
  --sampler numpy

# Track B2: QIHC methods on four H100s.
"${TORCHRUN}" --standalone --nproc_per_node=4 \
  "${REPO_ROOT}/experiments/run_cvrp_lns.py" \
  --dataset jsonl \
  --data "${RUN_ROOT}/nl_cvrp.jsonl" \
  --output "${RUN_ROOT}/track_b_qihc" \
  --limit 100 \
  --search-seeds 0 1 2 3 4 \
  --methods random knn llm \
  --sampler torch \
  --iterations 100 --patience 30 \
  --destroy-size 12 --routes-per-customer 4 \
  --sampling-steps 1000 --num-chains 2048 --top-samples 32 \
  --model-path "${MODEL_DIR}"

"${PYTHON}" "${REPO_ROOT}/experiments/merge_cvrp_results.py" \
  --inputs "${RUN_ROOT}/track_b_baselines" "${RUN_ROOT}/track_b_qihc" \
  --output "${RUN_ROOT}/track_b_combined"
"${PYTHON}" "${REPO_ROOT}/experiments/analyze_cvrp_results.py" \
  "${RUN_ROOT}/track_b_combined" --reference knn

RESULT_ARCHIVE="${RUN_ROOT}.tar.gz"
RESULT_ARCHIVE_DIR="$(dirname "${RESULT_ARCHIVE}")"
RESULT_ARCHIVE_NAME="$(basename "${RESULT_ARCHIVE}")"
tar -czf "${RESULT_ARCHIVE}" -C "${OUTPUT_ROOT}" "$(basename "${RUN_ROOT}")"
(cd "${RESULT_ARCHIVE_DIR}" && sha256sum "${RESULT_ARCHIVE_NAME}" > "${RESULT_ARCHIVE_NAME}.sha256")
echo "Formal experiment completed: ${RUN_ROOT}"
