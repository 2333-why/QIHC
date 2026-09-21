#!/usr/bin/env bash
# Restartable, no-p-bit/no-repair direct-LLM NL-CVRP experiment.
set -euo pipefail

WORK_ROOT="${WORK_ROOT:-/hdd/wl2}"
REPO_DIR="${REPO_DIR:-${WORK_ROOT}/QIHC}"
GLOBAL_ROOT="${GLOBAL_ROOT:-${WORK_ROOT}}"
MODEL_ID="${MODEL_ID:-Qwen/Qwen3.5-35B-A3B}"
MODEL_SLUG="${MODEL_ID//\//--}"
MODEL_DIR="${MODEL_DIR:-${GLOBAL_ROOT}/models/${MODEL_SLUG}}"
RUN_ROOT="${RUN_ROOT:-${WORK_ROOT}/results/direct_llm_qwen35_cvrp_v1}"
BENCHMARK_DIR="${BENCHMARK_DIR:-${WORK_ROOT}/data/CVRPLIB/X}"
MIN_CUSTOMERS="${MIN_CUSTOMERS:-100}"
MAX_CUSTOMERS="${MAX_CUSTOMERS:-800}"
CASE_LIMIT="${CASE_LIMIT:-30}"
CLAUSES_PER_TYPE="${CLAUSES_PER_TYPE:-4}"
SEARCH_SEEDS="${SEARCH_SEEDS:-0 1 2}"
LLM_DIRECT_MAX_NEW_TOKENS="${LLM_DIRECT_MAX_NEW_TOKENS:-12288}"
LLM_MAX_INPUT_TOKENS="${LLM_MAX_INPUT_TOKENS:-32768}"
LLM_TEMPERATURE="${LLM_TEMPERATURE:-0.2}"
export MODEL_DIR

source "${REPO_DIR}/scripts/s2e/activate_qihc.sh"
cd "${REPO_DIR}"
mkdir -p "${BENCHMARK_DIR}" "${RUN_ROOT}/data" "${RUN_ROOT}/logs"
test -s "${MODEL_DIR}/config.json"
test -s "${MODEL_DIR}/qihc_model_manifest.json"
cp "${MODEL_DIR}/qihc_model_manifest.json" "${RUN_ROOT}/model_manifest.json"

python experiments/download_hard_cvrplib.py "${BENCHMARK_DIR}" \
  --min-customers "${MIN_CUSTOMERS}" --max-customers "${MAX_CUSTOMERS}" \
  --limit "${CASE_LIMIT}" --selection spread \
  >"${RUN_ROOT}/logs/download.log" 2>&1

DATA="${RUN_ROOT}/data/direct_nlcvrp.jsonl"
python experiments/prepare_hard_nlcvrp.py "${BENCHMARK_DIR}" "${DATA}" \
  --min-customers "${MIN_CUSTOMERS}" --max-customers "${MAX_CUSTOMERS}" \
  --limit "${CASE_LIMIT}" --selection spread --clauses-per-type "${CLAUSES_PER_TYPE}" \
  >"${RUN_ROOT}/logs/prepare.log" 2>&1

# The direct model receives the raw numerical instance and natural-language
# requirements only. It emits routes once; no p-bit, OR-Tools, HGS, completion,
# greedy incumbent, or repair is reachable on this path.
# shellcheck disable=SC2086
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}" \
torchrun --standalone --nproc_per_node=2 experiments/run_cvrp_lns.py \
  --dataset jsonl --data "${DATA}" --output "${RUN_ROOT}/direct_llm" \
  --methods llm_direct --search-seeds ${SEARCH_SEEDS} --sampler torch \
  --constraint-source dataset --model-path "${MODEL_DIR}" \
  --llm-temperature "${LLM_TEMPERATURE}" \
  --llm-direct-max-new-tokens "${LLM_DIRECT_MAX_NEW_TOKENS}" \
  --llm-max-input-tokens "${LLM_MAX_INPUT_TOKENS}" --resume \
  >"${RUN_ROOT}/logs/direct_llm.log" 2>&1

# shellcheck disable=SC2086
python experiments/analyze_direct_llm_cvrp.py \
  --data "${DATA}" --output "${RUN_ROOT}/direct_llm" --seeds ${SEARCH_SEEDS} \
  >"${RUN_ROOT}/logs/report.log" 2>&1

echo "COMPLETED: ${RUN_ROOT}/direct_llm_report.json"
