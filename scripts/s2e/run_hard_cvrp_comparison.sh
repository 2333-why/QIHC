#!/usr/bin/env bash
# Hard, restartable two-GPU CVRPLIB-X comparison for QIHC, classical solvers and LLM-only.
set -euo pipefail

REPO_DIR="${REPO_DIR:-/hdd/wl2/QIHC}"
source "${REPO_DIR}/scripts/s2e/activate_qihc.sh"
cd "${REPO_DIR}"

MODEL_DIR="${MODEL_DIR:-/hdd/wl2/models/Qwen--Qwen3-Coder-30B-A3B-Instruct}"
BENCHMARK_DIR="${BENCHMARK_DIR:-/hdd/wl2/data/CVRPLIB/X}"
RUN_ROOT="${RUN_ROOT:-/hdd/wl2/results/qihc_cvrplib_x400_1000_v1}"
MIN_CUSTOMERS="${MIN_CUSTOMERS:-400}"
MAX_CUSTOMERS="${MAX_CUSTOMERS:-1000}"
CASE_LIMIT="${CASE_LIMIT:-12}"
SELECTION="${SELECTION:-spread}"
CLAUSES_PER_TYPE="${CLAUSES_PER_TYPE:-4}"
SEARCH_SEEDS="${SEARCH_SEEDS:-0 1 2}"
ITERATIONS="${ITERATIONS:-40}"
PATIENCE="${PATIENCE:-20}"
DESTROY_SIZE="${DESTROY_SIZE:-20}"
ROUTES_PER_CUSTOMER="${ROUTES_PER_CUSTOMER:-8}"
SAMPLING_STEPS="${SAMPLING_STEPS:-160}"
NUM_CHAINS="${NUM_CHAINS:-2048}"
TOP_SAMPLES="${TOP_SAMPLES:-128}"
BASELINE_TIME_LIMIT="${BASELINE_TIME_LIMIT:-300}"
LLM_DIRECT_MAX_NEW_TOKENS="${LLM_DIRECT_MAX_NEW_TOKENS:-8192}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
mkdir -p "${BENCHMARK_DIR}" "${RUN_ROOT}/data" "${RUN_ROOT}/logs"
test -s "${MODEL_DIR}/config.json"
if (( TOP_SAMPLES > NUM_CHAINS )); then
  echo "TOP_SAMPLES must not exceed NUM_CHAINS" >&2
  exit 2
fi

python experiments/download_hard_cvrplib.py "${BENCHMARK_DIR}" \
  --min-customers "${MIN_CUSTOMERS}" --max-customers "${MAX_CUSTOMERS}" \
  --limit "${CASE_LIMIT}" --selection "${SELECTION}" \
  >"${RUN_ROOT}/logs/download.log" 2>&1

DATA="${RUN_ROOT}/data/hard_nlcvrp.jsonl"
CPP="${RUN_ROOT}/cpp"
python experiments/prepare_hard_nlcvrp.py "${BENCHMARK_DIR}" "${DATA}" \
  --min-customers "${MIN_CUSTOMERS}" --max-customers "${MAX_CUSTOMERS}" \
  --limit "${CASE_LIMIT}" --selection "${SELECTION}" \
  --clauses-per-type "${CLAUSES_PER_TYPE}" \
  >"${RUN_ROOT}/logs/prepare.log" 2>&1

torchrun --standalone --nproc_per_node=2 experiments/run_s2e_pipeline.py \
  --data "${DATA}" --model-path "${MODEL_DIR}" --output "${CPP}" \
  --temperature 0 --max-new-tokens 1536 --resume \
  >"${RUN_ROOT}/logs/cpp.log" 2>&1

for ARM in full no_feedback pbit_only random_pbit oracle; do
  OUTPUT="${RUN_ROOT}/${ARM}"
  EXTRA=(--constraint-source cpp --cpp-records "${CPP}/records.jsonl")
  METHOD=llm
  if [[ "${ARM}" == no_feedback ]]; then
    EXTRA+=(--disable-logit-feedback --token-feedback-strength 0)
  elif [[ "${ARM}" == pbit_only ]]; then
    METHOD=knn
  elif [[ "${ARM}" == random_pbit ]]; then
    METHOD=random
  elif [[ "${ARM}" == oracle ]]; then
    EXTRA=(--constraint-source dataset)
  fi
  # shellcheck disable=SC2086
  torchrun --standalone --nproc_per_node=2 experiments/run_cvrp_lns.py \
    --dataset jsonl --data "${DATA}" --output "${OUTPUT}" \
    --methods "${METHOD}" --search-seeds ${SEARCH_SEEDS} --sampler torch \
    --initialization pbit-cold --cold-batch-size 8 \
    --cold-routes-per-customer "${ROUTES_PER_CUSTOMER}" \
    --iterations "${ITERATIONS}" --patience "${PATIENCE}" \
    --destroy-size "${DESTROY_SIZE}" --routes-per-customer "${ROUTES_PER_CUSTOMER}" \
    --sampling-steps "${SAMPLING_STEPS}" --num-chains "${NUM_CHAINS}" \
    --top-samples "${TOP_SAMPLES}" --model-path "${MODEL_DIR}" \
    --llm-temperature 0 --llm-refresh-interval 5 --llm-max-new-tokens 1536 \
    --llm-max-input-tokens 32768 --proposal-bias 5 --logit-feedback-rate 0.8 \
    "${EXTRA[@]}" --resume >"${RUN_ROOT}/logs/${ARM}.log" 2>&1
done

HGS_DIR="${HGS_DIR:-${WORK_ROOT:-/hdd/wl2}/runtime/src/HGS-CVRP}"
HGS_BINARY="${HGS_BINARY:-${HGS_DIR}/build/hgs}"
[[ -x "${HGS_BINARY}" ]] || HGS_BINARY="${HGS_DIR}/build/bin/hgs"
if [[ ! -x "${HGS_BINARY}" ]]; then
  echo "HGS executable not found under ${HGS_DIR}/build" >&2
  exit 2
fi

# The direct LLM receives raw numerical data and natural-language requirements,
# then emits a complete solution. No repair is performed before verification.
# shellcheck disable=SC2086
torchrun --standalone --nproc_per_node=2 experiments/run_cvrp_lns.py \
  --dataset jsonl --data "${DATA}" --output "${RUN_ROOT}/baselines" \
  --methods greedy ortools hgs llm_direct --search-seeds ${SEARCH_SEEDS} \
  --constraint-source dataset --model-path "${MODEL_DIR}" \
  --llm-temperature 0.2 --llm-direct-max-new-tokens "${LLM_DIRECT_MAX_NEW_TOKENS}" \
  --llm-max-input-tokens 32768 --baseline-time-limit "${BASELINE_TIME_LIMIT}" \
  --hgs-binary "${HGS_BINARY}" --resume \
  >"${RUN_ROOT}/logs/baselines.log" 2>&1

# shellcheck disable=SC2086
python experiments/analyze_cvrp_method_comparison.py "${RUN_ROOT}" \
  --seeds ${SEARCH_SEEDS} >"${RUN_ROOT}/logs/comparison.log" 2>&1

echo "Completed: ${RUN_ROOT}/method_comparison.json"
