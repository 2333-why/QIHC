#!/usr/bin/env bash
# Run from /hdd/wl2/QIHC after sourcing scripts/s2e/activate_qihc.sh.
set -euo pipefail

REPO_DIR="${REPO_DIR:-/hdd/wl2/QIHC}"
source "${REPO_DIR}/scripts/s2e/activate_qihc.sh"
MODEL_DIR="${MODEL_DIR:-/hdd/wl2/models/Qwen--Qwen3-Coder-30B-A3B-Instruct}"
BENCHMARK_DIR="${BENCHMARK_DIR:?Set BENCHMARK_DIR to the local CVRPLIB .vrp/.sol directory}"
RUN_ROOT="${RUN_ROOT:-/hdd/wl2/results/qihc_hard_dual_track}"
CASE_LIMIT="${CASE_LIMIT:-3}"
MIN_CUSTOMERS="${MIN_CUSTOMERS:-200}"
MAX_CUSTOMERS="${MAX_CUSTOMERS:-400}"
SELECTION="${SELECTION:-smallest}"
SEARCH_SEEDS="${SEARCH_SEEDS:-0}"
ITERATIONS="${ITERATIONS:-10}"
SAMPLING_STEPS="${SAMPLING_STEPS:-40}"
NUM_CHAINS="${NUM_CHAINS:-64}"
cd "${REPO_DIR}"
mkdir -p "${RUN_ROOT}/data" "${RUN_ROOT}/logs"
test -s "${MODEL_DIR}/config.json"
test -d "${BENCHMARK_DIR}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
DATA="${RUN_ROOT}/data/hard_nlcvrp.jsonl"
CPP="${RUN_ROOT}/cpp"

python experiments/prepare_hard_nlcvrp.py "${BENCHMARK_DIR}" "${DATA}" \
  --min-customers "${MIN_CUSTOMERS}" --max-customers "${MAX_CUSTOMERS}" \
  --limit "${CASE_LIMIT}" --selection "${SELECTION}" --clauses-per-type 2 \
  >"${RUN_ROOT}/logs/prepare.log" 2>&1

torchrun --standalone --nproc_per_node=2 experiments/run_s2e_pipeline.py \
  --data "${DATA}" --model-path "${MODEL_DIR}" --output "${CPP}" \
  --temperature 0 --max-new-tokens 1024 --resume \
  >"${RUN_ROOT}/logs/cpp.log" 2>&1

for ARM in full no_feedback pbit_only oracle; do
  OUTPUT="${RUN_ROOT}/${ARM}"
  EXTRA=(--constraint-source cpp --cpp-records "${CPP}/records.jsonl")
  METHOD=llm
  if [[ "${ARM}" == no_feedback ]]; then
    EXTRA+=(--disable-logit-feedback --token-feedback-strength 0)
  elif [[ "${ARM}" == pbit_only ]]; then
    METHOD=knn
  elif [[ "${ARM}" == oracle ]]; then
    EXTRA=(--constraint-source dataset)
  fi
  # shellcheck disable=SC2086
  torchrun --standalone --nproc_per_node=2 experiments/run_cvrp_lns.py \
    --dataset jsonl --data "${DATA}" --output "${OUTPUT}" \
    --methods "${METHOD}" --search-seeds ${SEARCH_SEEDS} \
    --sampler torch --initialization pbit-cold --cold-batch-size 6 \
    --cold-routes-per-customer 6 --iterations "${ITERATIONS}" --patience 15 \
    --destroy-size 12 --routes-per-customer 6 --sampling-steps "${SAMPLING_STEPS}" \
    --num-chains "${NUM_CHAINS}" --top-samples 32 --model-path "${MODEL_DIR}" \
    --llm-temperature 0 --llm-refresh-interval 5 --llm-max-new-tokens 1024 \
    --proposal-bias 5 --logit-feedback-rate 0.8 \
    "${EXTRA[@]}" --resume >"${RUN_ROOT}/logs/${ARM}.log" 2>&1
done
python experiments/analyze_hard_dual_track.py "${RUN_ROOT}" --seeds ${SEARCH_SEEDS} \
  >"${RUN_ROOT}/logs/comparison.log" 2>&1
echo "Completed. Compare reference_feasible_rate, mean_optimality_gap and total_elapsed_s in ${RUN_ROOT}/*/summary.json"
