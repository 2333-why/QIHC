#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/activate_qihc.sh"

export FORMAL_ROOT="${FORMAL_ROOT:-${WORK_ROOT}/results/qihc_formal_v1}"
export SPLIT_DIR="${SPLIT_DIR:-${FORMAL_ROOT}/data/splits}"
export TRAIN_DATASET="${TRAIN_DATASET:-${SPLIT_DIR}/train.jsonl}"
export COLLECTION_ROOT="${COLLECTION_ROOT:-${FORMAL_ROOT}/train_collection}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

if [[ ! -s "${TRAIN_DATASET}" ]]; then
  echo "Training split is missing: ${TRAIN_DATASET}" >&2
  exit 1
fi
if [[ ! -s "${MODEL_DIR}/config.json" ]]; then
  echo "Model is missing: ${MODEL_DIR}" >&2
  exit 1
fi

mkdir -p "${COLLECTION_ROOT}/solver" "${COLLECTION_ROOT}/logs"

exec torchrun \
  --standalone \
  --nproc_per_node=2 \
  experiments/run_cvrp_lns.py \
  --dataset jsonl \
  --data "${TRAIN_DATASET}" \
  --output "${COLLECTION_ROOT}/solver" \
  --search-seeds 0 1 \
  --methods llm \
  --sampler torch \
  --iterations 20 \
  --patience 20 \
  --destroy-size 12 \
  --routes-per-customer 4 \
  --sampling-steps 100 \
  --num-chains 256 \
  --top-samples 32 \
  --model-path "${MODEL_DIR}" \
  --llm-max-new-tokens 768 \
  --llm-refresh-interval 5 \
  --llm-temperature 0 \
  --proposal-bias 10 \
  --logit-feedback-rate 0.8 \
  --feedback-elite-fraction 0.25 \
  --resume
