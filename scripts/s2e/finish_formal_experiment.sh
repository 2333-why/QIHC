#!/usr/bin/env bash
# Resume the train collection and then run the formal post-training/evaluation chain.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/activate_qihc.sh"

export FORMAL_ROOT="${FORMAL_ROOT:-${WORK_ROOT}/results/qihc_formal_v1}"
export SPLIT_DIR="${SPLIT_DIR:-${FORMAL_ROOT}/data/splits}"
export TRAIN_DATASET="${TRAIN_DATASET:-${SPLIT_DIR}/train.jsonl}"
export COLLECTION_ROOT="${COLLECTION_ROOT:-${FORMAL_ROOT}/train_collection}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

SOLVER_DIR="${COLLECTION_ROOT}/solver"
CPP_DIR="${COLLECTION_ROOT}/cpp_train"
TRAINING_DIR="${COLLECTION_ROOT}/training_data"
CHECKPOINT_DIR="${COLLECTION_ROOT}/checkpoints"
EVAL_DIR="${COLLECTION_ROOT}/evaluation"
LOG_DIR="${COLLECTION_ROOT}/logs"
mkdir -p "${LOG_DIR}" "${SOLVER_DIR}" "${TRAINING_DIR}" "${CHECKPOINT_DIR}" "${EVAL_DIR}"

exec 9>"${LOG_DIR}/finish_pipeline.lock"
if ! flock -n 9; then
  echo "Another finish_formal_experiment.sh is already running." >&2
  exit 1
fi

for required in "${TRAIN_DATASET}" "${MODEL_DIR}/config.json"; do
  [[ -s "${required}" ]] || { echo "Missing required file: ${required}" >&2; exit 1; }
done
[[ "$(awk -F, '{print NF}' <<<"${CUDA_VISIBLE_DEVICES}")" -eq 2 ]] || {
  echo "This workflow requires exactly two visible GPUs." >&2; exit 1;
}

log() { printf '[%s] %s\n' "$(date '+%F %T')" "$*"; }
wait_for_gpus() {
  local used
  while true; do
    # A Qwen-32B rank needs most of its 96-GB GPU; do not compete with another job.
    used="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)"
    if awk 'BEGIN {ok=1} {if ($1 >= 8192) ok=0; n++} END {exit !(ok && n >= 2)}' <<<"${used}"; then
      return
    fi
    log "Waiting for both GPUs to be free (current memory used: $(tr '\n' ' ' <<<"${used}") MiB)."
    sleep 60
  done
}
run_cpp_stage() {
  local data="$1" output="$2" log_file="$3" attempt=1 rc attempt_log
  local max_attempts="${MAX_CPP_RETRIES:-5}"
  shift 3
  [[ "${max_attempts}" =~ ^[1-9][0-9]*$ ]] || {
    echo "MAX_CPP_RETRIES must be a positive integer" >&2
    return 2
  }
  while (( attempt <= max_attempts )); do
    wait_for_gpus
    log "Constraint compilation attempt ${attempt}/${max_attempts}: ${output}"
    attempt_log="${log_file}.attempt${attempt}.log"
    printf '\n===== attempt %s/%s at %s =====\n' "${attempt}" "${max_attempts}" "$(date '+%F %T')" >>"${log_file}"
    if torchrun --standalone --nproc_per_node=2 experiments/run_s2e_pipeline.py \
      --data "${data}" --model-path "${MODEL_DIR}" \
      --output "${output}" --max-new-tokens 768 --resume "$@" \
      >"${attempt_log}" 2>&1; then
      sed -n '1,$p' "${attempt_log}" >>"${log_file}"
      return 0
    else
      rc=$?
    fi
    sed -n '1,$p' "${attempt_log}" >>"${log_file}"
    if ! grep -Eq 'torch.OutOfMemoryError|CUDA out of memory' "${attempt_log}"; then
      log "Constraint compilation failed for a reason other than CUDA OOM; inspect ${log_file}."
      return "${rc}"
    fi
    log "CUDA OOM while compiling constraints; saved checkpoints will be reused after the GPUs are free."
    attempt=$((attempt + 1))
    sleep 30
  done
  log "Constraint compilation exceeded ${max_attempts} attempts; inspect ${log_file}."
  return 1
}
check_train() {
  python experiments/check_cvrp_collection.py \
    --data "${TRAIN_DATASET}" --output "${SOLVER_DIR}" \
    --search-seeds 0 1 --methods llm "$@"
}
solver_pids() {
  pgrep -af '[r]un_cvrp_lns.py' 2>/dev/null |
    grep -F -- "--output ${SOLVER_DIR}" |
    awk '{print $1}' || true
}

log "Checking the train collection before launching any new GPU work."
while [[ -n "$(solver_pids)" ]]; do
  log "Existing solver is active; waiting rather than starting a duplicate."
  check_train
  sleep 60
done
if ! check_train --require-complete; then
  log "Resuming only the missing train jobs."
  wait_for_gpus
  bash "${SCRIPT_DIR}/resume_train_collection.sh" >"${LOG_DIR}/resume_solver.log" 2>&1
fi
check_train --require-complete >"${SOLVER_DIR}/coverage.json"

log "Rebuilding de-duplicated solver aggregates."
python - "${SOLVER_DIR}" <<'PY'
import sys
from pathlib import Path
from experiments.run_cvrp_lns import aggregate
aggregate(Path(sys.argv[1]), world_size=2)
PY

if [[ ! -s "${CPP_DIR}/records.jsonl" ]]; then
  log "Running constraint compilation on train instances."
  run_cpp_stage "${TRAIN_DATASET}" "${CPP_DIR}" "${LOG_DIR}/cpp_train.log"
fi

log "Building joint train-only SFT/DPO/GRPO data."
python experiments/prepare_joint_training_data.py \
  --data "${TRAIN_DATASET}" \
  --constraint-records "${CPP_DIR}/records.jsonl" \
  --solver-results "${SOLVER_DIR}/results.json" \
  --llm-audits "${SOLVER_DIR}" \
  --output "${TRAINING_DIR}" \
  >"${LOG_DIR}/prepare_training_data.log" 2>&1
[[ -s "${TRAINING_DIR}/sft.jsonl" ]] || { echo "SFT data is empty" >&2; exit 1; }

LAST_ADAPTER="${CHECKPOINT_DIR}/sft"
if [[ ! -s "${LAST_ADAPTER}/adapter_config.json" ]]; then
  log "Starting two-GPU SFT."
  wait_for_gpus
  torchrun --standalone --nproc_per_node=2 experiments/train_s2e_lora.py \
    --stage sft --model-path "${MODEL_DIR}" --data "${TRAINING_DIR}/sft.jsonl" \
    --output "${LAST_ADAPTER}" --max-steps "${SFT_STEPS:-500}" \
    >"${LOG_DIR}/sft.log" 2>&1
fi
[[ -s "${LAST_ADAPTER}/adapter_config.json" ]] || { echo "SFT adapter missing" >&2; exit 1; }

if [[ -s "${TRAINING_DIR}/dpo.jsonl" ]]; then
  DPO_ADAPTER="${CHECKPOINT_DIR}/dpo"
  if [[ ! -s "${DPO_ADAPTER}/adapter_config.json" ]]; then
    log "Starting two-GPU DPO."
    wait_for_gpus
    torchrun --standalone --nproc_per_node=2 experiments/train_s2e_lora.py \
      --stage dpo --model-path "${MODEL_DIR}" --adapter-path "${LAST_ADAPTER}" \
      --data "${TRAINING_DIR}/dpo.jsonl" --output "${DPO_ADAPTER}" \
      --max-steps "${DPO_STEPS:-300}" >"${LOG_DIR}/dpo.log" 2>&1
  fi
  [[ -s "${DPO_ADAPTER}/adapter_config.json" ]] || { echo "DPO adapter missing" >&2; exit 1; }
  LAST_ADAPTER="${DPO_ADAPTER}"
else
  log "No DPO pairs; keeping SFT adapter."
fi

if [[ "${RUN_GRPO:-1}" == "1" && -s "${TRAINING_DIR}/grpo.jsonl" ]]; then
  GRPO_ADAPTER="${CHECKPOINT_DIR}/grpo"
  if [[ ! -s "${GRPO_ADAPTER}/adapter_config.json" ]]; then
    log "Starting two-GPU GRPO."
    wait_for_gpus
    torchrun --standalone --nproc_per_node=2 experiments/train_s2e_lora.py \
      --stage grpo --model-path "${MODEL_DIR}" --adapter-path "${LAST_ADAPTER}" \
      --data "${TRAINING_DIR}/grpo.jsonl" --output "${GRPO_ADAPTER}" \
      --max-steps "${GRPO_STEPS:-300}" >"${LOG_DIR}/grpo.log" 2>&1
  fi
  [[ -s "${GRPO_ADAPTER}/adapter_config.json" ]] || { echo "GRPO adapter missing" >&2; exit 1; }
  LAST_ADAPTER="${GRPO_ADAPTER}"
fi

if [[ "${RUN_EVAL:-1}" == "1" ]]; then
  for split in validation test; do
    DATASET="${SPLIT_DIR}/${split}.jsonl"
    [[ -s "${DATASET}" ]] || { echo "Missing ${split} split: ${DATASET}" >&2; exit 1; }
    for variant in base tuned; do
      OUTPUT="${EVAL_DIR}/${split}_${variant}"
      CPP_OUTPUT="${EVAL_DIR}/${split}_${variant}_cpp"
      ADAPTER_ARGS=()
      if [[ "${variant}" == tuned ]]; then ADAPTER_ARGS=(--adapter-path "${LAST_ADAPTER}"); fi
      if [[ ! -s "${CPP_OUTPUT}/summary.json" ]]; then
        log "Evaluating constraint compilation for ${split}/${variant}."
        run_cpp_stage "${DATASET}" "${CPP_OUTPUT}" \
          "${LOG_DIR}/${split}_${variant}_cpp.log" "${ADAPTER_ARGS[@]}"
      fi
      if ! python experiments/check_cvrp_collection.py --data "${DATASET}" \
        --output "${OUTPUT}" --search-seeds 0 1 --methods llm \
        --require-complete >"${LOG_DIR}/${split}_${variant}_coverage.log"; then
        log "Evaluating ${split}/${variant} with the same solver settings."
        wait_for_gpus
        torchrun --standalone --nproc_per_node=2 experiments/run_cvrp_lns.py \
          --dataset jsonl --data "${DATASET}" --output "${OUTPUT}" \
          --search-seeds 0 1 --methods llm --sampler torch \
          --iterations 20 --patience 20 --destroy-size 12 \
          --routes-per-customer 4 --sampling-steps 100 --num-chains 256 \
          --top-samples 32 --model-path "${MODEL_DIR}" \
          --llm-max-new-tokens 768 --llm-refresh-interval 5 \
          --llm-temperature 0 --proposal-bias 10 --logit-feedback-rate 0.8 \
          --feedback-elite-fraction 0.25 "${ADAPTER_ARGS[@]}" --resume \
          >"${LOG_DIR}/${split}_${variant}.log" 2>&1
      fi
      python experiments/check_cvrp_collection.py --data "${DATASET}" \
        --output "${OUTPUT}" --search-seeds 0 1 --methods llm \
        --require-complete >"${OUTPUT}/coverage.json"
      python - "${OUTPUT}" <<'PY'
import sys
from pathlib import Path
from experiments.run_cvrp_lns import aggregate
aggregate(Path(sys.argv[1]), world_size=2)
PY
    done
  done
fi

log "FORMAL PIPELINE COMPLETE. Final adapter: ${LAST_ADAPTER}"
