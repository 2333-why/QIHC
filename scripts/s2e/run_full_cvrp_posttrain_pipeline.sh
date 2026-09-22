#!/usr/bin/env bash
# One command: two formal CVRP comparisons, joint data, 8-GPU ZeRO-3 post-training, package.
set -euo pipefail

REPO_DIR="${REPO_DIR:-/hdd/wl2/QIHC}"
source "${REPO_DIR}/scripts/s2e/activate_qihc.sh"
cd "${REPO_DIR}"

WORK_ROOT="${WORK_ROOT:-${REPO_DIR}/deployment}"
PIPELINE_ROOT="${PIPELINE_ROOT:-${WORK_ROOT}/results/formal_cvrp_posttrain_8gpu_v1}"
MODEL_DIR="${MODEL_DIR:-${WORK_ROOT}/models/Qwen--Qwen3.5-35B-A3B}"
POSTTRAIN_ROOT="${POSTTRAIN_ROOT:-${PIPELINE_ROOT}/posttrain}"
DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-${REPO_DIR}/configs/deepspeed-zero3-bf16.json}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
AUTO_PACKAGE="${AUTO_PACKAGE:-1}"

export REPO_DIR WORK_ROOT PIPELINE_ROOT MODEL_DIR POSTTRAIN_ROOT
export NPROC_PER_NODE
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

mkdir -p "${PIPELINE_ROOT}" "${POSTTRAIN_ROOT}/logs"

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

log "Checking the existing environment, model and eight-GPU runtime."
python - "${MODEL_DIR}" "${DEEPSPEED_CONFIG}" <<'PY'
import importlib
import json
import sys
from pathlib import Path

model = Path(sys.argv[1])
ds_config = Path(sys.argv[2])
for package in ("torch", "transformers", "accelerate", "datasets", "peft", "trl", "deepspeed"):
    importlib.import_module(package)
if not (model / "config.json").is_file():
    raise SystemExit(f"missing model config: {model / 'config.json'}")
if not ((model / "tokenizer.json").is_file() or (model / "tokenizer_config.json").is_file()):
    raise SystemExit(f"missing tokenizer under {model}")
weight_files = list(model.glob("*.safetensors")) + list(model.glob("*.bin"))
if not weight_files:
    raise SystemExit(f"missing model weights under {model}")
json.loads(ds_config.read_text(encoding="utf-8"))
print(f"environment/model check passed: {model}; weight_files={len(weight_files)}")
PY

torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" \
  experiments/check_pro6000_environment.py --expected-gpus "${NPROC_PER_NODE}" \
  --min-compute-capability 8 --min-memory-gib "${MIN_GPU_MEMORY_GIB:-75}" \
  >"${POSTTRAIN_ROOT}/logs/gpu_preflight.log" 2>&1

log "Running or resuming the hard and medium formal comparisons."
HARD_RUN_ROOT="${HARD_RUN_ROOT:-${PIPELINE_ROOT}/hard_x400_1000}"
MEDIUM_RUN_ROOT="${MEDIUM_RUN_ROOT:-${PIPELINE_ROOT}/medium_x200_399}"
export HARD_RUN_ROOT MEDIUM_RUN_ROOT
bash "${REPO_DIR}/scripts/s2e/run_two_stage_cvrp_pipeline.sh"

prepare_stage_data() {
  local stage_root="$1"
  local output="$2"
  mkdir -p "${output}"
  python experiments/prepare_joint_training_data.py \
    --data "${stage_root}/data/hard_nlcvrp.jsonl" \
    --constraint-records "${stage_root}/cpp/records.jsonl" \
    --solver-results "${stage_root}/full/results.json" \
    --llm-audits "${stage_root}/full" \
    --output "${output}"
  python - "${output}/summary.json" <<'PY'
import json
import sys
from pathlib import Path

summary = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if not summary.get("used_exact_llm_audits"):
    raise SystemExit("exact seed-aware LLM audits are required for p-bit feedback training")
if summary.get("constraint_sft", 0) <= 0 or summary.get("proposal_sft", 0) <= 0:
    raise SystemExit(f"incomplete joint training data: {summary}")
print(json.dumps(summary, ensure_ascii=False))
PY
}

log "Building seed-aligned constraint and p-bit-feedback training records."
prepare_stage_data "${HARD_RUN_ROOT}" "${POSTTRAIN_ROOT}/data_hard"
prepare_stage_data "${MEDIUM_RUN_ROOT}" "${POSTTRAIN_ROOT}/data_medium"
python experiments/merge_joint_training_data.py \
  --inputs "${POSTTRAIN_ROOT}/data_hard" "${POSTTRAIN_ROOT}/data_medium" \
  --output "${POSTTRAIN_ROOT}/data"

for split in sft dpo grpo; do
  rows="$(awk 'END {print NR}' "${POSTTRAIN_ROOT}/data/${split}.jsonl")"
  if [[ "${rows}" -le 0 ]]; then
    echo "Post-training ${split} dataset is empty" >&2
    exit 2
  fi
  log "Training data ${split}: ${rows} rows."
done

adapter_complete() {
  local output="$1"
  [[ -s "${output}/adapter_config.json" ]] && \
    { [[ -s "${output}/adapter_model.safetensors" ]] || [[ -s "${output}/adapter_model.bin" ]]; } && \
    [[ -s "${output}/qihc_training_manifest.json" ]]
}

train_stage() {
  local stage="$1"
  local data="$2"
  local output="$3"
  local steps="$4"
  local learning_rate="$5"
  local adapter_path="${6:-}"

  if adapter_complete "${output}"; then
    log "${stage}: completed adapter found; skipping."
    return
  fi
  mkdir -p "${output}"
  local -a extra=()
  if [[ -n "${adapter_path}" ]]; then
    if ! adapter_complete "${adapter_path}"; then
      echo "Previous adapter is incomplete: ${adapter_path}" >&2
      exit 2
    fi
    extra+=(--adapter-path "${adapter_path}")
  fi
  log "${stage}: starting ZeRO-3 LoRA training for ${steps} steps."
  torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" \
    experiments/train_s2e_lora.py \
    --stage "${stage}" --model-path "${MODEL_DIR}" --data "${data}" \
    --output "${output}" --max-steps "${steps}" \
    --learning-rate "${learning_rate}" \
    --max-length "${TRAIN_MAX_LENGTH:-4096}" \
    --lora-rank "${LORA_RANK:-16}" \
    --gradient-accumulation-steps "${GRADIENT_ACCUMULATION_STEPS:-8}" \
    --save-steps "${TRAIN_SAVE_STEPS:-50}" \
    --num-generations "${GRPO_NUM_GENERATIONS:-4}" \
    --deepspeed-config "${DEEPSPEED_CONFIG}" --resume \
    "${extra[@]}" >"${POSTTRAIN_ROOT}/logs/${stage}.log" 2>&1
  if ! adapter_complete "${output}"; then
    echo "${stage} ended without a complete adapter" >&2
    exit 2
  fi
  log "${stage}: completed."
}

SFT_ADAPTER="${POSTTRAIN_ROOT}/adapters/sft"
DPO_ADAPTER="${POSTTRAIN_ROOT}/adapters/dpo"
GRPO_ADAPTER="${POSTTRAIN_ROOT}/adapters/grpo"

train_stage sft "${POSTTRAIN_ROOT}/data/sft.jsonl" "${SFT_ADAPTER}" \
  "${SFT_MAX_STEPS:-200}" "${SFT_LEARNING_RATE:-2e-5}"
train_stage dpo "${POSTTRAIN_ROOT}/data/dpo.jsonl" "${DPO_ADAPTER}" \
  "${DPO_MAX_STEPS:-100}" "${DPO_LEARNING_RATE:-5e-6}" "${SFT_ADAPTER}"
train_stage grpo "${POSTTRAIN_ROOT}/data/grpo.jsonl" "${GRPO_ADAPTER}" \
  "${GRPO_MAX_STEPS:-100}" "${GRPO_LEARNING_RATE:-2e-6}" "${DPO_ADAPTER}"

python - "${PIPELINE_ROOT}" "${POSTTRAIN_ROOT}" <<'PY'
import json
import subprocess
import sys
from pathlib import Path

pipeline, posttrain = map(Path, sys.argv[1:])
summary = {
    "status": "complete",
    "git_commit": subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True
    ).strip(),
    "experiments": json.loads(
        (pipeline / "two_stage_summary.json").read_text(encoding="utf-8")
    ),
    "training_data": json.loads(
        (posttrain / "data" / "summary.json").read_text(encoding="utf-8")
    ),
    "adapters": {
        stage: json.loads(
            (posttrain / "adapters" / stage / "qihc_training_manifest.json").read_text(encoding="utf-8")
        )
        for stage in ("sft", "dpo", "grpo")
    },
    "final_adapter": str(posttrain / "adapters" / "grpo"),
}
(pipeline / "full_pipeline_summary.json").write_text(
    json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
)
PY

if [[ "${AUTO_PACKAGE}" == "1" ]]; then
  PACKAGE_PATH="${PACKAGE_PATH:-${PIPELINE_ROOT}.tar.gz}"
  log "Packaging results without large resumable checkpoint directories."
  tar --exclude='*/checkpoint-*' \
    -C "$(dirname "${PIPELINE_ROOT}")" -czf "${PACKAGE_PATH}" \
    "$(basename "${PIPELINE_ROOT}")"
  log "Package ready: ${PACKAGE_PATH}"
fi

log "Full experiment and post-training pipeline completed."
