#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/activate_qihc.sh"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
RUN_ID="${RUN_ID:-s2e_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${WORK_ROOT}/results/${RUN_ID}}"
DATA="${OUTPUT_ROOT}/data/nl_rich_vrp.jsonl"
mkdir -p "${OUTPUT_ROOT}/data" "${OUTPUT_ROOT}/logs"
HGS_DIR="${BUNDLE_ROOT}/src/HGS-CVRP"
CVRP_DIR="${HGS_DIR}/Instances/CVRP"
HGS_BINARY="${HGS_DIR}/build/hgs"; [[ -x "${HGS_BINARY}" ]] || HGS_BINARY="${HGS_DIR}/build/bin/hgs"
python experiments/prepare_nlcvrp_jsonl.py "${CVRP_DIR}" "${DATA}" --limit "${INSTANCE_LIMIT:-100}"
torchrun --standalone --nproc_per_node=4 experiments/run_s2e_pipeline.py --data "${DATA}" --model-path "${MODEL_DIR}" --output "${OUTPUT_ROOT}/cpp_pretrain" --limit "${INSTANCE_LIMIT:-100}" 2>&1 | tee "${OUTPUT_ROOT}/logs/cpp_pretrain.log"
torchrun --standalone --nproc_per_node=4 experiments/run_cvrp_lns.py --dataset jsonl --data "${DATA}" --output "${OUTPUT_ROOT}/solver_full" --limit "${INSTANCE_LIMIT:-100}" --search-seeds 0 1 2 3 4 --methods random knn llm --sampler torch --iterations "${LNS_ITERATIONS:-100}" --patience 30 --destroy-size 12 --routes-per-customer 4 --sampling-steps "${SAMPLING_STEPS:-1000}" --num-chains "${NUM_CHAINS:-2048}" --top-samples 32 --model-path "${MODEL_DIR}" --proposal-bias "${PROPOSAL_BIAS:-1.0}" --logit-feedback-rate "${LOGIT_FEEDBACK_RATE:-0.8}" --feedback-elite-fraction "${ELITE_FRACTION:-0.25}" 2>&1 | tee "${OUTPUT_ROOT}/logs/solver.log"
if [[ "${RUN_FEEDBACK_ABLATION:-1}" == "1" ]]; then
  torchrun --standalone --nproc_per_node=4 experiments/run_cvrp_lns.py --dataset jsonl --data "${DATA}" --output "${OUTPUT_ROOT}/solver_no_feedback" --limit "${INSTANCE_LIMIT:-100}" --search-seeds 0 1 2 3 4 --methods llm --sampler torch --iterations "${LNS_ITERATIONS:-100}" --patience 30 --destroy-size 12 --routes-per-customer 4 --sampling-steps "${SAMPLING_STEPS:-1000}" --num-chains "${NUM_CHAINS:-2048}" --top-samples 32 --model-path "${MODEL_DIR}" --disable-logit-feedback 2>&1 | tee "${OUTPUT_ROOT}/logs/solver_no_feedback.log"
fi
torchrun --standalone --nproc_per_node=4 experiments/run_s2e_hybrid_ablation.py --data "${DATA}" --output "${OUTPUT_ROOT}/hybrid_pdit_mfc" --limit "${INSTANCE_LIMIT:-100}" --steps "${HYBRID_STEPS:-200}" --chains "${HYBRID_CHAINS:-512}" 2>&1 | tee "${OUTPUT_ROOT}/logs/hybrid.log"
python experiments/prepare_joint_training_data.py --data "${DATA}" --constraint-records "${OUTPUT_ROOT}/cpp_pretrain/records.jsonl" --solver-results "${OUTPUT_ROOT}/solver_full/results.json" --output "${OUTPUT_ROOT}/training_data"
if [[ "${RUN_TRAINING:-1}" == "1" ]]; then
  torchrun --standalone --nproc_per_node=4 experiments/train_s2e_lora.py --stage sft --model-path "${MODEL_DIR}" --data "${OUTPUT_ROOT}/training_data/sft.jsonl" --output "${OUTPUT_ROOT}/checkpoints/sft" --max-steps "${SFT_STEPS:-500}" 2>&1 | tee "${OUTPUT_ROOT}/logs/sft.log"
  LAST_ADAPTER="${OUTPUT_ROOT}/checkpoints/sft"
  if [[ -s "${OUTPUT_ROOT}/training_data/dpo.jsonl" ]]; then
    torchrun --standalone --nproc_per_node=4 experiments/train_s2e_lora.py --stage dpo --model-path "${MODEL_DIR}" --adapter-path "${LAST_ADAPTER}" --data "${OUTPUT_ROOT}/training_data/dpo.jsonl" --output "${OUTPUT_ROOT}/checkpoints/dpo" --max-steps "${DPO_STEPS:-300}" 2>&1 | tee "${OUTPUT_ROOT}/logs/dpo.log"
    LAST_ADAPTER="${OUTPUT_ROOT}/checkpoints/dpo"
  fi
  torchrun --standalone --nproc_per_node=4 experiments/train_s2e_lora.py --stage grpo --model-path "${MODEL_DIR}" --adapter-path "${LAST_ADAPTER}" --data "${OUTPUT_ROOT}/training_data/grpo.jsonl" --output "${OUTPUT_ROOT}/checkpoints/grpo" --max-steps "${GRPO_STEPS:-300}" 2>&1 | tee "${OUTPUT_ROOT}/logs/grpo.log"
fi
python experiments/analyze_cvrp_results.py "${OUTPUT_ROOT}/solver_full" --reference knn || true
tar -czf "${OUTPUT_ROOT}.tar.gz" -C "$(dirname "${OUTPUT_ROOT}")" "$(basename "${OUTPUT_ROOT}")"
sha256sum "${OUTPUT_ROOT}.tar.gz" > "${OUTPUT_ROOT}.tar.gz.sha256"
echo "FORMAL RUN COMPLETE: ${OUTPUT_ROOT}"
