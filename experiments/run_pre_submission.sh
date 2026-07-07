#!/usr/bin/env bash
# Pre-submission evidence batch — CR paper-aligned (arXiv:2407.00071).
# Baseline: zeroshot = LLM T=0 direct answer (NOT logits-greedy).
#
# Run on server:
#   CUDA_VISIBLE_DEVICES=0,1 MODEL_NAME=Qwen/Qwen2.5-7B-Instruct \
#     bash experiments/run_pre_submission.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-python}"
MODEL="${MODEL_NAME:-Qwen/Qwen2.5-7B-Instruct}"
OUT="${OUT_DIR:-experiments/outputs/nsfc_evidence/pre_submission_$(date +%Y%m%d_%H%M%S)}"
SEEDS="${SEEDS:-0 1 2 3 4}"
mkdir -p "$OUT"

echo "=== QIHC pre-submission (CR paper-aligned) ==="
echo "OUT=$OUT  MODEL=$MODEL  SEEDS=$SEEDS"

run() {
  echo ""
  echo ">>> $*"
  "$@"
}

# --- Track A: CR paper accuracy + gain_over_zeroshot ---

# 1a. Smoke (mock LLM, CPU) — pipeline sanity only
run "$PYTHON" experiments/nsfc_evidence/run_unified_ablation.py \
  --dataset bundled --budget-steps 400 --n-samples 50 \
  --seeds 0 --output-dir "$OUT/smoke_bundled"

# 1b. Bundled constrained — multi-seed mock (feasibility narrative)
run "$PYTHON" experiments/nsfc_evidence/run_unified_ablation.py \
  --dataset bundled --budget-steps 400 --n-samples 50 \
  --seeds $SEEDS --output-dir "$OUT/unified_bundled"

run "$PYTHON" experiments/nsfc_evidence/run_unified_ablation.py \
  --dataset synthetic --n-tasks 200 --budget-steps 400 --n-samples 50 \
  --seeds $SEEDS --output-dir "$OUT/unified_synthetic"

# 1c. Real LLM — paper-comparable accuracy (GPU)
run "$PYTHON" experiments/nsfc_evidence/run_unified_ablation.py \
  --dataset bundled --use-llm --model-name "$MODEL" \
  --budget-steps 400 --n-samples 50 --seeds 0 1 2 \
  --output-dir "$OUT/unified_bundled_7b"

run "$PYTHON" experiments/nsfc_evidence/run_cr_bbh.py \
  --source hf --use-llm --model-name "$MODEL" \
  --n-samples 50 --limit-per-task 25 --budget-steps 400 \
  --output-dir "$OUT/cr_hf_bbh"

# --- Track B: VCI advantage under noise (vs CR zeroshot/linear) ---

run "$PYTHON" experiments/run_vci_handoff.py \
  --source bundled --logits llm --model-name "$MODEL" \
  --limit 40 --budget-steps 400 --noise-scales-dense \
  --seeds 0 1 2 --include-cr-baselines \
  --output-dir "$OUT/handoff_7b_dense"

# --- Track C: Pareto (CR linear/quadratic vs VCI at matched p-bit budget) ---

run "$PYTHON" experiments/run_vci_pareto.py \
  --budgets 100 200 300 400 600 --include-cr \
  --output-dir "$OUT/pareto_bundled"

run "$PYTHON" experiments/run_vci_pareto.py \
  --budgets 100 200 300 400 600 --include-cr \
  --use-llm --model-name "$MODEL" \
  --output-dir "$OUT/pareto_bundled_7b"

# --- Track D: Cases B/F/G ---

run "$PYTHON" experiments/nsfc_evidence/run_case_b_dual_synthetic.py \
  --n-tasks 200 --n-samples 50 --budget-steps 400 \
  --output-dir "$OUT/case_b"

run "$PYTHON" experiments/nsfc_evidence/run_case_f_vci_ablation.py \
  --n-tasks 200 --n-samples 50 --budget-steps 400 \
  --output-dir "$OUT/case_f"

run "$PYTHON" experiments/nsfc_evidence/run_case_g_constrained_bbh.py \
  --limit-per-task 50 --use-llm --model-name "$MODEL" \
  --budget-steps 400 --output-dir "$OUT/case_g"

run "$PYTHON" experiments/nsfc_evidence/run_case_h_model_compare.py \
  --n-tasks 200 --model-7b "$MODEL" \
  --model-14b "${MODEL_14B:-Qwen/Qwen2.5-14B-Instruct}" \
  --budget-steps 400 --output-dir "$OUT/case_h"

# --- Track E: Max-Cut scaling (p-bit backend, independent) ---

run "$PYTHON" experiments/run_sampler_scaling.py \
  --nodes 12 16 20 24 50 100 200 500 --trials 8 --steps 800 \
  --output-dir "$OUT/scaling_fixed"

run "$PYTHON" experiments/run_sampler_scaling.py \
  --nodes 50 100 200 500 --trials 5 --measure-tts \
  --output-dir "$OUT/scaling_tts"

# --- Aggregate ---

run "$PYTHON" experiments/nsfc_evidence/run_case_c_compute_budget.py \
  --run-dir "$OUT" --output-dir "$OUT/compute_budget" || true

echo ""
echo "=== Done. Results in $OUT ==="
echo "Pack: tar -czf ${OUT}.tar.gz -C $(dirname "$OUT") $(basename "$OUT")"
