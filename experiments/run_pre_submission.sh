#!/usr/bin/env bash
# Pre-submission evidence batch for NSFC youth student project.
# Run on server: CUDA_VISIBLE_DEVICES=0,1 bash experiments/run_pre_submission.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-python}"
MODEL="${MODEL_NAME:-Qwen/Qwen2.5-7B-Instruct}"
OUT="${OUT_DIR:-experiments/outputs/nsfc_evidence/pre_submission_$(date +%Y%m%d_%H%M%S)}"
SEEDS="${SEEDS:-0 1 2 3 4}"
mkdir -p "$OUT"

echo "=== QIHC pre-submission batch ==="
echo "OUT=$OUT  MODEL=$MODEL  SEEDS=$SEEDS"

run() {
  echo ""
  echo ">>> $*"
  "$@"
}

# 1. Unified ablation (bundled + synthetic), pseudo logits, multi-seed
run "$PYTHON" experiments/nsfc_evidence/run_unified_ablation.py \
  --dataset bundled --budget-steps 400 --n-samples 50 \
  --seeds $SEEDS --output-dir "$OUT/unified_bundled"

run "$PYTHON" experiments/nsfc_evidence/run_unified_ablation.py \
  --dataset synthetic --n-tasks 200 --budget-steps 400 --n-samples 50 \
  --seeds $SEEDS --output-dir "$OUT/unified_synthetic"

# 2. Unified ablation with 7B real logits (bundled n=40)
run "$PYTHON" experiments/nsfc_evidence/run_unified_ablation.py \
  --dataset bundled --logits llm --model-name "$MODEL" \
  --budget-steps 400 --n-samples 50 --seeds 0 1 2 \
  --output-dir "$OUT/unified_bundled_7b"

# 3. Handoff dense σ scan 0.2–1.0, 7B
run "$PYTHON" experiments/run_vci_handoff.py \
  --source bundled --logits llm --model-name "$MODEL" \
  --limit 40 --budget-steps 400 --noise-scales-dense \
  --seeds 0 1 2 --output-dir "$OUT/handoff_7b_dense"

# 4. Pareto with CR modes
run "$PYTHON" experiments/run_vci_pareto.py \
  --budgets 100 200 300 400 600 --include-cr \
  --output-dir "$OUT/pareto_bundled"

run "$PYTHON" experiments/run_vci_pareto.py \
  --budgets 100 200 300 400 600 --include-cr \
  --logits llm --model-name "$MODEL" \
  --output-dir "$OUT/pareto_bundled_7b"

# 5. Max-Cut scaling n=50–500 + TTS
run "$PYTHON" experiments/run_sampler_scaling.py \
  --nodes 12 16 20 24 50 100 200 500 --trials 8 --steps 800 \
  --output-dir "$OUT/scaling_fixed"

run "$PYTHON" experiments/run_sampler_scaling.py \
  --nodes 50 100 200 500 --trials 5 --measure-tts \
  --output-dir "$OUT/scaling_tts"

# 6. P012 cases not yet in local run (if GPU available)
run "$PYTHON" experiments/nsfc_evidence/run_case_g_constrained_bbh.py \
  --limit-per-task 50 --logits llm --model-name "$MODEL" \
  --budget-steps 400 --output-dir "$OUT/case_g"

run "$PYTHON" experiments/nsfc_evidence/run_case_h_model_compare.py \
  --n-tasks 200 --model-7b "$MODEL" \
  --model-14b "${MODEL_14B:-Qwen/Qwen2.5-14B-Instruct}" \
  --budget-steps 400 --output-dir "$OUT/case_h"

# 7. Aggregate same-compute budget table
run "$PYTHON" experiments/nsfc_evidence/run_case_c_compute_budget.py \
  --run-dir "$OUT" --output-dir "$OUT/compute_budget" || true

echo ""
echo "=== Done. Results in $OUT ==="
echo "Pack: tar -czf ${OUT}.tar.gz -C $(dirname "$OUT") $(basename "$OUT")"
