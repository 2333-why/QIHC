# CR Paper-Aligned Experiment Protocol (arXiv:2407.00071)

## Baseline definition

| Mode | CR paper definition | NOT this |
|------|---------------------|----------|
| **zeroshot** | LLM direct answer, T=0, 1 call | logits top-k greedy |
| **linear** | N samples T=1 → majority vote | frequency logits + greedy |
| **quadratic** | sample → QUBO reason select → enhanced prompt → LLM T=0 | direct QUBO on answer mask only |
| **vci-1** | CR-encoded logits + one-way p-bit + IF constraints | greedy |
| **vci-2** | CR-encoded logits + q↔s VCI loop + IF constraints | greedy |

## Metrics

| Metric | HF BBH (top_k=1) | Bundled constrained (top_k≥2) |
|--------|------------------|-------------------------------|
| **accuracy** | exact match | gold-hit for CR single-answer modes |
| **exact_match_rate** | same as accuracy | full set match (VCI primary) |
| **feasible_rate** | usually 100% | **primary QIHC metric** |
| **gain_over_zeroshot** | CR paper Table 3 style | accuracy gain vs CR zeroshot |

## Tracks

1. **Paper accuracy** (`--use-llm --source hf`): reproduce CR zeroshot < linear < quadratic
2. **Constrained IF** (bundled/synthetic): VCI-2 feasible_rate >> CR paper modes
3. **Handoff** (`--include-cr-baselines`): VCI-2 vs CR under semantic noise
4. **Pareto**: p-bit budget vs feasible/accuracy (zeroshot = 0 p-bit reference)

## Commands

```bash
# Smoke (CPU, mock LLM)
python experiments/nsfc_evidence/run_cr_bbh.py --source bundled --limit 15

# Paper-comparable (GPU)
python experiments/nsfc_evidence/run_cr_bbh.py \
  --source hf --use-llm --model-name Qwen/Qwen2.5-7B-Instruct \
  --n-samples 50 --limit-per-task 25

# Full pre-submission batch
CUDA_VISIBLE_DEVICES=0,1 MODEL_NAME=Qwen/Qwen2.5-7B-Instruct \
  bash experiments/run_pre_submission.sh
```

## Code entry

- `qihc/orchestrator/cr_pipeline.py` — paper pipeline
- `experiments/nsfc_evidence/run_cr_bbh.py` — main benchmark
- `experiments/nsfc_evidence/run_unified_ablation.py` — unified table
