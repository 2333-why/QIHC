# Hard NL-CVRP dual-track experiment (implementation status)

This protocol tests *end-to-end* constraint understanding, p-bit construction
from empty routes, LLM candidate selection, p-bit search, and p-bit-to-LLM
token-logit feedback. It does not use `known_feasible_routes` as an incumbent.

## What executes

1. The constraint model generates a CPP from the natural-language clauses.
   CPP identity and customer IDs are checked; A0–A4 validation and compilation
   run before the search instance is created. Search sees `cpp.specs()`, not
   the reference `instance.constraints`. Reference constraints are used only
   for held-out feasibility/exact-match metrics.
2. The compilation plan selects the p-bit backend. Pairwise same/different
   route constraints run as QUBO or categorical p-dit penalties. Capacity runs
   with the existing QUBO penalty or multiplier-feedback hybrid sampler.
   Precedence receives a same-route energy penalty; its ordering is checked
   exactly after decoding. Time-window and distance clauses are currently
   **verifier-only**; the plan records this explicitly. They are never silently
   reported as lowered energy terms. Infeasible samples cannot replace the incumbent.
3. `--initialization pbit-cold` starts from empty routes. The LLM may rank
   candidate route domains for each batch, and p-bit constructs a feasible
   solution. It never loads the published witness routes. The subsequent LNS
   loop uses LLM proposals and p-bit sampling to improve it.
4. p-bit elite/negative sample residuals are fed both into the next prompt and
   into *generation token logits* only when the model is writing the matching
   customer's route ID inside `candidate_routes`. This changes inference-time
   token probabilities, not model weights. SFT/DPO/GRPO remains a separate
   offline post-training phase. `llm_audit_rank*.jsonl` records
   `token_feedback_applied_steps` to verify actual activation.

## Data and leakage boundary

Use [official CVRPLIB X-family instances](https://galgos.inf.puc-rio.br/cvrplib/en/instances)
(`.vrp` files with matching `.sol` files).
`prepare_hard_nlcvrp.py` checks the `.sol` as a feasibility witness, derives
2 same-vehicle, 2 precedence and 2 mutual-exclusion clauses, then exports
only the numerical problem, natural-language clauses, reference constraints,
and public BKS cost. **No witness route is exported.** These language clauses
are controlled additions to public CVRP instances, not native CVRPLIB data.
Report that fact clearly. Split by instance family/name before training;
never train on held-out instances or use their witness solutions as prompts.

For the two-RTX-PRO-6000 server, from a fresh terminal:

```bash
source /hdd/wl2/QIHC/scripts/s2e/activate_qihc.sh
export BENCHMARK_DIR=/hdd/wl2/data/CVRPLIB/X
python /hdd/wl2/QIHC/experiments/download_hard_cvrplib.py "$BENCHMARK_DIR" \
  --min-customers 200 --max-customers 400 --limit 3
export MODEL_DIR=/hdd/wl2/models/Qwen--Qwen2.5-32B-Instruct
export RUN_ROOT=/hdd/wl2/results/qihc_hard_pilot
export CASE_LIMIT=3 MIN_CUSTOMERS=200 MAX_CUSTOMERS=400 SEARCH_SEEDS=0
bash /hdd/wl2/QIHC/scripts/s2e/run_hard_dual_track.sh
```

Inspect `${RUN_ROOT}/data/hard_nlcvrp.manifest.json`; if zero instances were
written, confirm `.vrp` and `.sol` files have matching stems. Review
`${RUN_ROOT}/cpp/summary.json`, `${RUN_ROOT}/hard_comparison.json`, each
arm's `summary.json` and `failures.json`,
then expand with a **new** `RUN_ROOT`, e.g.
`CASE_LIMIT=20 MAX_CUSTOMERS=800 SEARCH_SEEDS='0 1 2' ITERATIONS=30
SAMPLING_STEPS=120 NUM_CHAINS=256`. Reusing a pilot output directory
with `--resume` would silently retain the pilot jobs under changed settings.

Arms: `full` (CPP + cold p-bit + LLM + token feedback), `no_feedback`,
`pbit_only` (CPP + cold p-bit + KNN neighborhood), and `oracle` (reference
constraints available to the solver; an upper-bound ablation, **not** a fair
unassisted system). The script holds p-bit sampling settings fixed and reports
wall-clock time; a separate time-matched run is needed for runtime-fair claims.
Compare paired instances/seeds. Primary outcomes are reference-feasible rate and paired
cost differences; report completion failures too. The public BKS is for the
*unaugmented* CVRP, so gap to it is a common lower-bound comparison, **not**
the optimality gap of the augmented NL-CVRP. Improvement
relative to the p-bit-constructed solution is secondary and must not be
confused with improvement from a published initial solution.

Cold construction may fail on very tight fleets; this is counted as a failed
run, not replaced by a greedy or published initial solution. Publish success
rate and failure reasons, not only the successful-case objective gap.
