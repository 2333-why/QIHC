#!/usr/bin/env bash
# Sequential, restartable 8-GPU formal comparison: hard first, then medium.
set -euo pipefail

REPO_DIR="${REPO_DIR:-/hdd/wl2/QIHC}"
source "${REPO_DIR}/scripts/s2e/activate_qihc.sh"
cd "${REPO_DIR}"

WORK_ROOT="${WORK_ROOT:-${REPO_DIR}/deployment}"
PIPELINE_ROOT="${PIPELINE_ROOT:-${WORK_ROOT}/results/formal_two_stage_8gpu_v1}"
MODEL_DIR="${MODEL_DIR:-${WORK_ROOT}/models/Qwen--Qwen3.5-35B-A3B}"
BENCHMARK_DIR="${BENCHMARK_DIR:-${WORK_ROOT}/data/CVRPLIB/X}"
HGS_DIR="${HGS_DIR:-${WORK_ROOT}/runtime/src/HGS-CVRP}"
HGS_BINARY="${HGS_BINARY:-${HGS_DIR}/build/hgs}"
[[ -x "${HGS_BINARY}" ]] || HGS_BINARY="${HGS_DIR}/build/bin/hgs"

export REPO_DIR WORK_ROOT MODEL_DIR BENCHMARK_DIR HGS_DIR HGS_BINARY
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

mkdir -p "${PIPELINE_ROOT}"
test -s "${MODEL_DIR}/config.json"
test -x "${HGS_BINARY}"

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

validate_stage() {
  local run_root="$1"
  python - "${run_root}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
cpp = json.loads((root / "cpp" / "summary.json").read_text(encoding="utf-8"))
required = ("success_rate", "a4_pass_rate", "semantic_match_rate")
bad = {key: cpp.get(key) for key in required if cpp.get(key) != 1.0}
if bad:
    raise SystemExit(f"invalid CPP summary: {bad}")
comparison = json.loads((root / "method_comparison.json").read_text(encoding="utf-8"))
methods = comparison.get("methods", {})
expected = {
    "qihc_full", "qihc_no_feedback", "pbit_only",
    "random_neighborhood_pbit", "oracle_constraints_qihc",
    "greedy_oracle", "ortools_gls_oracle",
    "hgs_constraint_unaware", "llm_direct_unrepaired",
}
missing = sorted(expected - set(methods))
if missing:
    raise SystemExit(f"missing comparison methods: {missing}")
print(json.dumps({
    "run_root": str(root),
    "benchmark": comparison.get("benchmark"),
    "cpp": {key: cpp[key] for key in required},
    "methods": len(methods),
}, ensure_ascii=False))
PY
}

run_stage() {
  local name="$1"
  local run_root="$2"
  local min_customers="$3"
  local max_customers="$4"
  local clauses_per_type="$5"
  local num_chains="$6"
  local sampling_steps="$7"

  mkdir -p "${run_root}"
  if [[ -s "${run_root}/method_comparison.json" ]]; then
    log "${name}: existing final result found; validating before reuse."
    validate_stage "${run_root}"
    log "${name}: already complete."
    return
  fi

  # The hard stage may already have been launched with the documented
  # standalone command.  Attach to it instead of starting a duplicate 8-GPU
  # job, then reuse or resume its output after that process exits.
  if [[ -s "${run_root}/launcher.pid" ]]; then
    local external_pid
    external_pid="$(tr -dc '0-9' < "${run_root}/launcher.pid")"
    if [[ -n "${external_pid}" ]] && kill -0 "${external_pid}" 2>/dev/null; then
      log "${name}: an existing launcher (${external_pid}) is active; waiting for it."
      while kill -0 "${external_pid}" 2>/dev/null; do
        sleep 60
      done
      if [[ -s "${run_root}/method_comparison.json" ]]; then
        validate_stage "${run_root}"
        log "${name}: attached run completed and validated."
        return
      fi
      log "${name}: attached run ended without a final comparison; resuming it."
    fi
  fi

  log "${name}: starting ${min_customers}-${max_customers} customer benchmark."
  env \
    RUN_ROOT="${run_root}" \
    MIN_CUSTOMERS="${min_customers}" \
    MAX_CUSTOMERS="${max_customers}" \
    CASE_LIMIT="${CASE_LIMIT:-12}" \
    SELECTION="${SELECTION:-spread}" \
    CLAUSES_PER_TYPE="${clauses_per_type}" \
    SEARCH_SEEDS="${SEARCH_SEEDS:-0 1 2}" \
    ITERATIONS="${ITERATIONS:-40}" \
    PATIENCE="${PATIENCE:-20}" \
    DESTROY_SIZE="${DESTROY_SIZE:-20}" \
    ROUTES_PER_CUSTOMER="${ROUTES_PER_CUSTOMER:-8}" \
    NUM_CHAINS="${num_chains}" \
    SAMPLING_STEPS="${sampling_steps}" \
    TOP_SAMPLES="${TOP_SAMPLES:-128}" \
    BASELINE_TIME_LIMIT="${BASELINE_TIME_LIMIT:-300}" \
    LLM_DIRECT_MAX_NEW_TOKENS="${LLM_DIRECT_MAX_NEW_TOKENS:-8192}" \
    bash "${REPO_DIR}/scripts/s2e/run_hard_cvrp_comparison.sh"

  validate_stage "${run_root}"
  log "${name}: completed and validated."
}

HARD_RUN_ROOT="${HARD_RUN_ROOT:-${PIPELINE_ROOT}/hard_x400_1000}"
MEDIUM_RUN_ROOT="${MEDIUM_RUN_ROOT:-${PIPELINE_ROOT}/medium_x200_399}"

log "Two-stage formal pipeline started."
run_stage "hard" "${HARD_RUN_ROOT}" 400 1000 4 2048 160
run_stage "medium" "${MEDIUM_RUN_ROOT}" 200 399 3 1024 128

python - "${PIPELINE_ROOT}" "${HARD_RUN_ROOT}" "${MEDIUM_RUN_ROOT}" <<'PY'
import json
import sys
from pathlib import Path

pipeline, hard, medium = map(Path, sys.argv[1:])
summary = {
    "status": "complete",
    "stages": {
        "hard": json.loads((hard / "method_comparison.json").read_text(encoding="utf-8")),
        "medium": json.loads((medium / "method_comparison.json").read_text(encoding="utf-8")),
    },
}
(pipeline / "two_stage_summary.json").write_text(
    json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
)
PY

log "Two-stage formal pipeline completed: ${PIPELINE_ROOT}/two_stage_summary.json"
