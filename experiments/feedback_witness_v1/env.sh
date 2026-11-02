#!/usr/bin/env bash
set -Eeuo pipefail
CODE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)"
BASE="${QIHC_BASE:-/inspire/hdd/global_user/czxs25210022/why}"
PY="${QIHC_ENV_PREFIX:-$BASE/envs/qihc-cu128}/bin/python"
FORMAT="${QIHC_FORMAT_DEPS:-$BASE/deps/modeling-v5-format-0113}"
SOLVERS="${QIHC_SOLVER_DEPS:-$BASE/deps/e002-solvers-highs1151-scip630}"
export QIHC_WITNESS_RUN="${QIHC_WITNESS_RUN:-$BASE/results/E004-1.4_feedback_witness_8h}"
export QIHC_PILOT_SOURCE="${QIHC_PILOT_SOURCE:-$BASE/results/E004-1.2_feedback_pilot}"
export QIHC_MODEL="${QIHC_MODEL:-$BASE/models/Qwen3-32B}"
export PYTHONPATH="$CODE:$SOLVERS:$FORMAT" PYTHONNOUSERSITE=1 MPLBACKEND=Agg
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export PATH="$(dirname -- "$PY"):$PATH"
[[ -x "$PY" ]] || { echo "Missing project interpreter: $PY"; exit 2; }
cd "$CODE"
