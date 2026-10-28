#!/usr/bin/env bash
CODE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)"
BASE="${QIHC_BASE:-/inspire/hdd/global_user/czxs25210022/why}"
ENV_PREFIX="${QIHC_ENV_PREFIX:-$BASE/envs/qihc-cu128}"
FORMAT="${QIHC_FORMAT_DEPS:-$BASE/deps/modeling-v5-format-0113}"
SOLVERS="${QIHC_SOLVER_DEPS:-$BASE/deps/e002-solvers-highs1151-scip630}"
export QIHC_PROMPT_V3_RUN="${QIHC_PROMPT_V3_RUN:-$BASE/results/E001-1.3_prompt_fixed}"
export QIHC_E002_RUN="${QIHC_E002_RUN:-$BASE/results/E002-1.1_solver_comparison}"
export PATH="$ENV_PREFIX/bin:$PATH" PYTHONPATH="$CODE:$SOLVERS:$FORMAT" PYTHONNOUSERSITE=1 MPLBACKEND=Agg
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
PY="$ENV_PREFIX/bin/python"
cd "$CODE"
