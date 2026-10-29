#!/usr/bin/env bash
set -Eeuo pipefail
CODE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)"
BASE="${QIHC_BASE:-/inspire/hdd/global_user/czxs25210022/why}"
PY="${QIHC_ENV_PREFIX:-$BASE/envs/qihc-cu128}/bin/python"
FORMAT="${QIHC_FORMAT_DEPS:-$BASE/deps/modeling-v5-format-0113}"
SOLVERS="${QIHC_SOLVER_DEPS:-$BASE/deps/e002-solvers-highs1151-scip630}"
export QIHC_E003_RUN="${QIHC_E003_RUN:-$BASE/results/E003-1.1_public_solvers}"
export PYTHONPATH="$CODE:$SOLVERS:$FORMAT" PYTHONNOUSERSITE=1 MPLBACKEND=Agg
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
export PATH="$(dirname -- "$PY"):$PATH"
cd "$CODE"
