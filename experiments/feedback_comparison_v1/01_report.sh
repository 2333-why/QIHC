#!/usr/bin/env bash
set -Eeuo pipefail
CODE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)"
BASE="${QIHC_BASE:-/inspire/hdd/global_user/czxs25210022/why}"
PY="${QIHC_ENV_PREFIX:-$BASE/envs/qihc-cu128}/bin/python"
export PYTHONPATH="$CODE" PYTHONNOUSERSITE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
cd "$CODE"
exec "$PY" -u -m experiments.feedback_comparison_v1.report \
  --e003 "${QIHC_E003_RUN:-$BASE/results/E003-1.1_public_solvers}" \
  --e004 "${QIHC_E004_RUN:-$BASE/results/E004-1.1_feedback}" \
  --output "${QIHC_COMPARISON_OUT:-$BASE/results/E004-1.1_vs_E003}" "$@"
