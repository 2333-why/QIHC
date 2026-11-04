#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ENV="${QIHC_ENV:-${CONDA_PREFIX:-${VIRTUAL_ENV:-/inspire/hdd/global_user/czxs25210022/why/.venv-qihc}}}"
export QIHC_ENV="$ENV"
cd "$ROOT"
"$ENV/bin/python" -m experiments.feedback_active_v3.report \
  --run "${QIHC_ACTIVE_RUN:-/inspire/hdd/global_user/czxs25210022/why/results/E004_2.1_active_feedback}" --statistics
