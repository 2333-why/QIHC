#!/usr/bin/env bash
set -Eeuo pipefail
export QIHC_E002_RUN="${QIHC_E002_RUN:-${QIHC_BASE:-/inspire/hdd/global_user/czxs25210022/why}/results/E002-1.1_solver_standalone}"
source "$(dirname -- "${BASH_SOURCE[0]}")/env.sh"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 PIP_NO_INDEX=1
exec "$PY" -u -m experiments.solver_comparison_v1.standalone "$@"
