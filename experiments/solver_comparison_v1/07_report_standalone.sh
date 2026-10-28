#!/usr/bin/env bash
set -Eeuo pipefail
export QIHC_E002_RUN="${QIHC_E002_RUN:-${QIHC_BASE:-/inspire/hdd/global_user/czxs25210022/why}/results/E002-1.1_solver_standalone}"
source "$(dirname -- "${BASH_SOURCE[0]}")/env.sh"
"$PY" -m experiments.solver_comparison_v1.run report "$@"
printf '%s\n' "$QIHC_E002_RUN/report.md"
