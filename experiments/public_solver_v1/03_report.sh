#!/usr/bin/env bash
set -Eeuo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/env.sh"
"$PY" -m experiments.public_solver_v1.solve report "$@"
printf 'Report: %s/report.md\n' "$QIHC_E003_RUN"
