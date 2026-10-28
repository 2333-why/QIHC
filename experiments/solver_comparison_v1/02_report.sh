#!/usr/bin/env bash
set -Eeuo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/env.sh"
"$PY" -m experiments.prompt_efficiency_v3.run report
"$PY" -m experiments.solver_comparison_v1.run report
printf '%s\n' "$QIHC_PROMPT_V3_RUN/formal/report.md" "$QIHC_E002_RUN/report.md"
