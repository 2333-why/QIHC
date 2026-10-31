#!/usr/bin/env bash
set -Eeuo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/env.sh"
"$PY" -u -m experiments.feedback_study_v1.report --pairs "$@"
cat "$QIHC_E004_RUN/status.json"
echo "Report: $QIHC_E004_RUN/report.md"
