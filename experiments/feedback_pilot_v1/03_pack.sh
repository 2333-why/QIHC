#!/usr/bin/env bash
set -Eeuo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/env.sh"
"$PY" -u -m experiments.feedback_pilot_v1.report
OUT="$BASE/E004-1.2_analysis_$(TZ=Asia/Shanghai date +%Y%m%dT%H%M%S).tar.gz"
"$PY" -u -m experiments.feedback_study_v1.pack --run "$QIHC_PILOT_RUN" --output "$OUT"
echo "$OUT"
