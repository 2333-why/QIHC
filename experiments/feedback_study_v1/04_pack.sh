#!/usr/bin/env bash
set -Eeuo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/env.sh"
"$PY" -u -m experiments.feedback_study_v1.report --pairs
OUT="$BASE/E004-1.1_analysis_$(TZ=Asia/Shanghai date +%Y%m%dT%H%M%S).tar.gz"
"$PY" -u -m experiments.feedback_study_v1.pack --run "$QIHC_E004_RUN" --output "$OUT"
echo "$OUT"
