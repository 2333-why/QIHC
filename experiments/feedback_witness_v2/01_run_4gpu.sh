#!/usr/bin/env bash
set -Eeuo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/env.sh"
"$PY" -c 'from experiments.feedback_witness_v2 import runtime, protocol, cpu, report; print("Actual runtime imports passed")'
mkdir -p "$QIHC_WITNESS_RUN/logs"
LOG="$QIHC_WITNESS_RUN/logs/launch-$(TZ=Asia/Shanghai date +%Y%m%dT%H%M%S)-$$.log"
exec > >(tee -a "$LOG") 2>&1
echo "E004-1.5 start/resume; original deadline persists; run=$QIHC_WITNESS_RUN"
exec "$PY" -u -m experiments.feedback_witness_v2.supervise --run "$QIHC_WITNESS_RUN" --source "$QIHC_PILOT_SOURCE" --model "$QIHC_MODEL" "$@"
