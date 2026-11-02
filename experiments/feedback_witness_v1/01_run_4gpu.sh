#!/usr/bin/env bash
set -Eeuo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/env.sh"
mkdir -p "$QIHC_WITNESS_RUN/logs"
LOG="$QIHC_WITNESS_RUN/logs/launch-$(TZ=Asia/Shanghai date +%Y%m%dT%H%M%S)-$$.log"
exec > >(tee -a "$LOG") 2>&1
echo "E004-1.4 start/resume; original deadline persists; run=$QIHC_WITNESS_RUN"
exec "$PY" -u -m experiments.feedback_witness_v1.supervise --run "$QIHC_WITNESS_RUN" --source "$QIHC_PILOT_SOURCE" --model "$QIHC_MODEL" "$@"
