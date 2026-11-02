#!/usr/bin/env bash
set -Eeuo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/env.sh"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
mkdir -p "$QIHC_PILOT_RUN/logs"
LOG="$QIHC_PILOT_RUN/logs/launch-$(date -u +%Y%m%dT%H%M%SZ)-$$.log"
exec > >(tee -a "$LOG") 2>&1
echo "E004-1.2 pilot launch/resume; log=$LOG"
exec "$PY" -u -m experiments.feedback_pilot_v1.runtime "$@"
