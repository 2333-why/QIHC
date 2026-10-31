#!/usr/bin/env bash
set -Eeuo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/env.sh"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
# Scheduler rank variables do not prove this command is nested in torchrun.
# Validate actual CUDA visibility and acquire the single-launch lock in Python.
mkdir -p "$QIHC_E004_RUN/logs"
LOG="$QIHC_E004_RUN/logs/launch-$(date -u +%Y%m%dT%H%M%SZ)-$$.log"
exec > >(tee -a "$LOG") 2>&1
echo "E004 launch/resume; log=$LOG"
exec "$PY" -u -m experiments.feedback_study_v1.runtime "$@"
