#!/usr/bin/env bash
set -Eeuo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/env.sh"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
# Platform single-worker rank variables are allowed; do not nest torchrun.
if [[ "${WORLD_SIZE:-1}" -gt 1 || "${RANK:-0}" -gt 0 ]]; then
  echo 'Launch this script once in one worker with four GPUs, not once per GPU.'; exit 2
fi
mkdir -p "$QIHC_E004_RUN/logs"
LOG="$QIHC_E004_RUN/logs/launch-$(date -u +%Y%m%dT%H%M%SZ)-$$.log"
exec > >(tee -a "$LOG") 2>&1
echo "E004 launch/resume; log=$LOG"
exec "$PY" -u -m experiments.feedback_study_v1.runtime "$@"
