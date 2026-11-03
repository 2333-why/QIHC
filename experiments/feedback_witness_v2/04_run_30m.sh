#!/usr/bin/env bash
set -Eeuo pipefail
export QIHC_WITNESS_CONFIG=config_30m.json
export QIHC_WITNESS_RUN="${QIHC_QUICK_RUN:-${QIHC_BASE:-/inspire/hdd/global_user/czxs25210022/why}/results/E004-1.6_feedback_quick_30m_fix2}"
source "$(dirname -- "${BASH_SOURCE[0]}")/env.sh"
"$PY" -c 'from experiments.feedback_witness_v2 import runtime, protocol, cpu, report; print("Actual runtime imports passed; jsonschema is not required")'
"$PY" -m experiments.feedback_witness_v2.budget
mkdir -p "$QIHC_WITNESS_RUN/logs"
LOG="$QIHC_WITNESS_RUN/logs/launch-$(TZ=Asia/Shanghai date +%Y%m%dT%H%M%S)-$$.log"
exec > >(tee -a "$LOG") 2>&1
echo "E004-1.6 start/resume; 30-minute original deadline persists; run=$QIHC_WITNESS_RUN"
exec "$PY" -u -m experiments.feedback_witness_v2.supervise --run "$QIHC_WITNESS_RUN" --source "$QIHC_PILOT_SOURCE" --model "$QIHC_MODEL" "$@"
