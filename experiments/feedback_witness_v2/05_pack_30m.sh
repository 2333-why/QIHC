#!/usr/bin/env bash
set -Eeuo pipefail
export QIHC_WITNESS_CONFIG=config_30m.json
export QIHC_WITNESS_RUN="${QIHC_QUICK_RUN:-${QIHC_BASE:-/inspire/hdd/global_user/czxs25210022/why}/results/E004-1.6_feedback_quick_30m_fix2}"
source "$(dirname -- "${BASH_SOURCE[0]}")/env.sh"
exec "$PY" -u -m experiments.feedback_witness_v2.report --run "$QIHC_WITNESS_RUN" --pack "$@"
