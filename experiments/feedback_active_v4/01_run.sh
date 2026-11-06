#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export QIHC_ACTIVE_RUN="${QIHC_ACTIVE_RUN:-/inspire/hdd/global_user/czxs25210022/why/results/E004_2.2_active_feedback}"
bash "$ROOT/experiments/feedback_active_v3/01_run.sh" --config "$ROOT/experiments/feedback_active_v4/config.json" "$@"
