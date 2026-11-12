#!/usr/bin/env bash
set -euo pipefail
export QIHC_CONTROL_RUN="${QIHC_CONTROL_RUN:-/inspire/hdd/global_user/czxs25210022/why/results/E005-1.12_controller_full780}"
export QIHC_CONTROL_CONFIG="$(cd "$(dirname "$0")" && pwd)/config_full.json"
exec bash "$(dirname "$0")/02_report.sh" full
