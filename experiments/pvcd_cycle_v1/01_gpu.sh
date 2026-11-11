#!/usr/bin/env bash
source "$(dirname "$0")/common.sh"
"$PY" -m experiments.feedback_active_v3.check_env --gpu
mode="${1:-all}"
if [[ "$mode" == all ]]; then
  "$PY" -m experiments.pvcd_cycle_v1.launch pilot
  exec "$PY" -m experiments.pvcd_cycle_v1.launch formal
fi
exec "$PY" -m experiments.pvcd_cycle_v1.launch "$mode"
