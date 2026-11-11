#!/usr/bin/env bash
source "$(dirname "$0")/common.sh"
"$PY" -m experiments.feedback_active_v3.check_env --gpu
exec "$PY" -m experiments.pvcd_cycle_v1.prepare
