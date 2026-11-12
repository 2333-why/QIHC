#!/usr/bin/env bash
source "$(dirname "$0")/common.sh"
"$PY" -m experiments.feedback_active_v3.check_env --gpu
"$PY" -m experiments.solver_control_v1.run prepare
exec "$PY" -m experiments.solver_control_v1.run labels "${1:-pilot}"
