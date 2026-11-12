#!/usr/bin/env bash
source "$(dirname "$0")/common.sh"
exec "$PY" -m experiments.solver_control_v1.run gpu "${1:-pilot}"
