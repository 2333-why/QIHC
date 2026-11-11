#!/usr/bin/env bash
source "$(dirname "$0")/common.sh"
"$PY" -m experiments.boundary_feedback_v1.run pilot
"$PY" -m experiments.boundary_feedback_v1.gate
exec "$PY" -m experiments.boundary_feedback_v1.run formal
