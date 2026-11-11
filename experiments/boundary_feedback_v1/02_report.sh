#!/usr/bin/env bash
source "$(dirname "$0")/common.sh"
exec "$PY" -m experiments.boundary_feedback_v1.report "${1:-formal}"
