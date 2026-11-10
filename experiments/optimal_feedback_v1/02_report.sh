#!/usr/bin/env bash
source "$(dirname "$0")/common.sh"
exec "$PY" -m experiments.optimal_feedback_v1.report "${1:-formal}"
