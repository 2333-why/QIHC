#!/usr/bin/env bash
source "$(dirname "$0")/common.sh"
exec "$PY" -m experiments.pvcd_cycle_v1.report "${1:-formal}"
