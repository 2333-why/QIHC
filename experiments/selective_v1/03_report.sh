#!/usr/bin/env bash
source "$(dirname "$0")/common.sh"
exec "$PY" -m experiments.selective_v1.report "${1:?specify feedback or train}" "${2:-formal}"
