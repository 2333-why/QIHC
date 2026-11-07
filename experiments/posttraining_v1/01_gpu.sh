#!/usr/bin/env bash
source "$(dirname "$0")/common.sh"
exec "$PY" -m experiments.posttraining_v1.schedule "${1:-formal}"
