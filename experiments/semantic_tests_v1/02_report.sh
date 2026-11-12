#!/usr/bin/env bash
source "$(dirname "$0")/common.sh"
exec "$PY" -m experiments.semantic_tests_v1.report "${1:-formal}"
