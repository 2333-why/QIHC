#!/usr/bin/env bash
source "$(dirname "$0")/common.sh"
exec "$PY" -m experiments.semantic_tests_v1.run "${1:-formal}"
