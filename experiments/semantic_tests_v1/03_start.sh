#!/usr/bin/env bash
source "$(dirname "$0")/common.sh"
"$PY" -m experiments.semantic_tests_v1.run pilot
"$PY" -m experiments.semantic_tests_v1.gate
exec "$PY" -m experiments.semantic_tests_v1.run formal
