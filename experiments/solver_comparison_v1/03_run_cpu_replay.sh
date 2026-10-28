#!/usr/bin/env bash
set -Eeuo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/env.sh"
exec "$PY" -u -m experiments.solver_comparison_v1.run run "$@"
