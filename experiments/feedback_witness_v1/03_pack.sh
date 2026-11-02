#!/usr/bin/env bash
set -Eeuo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/env.sh"
exec "$PY" -u -m experiments.feedback_witness_v1.report --run "$QIHC_WITNESS_RUN" --pack "$@"
