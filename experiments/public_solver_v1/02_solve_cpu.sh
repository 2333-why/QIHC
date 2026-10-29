#!/usr/bin/env bash
set -Eeuo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/env.sh"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
exec "$PY" -u -m experiments.public_solver_v1.solve run --workers "${QIHC_CPU_WORKERS:-24}" "$@"
