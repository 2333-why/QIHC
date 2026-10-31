#!/usr/bin/env bash
set -Eeuo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/env.sh"
exec "$PY" -u -m experiments.feedback_study_v1.runtime --cpu-only "$@"
