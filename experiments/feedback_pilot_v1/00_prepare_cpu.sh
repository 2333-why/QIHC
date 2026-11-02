#!/usr/bin/env bash
set -Eeuo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/env.sh"
# Reuse the already prepared environment; no package upgrades or model downloads.
exec "$PY" -u -m experiments.feedback_pilot_v1.prepare "$@"
