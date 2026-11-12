#!/usr/bin/env bash
source "$(dirname "$0")/common.sh"
"$PY" -m experiments.feedback_active_v3.check_env --gpu
exec "$PY" -m experiments.pvcd_verified_v1.run prepare
