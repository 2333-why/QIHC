#!/usr/bin/env bash
source "$(dirname "$0")/common.sh"
exec "$PY" -m experiments.pvcd_verified_v1.run report "${1:-screen}"
