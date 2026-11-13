#!/usr/bin/env bash
source "$(dirname "$0")/common.sh"
exec "$PY" -m experiments.pbit_joint_v1.run report
