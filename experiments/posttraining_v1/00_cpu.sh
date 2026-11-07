#!/usr/bin/env bash
source "$(dirname "$0")/common.sh"
"$PY" -m experiments.feedback_active_v3.check_env --gpu
"$PY" -m experiments.posttraining_v1.prepare
