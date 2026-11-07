#!/usr/bin/env bash
source "$(dirname "$0")/common.sh"
"$PY" -m experiments.posttraining_v1.report "${1:-formal}"
cat "$QIHC_POST_RUN/${1:-formal}/report.md"
