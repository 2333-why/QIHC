#!/usr/bin/env bash
set -Eeuo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/env.sh"
"$PY" -m experiments.feedback_witness_v1.budget
"$PY" -m compileall -q experiments/feedback_witness_v1
echo "Capacity and syntax checks passed. Full solver/environment preflight is inside the timed run."
