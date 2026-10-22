#!/usr/bin/env bash
set -Eeuo pipefail
CODE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd -P)"
BASE="${QIHC_BASE:-/inspire/hdd/global_user/czxs25210022/why}"
ENV_PREFIX="${QIHC_ENV_PREFIX:-$BASE/envs/qihc-cu128}"
RUN="${QIHC_DISTRIBUTION_RUN:-$BASE/results/distribution_feedback_v1}"
cd "$CODE"
exec "$ENV_PREFIX/bin/python" tools/benchmarks/distribution/experiment.py status --run "$RUN"
