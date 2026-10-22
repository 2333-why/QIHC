#!/usr/bin/env bash
set -Eeuo pipefail
CODE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)"
BASE="${QIHC_BASE:-/inspire/hdd/global_user/czxs25210022/why}"
export QIHC_COMPARISON_RUN="${QIHC_CO_RUN:-$BASE/results/co_comparison_v1}"
export QIHC_PARALLEL_INFERENCE=1
exec bash "$CODE/tools/benchmarks/02_comparison_gpu.sh"
