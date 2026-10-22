#!/usr/bin/env bash
# Read-only sidecar: may run while both legacy and new experiment workers run.
set -Eeuo pipefail
CODE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd -P)"
BASE="${QIHC_BASE:-/inspire/hdd/global_user/czxs25210022/why}"
ENV_PREFIX="${QIHC_ENV_PREFIX:-$BASE/envs/qihc-cu128}"
RUN="${QIHC_DISTRIBUTION_RUN:-$BASE/results/distribution_feedback_v1}"
OUTPUT="${QIHC_DISTRIBUTION_REPORT:-${RUN}_reports}"
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES="" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
cd "$CODE"
exec "$ENV_PREFIX/bin/python" tools/analysis/distribution_feedback/report.py --run "$RUN" --output "$OUTPUT" \
  --cpu-workers "${QIHC_DISTRIBUTION_CPU_WORKERS:-8}" --replicas "${QIHC_DISTRIBUTION_REPLICAS:-2}" \
  --window-seconds "${QIHC_DISTRIBUTION_ETA_WINDOW:-3600}" --watch-seconds "${QIHC_DISTRIBUTION_REPORT_INTERVAL:-0}"
