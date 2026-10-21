#!/usr/bin/env bash
# Run on a CPU compute allocation. Keep it running while the GPU worker runs.
set -Eeuo pipefail
CODE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)"
BASE="${QIHC_BASE:-/inspire/hdd/global_user/czxs25210022/why}"
ENV_PREFIX="${QIHC_ENV_PREFIX:-$BASE/envs/qihc-cu128}"
RUN="${QIHC_COMPARISON_RUN:-$BASE/results/comparison_v1}"
export PATH="$ENV_PREFIX/bin:$PATH" PYTHONPATH="$CODE" PYTHONNOUSERSITE=1
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
mkdir -p "$BASE/logs"
exec > >(tee -a "$BASE/logs/comparison-cpu-$(date -u +%Y%m%dT%H%M%SZ)-$$.log") 2>&1
cd "$CODE"
if [[ ! -f "$RUN/manifest.json" ]]; then
  "$ENV_PREFIX/bin/python" tools/benchmarks/prepare_comparison.py --base "$BASE" --run "$RUN" --per-dataset "${QIHC_COMPARISON_PER_DATASET:-40}"
fi
echo 'CPU queue worker ready. Launch script 02 in the GPU allocation. Do not close this CPU process.'
exec "$ENV_PREFIX/bin/python" tools/benchmarks/comparison.py cpu --run "$RUN"
