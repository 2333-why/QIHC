#!/usr/bin/env bash
set -Eeuo pipefail
CODE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)"
BASE="${QIHC_BASE:-/inspire/hdd/global_user/czxs25210022/why}"
ENV_PREFIX="${QIHC_ENV_PREFIX:-$BASE/envs/qihc-cu128}"
RUN="${QIHC_CO_RUN:-$BASE/results/co_comparison_v1}"
export PATH="$ENV_PREFIX/bin:$PATH" PYTHONPATH="$CODE" PYTHONNOUSERSITE=1
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
mkdir -p "$BASE/logs"
exec > >(tee -a "$BASE/logs/co-cpu-$(date -u +%Y%m%dT%H%M%SZ)-$$.log") 2>&1
trap 'echo "CPU failed at line $LINENO; inspect this log"' ERR
cd "$CODE"
"$ENV_PREFIX/bin/python" tools/benchmarks/prepare_co.py --base "$BASE" --run "$RUN" --per-tier "${QIHC_NLCO_PER_TIER:-20}"
echo 'CPU ready. Keep this CPU compute process running and launch 02_co_gpu.sh on GPU.'
exec "$ENV_PREFIX/bin/python" tools/benchmarks/comparison.py cpu --run "$RUN"
