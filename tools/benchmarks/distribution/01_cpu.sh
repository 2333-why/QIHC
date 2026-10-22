#!/usr/bin/env bash
# New experiment only. Reads the existing audited cases/model, never changes them.
set -Eeuo pipefail
CODE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd -P)"
BASE="${QIHC_BASE:-/inspire/hdd/global_user/czxs25210022/why}"
ENV_PREFIX="${QIHC_ENV_PREFIX:-$BASE/envs/qihc-cu128}"
RUN="${QIHC_DISTRIBUTION_RUN:-$BASE/results/distribution_feedback_v1}"
SOURCE_RUN="${QIHC_DISTRIBUTION_SOURCE:-$BASE/results/co_comparison_v1}"
export PATH="$ENV_PREFIX/bin:$PATH" PYTHONPATH="$CODE" PYTHONNOUSERSITE=1
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 PIP_NO_INDEX=1
mkdir -p "$BASE/logs"
exec > >(tee -a "$BASE/logs/distribution-cpu-$(date -u +%Y%m%dT%H%M%SZ)-$$.log") 2>&1
trap 'echo "New experiment CPU failed at line $LINENO; old experiment is untouched"' ERR
cd "$CODE"
"$ENV_PREFIX/bin/python" tools/benchmarks/distribution/smoke.py
"$ENV_PREFIX/bin/python" tools/benchmarks/distribution/experiment.py prepare --from-run "$SOURCE_RUN" --run "$RUN" \
  --budget "${QIHC_DISTRIBUTION_BUDGET:-8192}" --chains "${QIHC_DISTRIBUTION_CHAINS:-16}" \
  --seeds "${QIHC_DISTRIBUTION_SEEDS:-3}" --solver-seconds "${QIHC_DISTRIBUTION_SOLVER_SECONDS:-5}"
echo 'New CPU queue ready. Keep this CPU process running; use distribution/02_gpu.sh in a SEPARATE GPU allocation.'
exec "$ENV_PREFIX/bin/python" tools/benchmarks/distribution/experiment.py cpu --run "$RUN" --workers "${QIHC_DISTRIBUTION_CPU_WORKERS:-8}"
