#!/usr/bin/env bash
# One scheduler worker: 4 H100 + 80 CPUs. Existing environment and weights only.
set -Eeuo pipefail
CODE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd -P)"
BASE="${QIHC_BASE:-/inspire/hdd/global_user/czxs25210022/why}"
ENV_PREFIX="${QIHC_ENV_PREFIX:-$BASE/envs/qihc-cu128}"
export PATH="$ENV_PREFIX/bin:$PATH" PYTHONPATH="$CODE" PYTHONNOUSERSITE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 PIP_NO_INDEX=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
cd "$CODE"
exec "$ENV_PREFIX/bin/python" -u tools/runtime/distribution_job/run.py launch "$@"
