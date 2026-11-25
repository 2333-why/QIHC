#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
export QIHC_BASE="${QIHC_BASE:-/inspire/hdd/global_user/czxs25210022/why}"
export QIHC_ENV="${QIHC_ENV:-$QIHC_BASE/envs/qihc-cu128}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
exec "$QIHC_ENV/bin/python" -m experiments.pbit_stack_v1.run "$@"
