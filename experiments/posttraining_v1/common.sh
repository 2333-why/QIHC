#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
export QIHC_ENV="${QIHC_ENV:-/inspire/hdd/global_user/czxs25210022/why/envs/qihc-cu128}"
export QIHC_POST_RUN="${QIHC_POST_RUN:-/inspire/hdd/global_user/czxs25210022/why/results/E005-1.7_eight}"
export QIHC_MODEL="${QIHC_MODEL:-/inspire/hdd/global_user/czxs25210022/why/models/Qwen3-32B}"
PY="$QIHC_ENV/bin/python"
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
