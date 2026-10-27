#!/usr/bin/env bash
set -Eeuo pipefail
CODE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)"
BASE="${QIHC_BASE:-/inspire/hdd/global_user/czxs25210022/why}"
ENV_PREFIX="${QIHC_ENV_PREFIX:-$BASE/envs/qihc-cu128}"
DEPS="${QIHC_FORMAT_DEPS:-$BASE/deps/modeling-v5-format-0113}"
export PATH="$ENV_PREFIX/bin:$PATH" PYTHONPATH="$CODE:$DEPS" PYTHONNOUSERSITE=1 MPLBACKEND=Agg
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 PIP_NO_INDEX=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
cd "$CODE"
exec "$ENV_PREFIX/bin/python" -u -m experiments.prompt_efficiency_v2.run launch "$@"
