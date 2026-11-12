#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
export QIHC_ENV="${QIHC_ENV:-/inspire/hdd/global_user/czxs25210022/why/envs/qihc-cu128}"
PY="$QIHC_ENV/bin/python"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
export QIHC_PROBE_LANES="${QIHC_PROBE_LANES:-2}" QIHC_SCORE_WORKERS="${QIHC_SCORE_WORKERS:-8}"
[[ -x "$PY" ]] || { echo "Existing shared Python missing: $PY" >&2; exit 1; }
