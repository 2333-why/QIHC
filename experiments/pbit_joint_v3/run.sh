#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
export QIHC_ENV="${QIHC_ENV:-/inspire/hdd/global_user/czxs25210022/why/envs/qihc-cu128}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
if [[ ${1:-} == prepare ]]; then
  "$QIHC_ENV/bin/python" -m experiments.feedback_active_v3.check_env --gpu
fi
exec "$QIHC_ENV/bin/python" -m experiments.pbit_joint_v3.run "$@"
