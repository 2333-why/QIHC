#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ENV="${QIHC_ENV:-/inspire/hdd/global_user/czxs25210022/why/envs/qihc-cu128}"
export QIHC_ENV="$ENV"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1 VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1
if [[ ! -x "$ENV/bin/python" ]]; then
  echo "Shared environment missing: $ENV/bin/python" >&2
  exit 1
fi
"$ENV/bin/python" "$ROOT/experiments/feedback_active_v3/check_env.py" --gpu
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
exec "$ENV/bin/python" -u -m experiments.feedback_active_v3.launch \
  --model "${QIHC_MODEL:-/inspire/hdd/global_user/czxs25210022/why/models/Qwen3-32B}" \
  --run "${QIHC_ACTIVE_RUN:-/inspire/hdd/global_user/czxs25210022/why/results/E004_2.1_active_feedback}" "$@"
