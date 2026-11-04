#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ENV="${QIHC_ACTIVE_ENV:-$ROOT/.venv-e004-active}"
if [[ ! -x "$ENV/bin/python" ]]; then
  bash "$ROOT/experiments/feedback_active_v3/00_setup.sh"
fi
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
exec "$ENV/bin/python" -u -m experiments.feedback_active_v3.launch \
  --model "${QIHC_MODEL:-/inspire/hdd/global_user/czxs25210022/why/models/Qwen3-32B}" \
  --run "${QIHC_ACTIVE_RUN:-/inspire/hdd/global_user/czxs25210022/why/results/E004_2.1_active_feedback}" "$@"
