#!/usr/bin/env bash
# GPU jobs only perform inference; CPU jobs perform search/validation.
set -Eeuo pipefail
CODE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)"
BASE="${QIHC_BASE:-/inspire/hdd/global_user/czxs25210022/why}"
ENV_PREFIX="${QIHC_ENV_PREFIX:-$BASE/envs/qihc-cu128}"
RUN="${QIHC_PUBLIC_RUN:-$BASE/results/public_simulation_v1}"
STAGE="${1:-}"
ARM="${2:-base}"
[[ "$ARM" == base || "$ARM" == feedback || "$ARM" == self_review ]] || { echo 'Invalid arm'; exit 2; }
[[ "$STAGE" == prepare || "$STAGE" == infer || "$STAGE" == score ]] || { echo 'Usage: bash tools/benchmarks/run_public_simulation.sh prepare|infer|score [base|feedback|self_review]'; exit 2; }
if [[ "$STAGE" == infer ]]; then
  for name in WORLD_SIZE LOCAL_WORLD_SIZE GROUP_WORLD_SIZE ROLE_WORLD_SIZE; do
    [[ "${!name:-1}" == 1 ]] || { echo 'Launch once per worker, not inside torchrun'; exit 2; }
  done
  for name in RANK LOCAL_RANK GROUP_RANK ROLE_RANK; do
    [[ "${!name:-0}" == 0 ]] || { echo 'Only rank zero supported'; exit 2; }
  done
  unset RANK LOCAL_RANK WORLD_SIZE LOCAL_WORLD_SIZE GROUP_RANK GROUP_WORLD_SIZE ROLE_RANK ROLE_WORLD_SIZE MASTER_ADDR MASTER_PORT
else
  export CUDA_VISIBLE_DEVICES=""
fi
export PYTHONPATH="$CODE" PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
mkdir -p "$BASE/logs"
exec > >(tee -a "$BASE/logs/public-sim-$STAGE-$ARM-$(date -u +%Y%m%dT%H%M%SZ)-$$.log") 2>&1
cd "$CODE"
PYTHON="$ENV_PREFIX/bin/python"
[[ -x "$PYTHON" ]] || { echo "Missing existing environment: $PYTHON"; exit 2; }
REQUESTS="$RUN/data/requests.jsonl"
if [[ "$ARM" != base ]]; then REQUESTS="$RUN/base.scores.$ARM.requests.jsonl"; fi
if [[ "$STAGE" == prepare ]]; then
  exec "$PYTHON" tools/benchmarks/public_simulation.py prepare \
    --data "${QIHC_PUBLIC_DATA:-$BASE/data/public_benchmarks_v1}/54ca621042b4619b90ce23ba3edd710611822e02" \
    --output "$RUN/data" --per-dataset "${QIHC_PUBLIC_PER_DATASET:-40}"
elif [[ "$STAGE" == infer ]]; then
  EXTRA=()
  if [[ -n "${QIHC_PUBLIC_ADAPTER:-}" ]]; then EXTRA+=(--adapter "$QIHC_PUBLIC_ADAPTER"); fi
  exec "$PYTHON" tools/benchmarks/public_simulation.py infer --requests "$REQUESTS" \
    --model "${QIHC_PUBLIC_MODEL:-$BASE/models/Qwen3-32B}" --output "$RUN/$ARM.predictions.jsonl" "${EXTRA[@]}"
else
  exec "$PYTHON" tools/benchmarks/public_simulation.py score --requests "$REQUESTS" \
    --predictions "$RUN/$ARM.predictions.jsonl" --answers "$RUN/data/answers.DO_NOT_PROMPT.jsonl" \
    --output "$RUN/$ARM.scores.jsonl" --steps "${QIHC_PUBLIC_STEPS:-4096}" \
    --chains "${QIHC_PUBLIC_CHAINS:-16}" --seeds "${QIHC_PUBLIC_SEEDS:-3}"
fi
