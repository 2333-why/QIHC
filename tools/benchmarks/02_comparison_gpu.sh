#!/usr/bin/env bash
# Single worker entry; supports inherited rank-zero scheduler variables, no torchrun.
set -Eeuo pipefail
CODE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)"
BASE="${QIHC_BASE:-/inspire/hdd/global_user/czxs25210022/why}"
ENV_PREFIX="${QIHC_ENV_PREFIX:-$BASE/envs/qihc-cu128}"
RUN="${QIHC_COMPARISON_RUN:-$BASE/results/comparison_v1}"
for name in WORLD_SIZE LOCAL_WORLD_SIZE GROUP_WORLD_SIZE ROLE_WORLD_SIZE; do
  [[ "${!name:-1}" == 1 ]] || { echo 'Launch once, not inside torchrun'; exit 2; }
done
for name in RANK LOCAL_RANK GROUP_RANK ROLE_RANK; do
  [[ "${!name:-0}" == 0 ]] || { echo 'Only rank-zero worker supported'; exit 2; }
done
unset RANK LOCAL_RANK WORLD_SIZE LOCAL_WORLD_SIZE GROUP_RANK GROUP_WORLD_SIZE ROLE_RANK ROLE_WORLD_SIZE MASTER_ADDR MASTER_PORT
export PYTHONPATH="$CODE" PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 PIP_NO_INDEX=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$BASE/logs"
exec > >(tee -a "$BASE/logs/comparison-gpu-$(date -u +%Y%m%dT%H%M%SZ)-$$.log") 2>&1
cd "$CODE"
[[ -f "$RUN/manifest.json" ]] || { echo 'CPU preparation not complete'; exit 2; }
"$ENV_PREFIX/bin/python" -c 'import torch; assert torch.cuda.is_available(), "CUDA unavailable"; print("Visible GPUs:", torch.cuda.device_count())'
if [[ "${QIHC_PARALLEL_INFERENCE:-0}" == 1 ]]; then
  exec "$ENV_PREFIX/bin/python" tools/benchmarks/parallel_inference.py launch --run "$RUN" --replicas "${QIHC_INFERENCE_REPLICAS:-2}" --batch-size "${QIHC_INFERENCE_BATCH:-4}" --idle-exit 45
fi
exec "$ENV_PREFIX/bin/python" tools/benchmarks/comparison.py gpu --run "$RUN" --idle-exit 45
