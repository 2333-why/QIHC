#!/usr/bin/env bash
# One scheduler worker / four H100 recommended; NOT torchrun or per-GPU launch.
set -Eeuo pipefail
CODE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd -P)"
BASE="${QIHC_BASE:-/inspire/hdd/global_user/czxs25210022/why}"
ENV_PREFIX="${QIHC_ENV_PREFIX:-$BASE/envs/qihc-cu128}"
RUN="${QIHC_DISTRIBUTION_RUN:-$BASE/results/distribution_feedback_v1}"
for name in WORLD_SIZE LOCAL_WORLD_SIZE GROUP_WORLD_SIZE ROLE_WORLD_SIZE; do
  [[ "${!name:-1}" == 1 ]] || { echo 'Launch script once in one worker, not through torchrun'; exit 2; }
done
for name in RANK LOCAL_RANK GROUP_RANK ROLE_RANK; do
  [[ "${!name:-0}" == 0 ]] || { echo 'Only scheduler rank-zero worker is supported'; exit 2; }
done
unset RANK LOCAL_RANK WORLD_SIZE LOCAL_WORLD_SIZE GROUP_RANK GROUP_WORLD_SIZE ROLE_RANK ROLE_WORLD_SIZE MASTER_ADDR MASTER_PORT
export PATH="$ENV_PREFIX/bin:$PATH" PYTHONPATH="$CODE" PYTHONNOUSERSITE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 PIP_NO_INDEX=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$BASE/logs"
exec > >(tee -a "$BASE/logs/distribution-gpu-$(date -u +%Y%m%dT%H%M%SZ)-$$.log") 2>&1
trap 'echo "New experiment GPU failed at line $LINENO; completed checkpoints are retained"' ERR
cd "$CODE"
exec "$ENV_PREFIX/bin/python" tools/benchmarks/distribution/inference.py launch --run "$RUN" \
  --replicas "${QIHC_DISTRIBUTION_REPLICAS:-2}" --batch-size "${QIHC_DISTRIBUTION_BATCH:-4}" \
  --idle-exit "${QIHC_DISTRIBUTION_IDLE_EXIT:-120}"
