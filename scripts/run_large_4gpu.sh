#!/usr/bin/env bash
# Submit ONCE on one worker with four H100 80GB. Do not wrap in torchrun.
set -Eeuo pipefail
CODE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
BASE="${QIHC_BASE:-/inspire/hdd/global_user/czxs25210022/why}"
ENV_PREFIX="${QIHC_ENV_PREFIX:-$BASE/envs/qihc-cu128}"
[[ $# == 0 ]] || { echo 'Usage: bash scripts/run_large_4gpu.sh'; exit 2; }
for name in WORLD_SIZE LOCAL_WORLD_SIZE GROUP_WORLD_SIZE ROLE_WORLD_SIZE; do
  [[ "${!name:-1}" == 1 ]] || { echo "Launch once, not per GPU: $name=${!name}"; exit 2; }
done
for name in RANK LOCAL_RANK GROUP_RANK ROLE_RANK; do
  [[ "${!name:-0}" == 0 ]] || { echo 'Requires worker rank zero'; exit 2; }
done
unset RANK LOCAL_RANK WORLD_SIZE LOCAL_WORLD_SIZE GROUP_RANK GROUP_WORLD_SIZE ROLE_RANK ROLE_WORLD_SIZE MASTER_ADDR MASTER_PORT
set +u
source "$BASE/miniconda3/etc/profile.d/conda.sh"
conda activate "$ENV_PREFIX"
set -u
export PYTHONNOUSERSITE=1 PYTHONPATH="$CODE" HF_HOME="$BASE/cache/huggingface"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 PIP_NO_INDEX=1
export WANDB_MODE=disabled TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1
export CUDA_HOME="$ENV_PREFIX" TORCH_EXTENSIONS_DIR="$BASE/cache/torch_extensions" TMPDIR="$BASE/tmp"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$BASE/logs" "$TMPDIR" "$TORCH_EXTENSIONS_DIR"
exec > >(tee -a "$BASE/logs/large-four-gpu-$(date -u +%Y%m%dT%H%M%SZ)-$$.log") 2>&1
cd "$CODE"
EXTRA=()
if [[ -n "${QIHC_LARGE_RESUME:-}" ]]; then EXTRA+=(--output "$QIHC_LARGE_RESUME"); fi
exec python scripts/large_model_pipeline.py run --base "$BASE" "${EXTRA[@]}"
