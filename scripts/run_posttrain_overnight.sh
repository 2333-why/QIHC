#!/usr/bin/env bash
# Independent, bounded medium-scale post-training experiment.
set -Eeuo pipefail
CODE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
BASE="${QIHC_BASE:-/inspire/hdd/global_user/czxs25210022/why}"
if [[ "${1:-}" == '--print-plan' ]]; then
  printf '%s\n' 'Qwen3.5-9B; 100/200/300 customers; 36 instances/size (108 total).' \
    '72 training / 18 validation / 18 test geometries; test seeds 0,1 (36 paired jobs).' \
    'Base and pure KNN+pbit -> 6-iteration, 2-seed teacher (up to 864 proposals) -> SFT 120 steps -> regularized DPO <=40 steps.' \
    'DPO requires >=64 verified proposal pairs; at most one data pass; validation strict-output gate >=95%.' \
    'Shared neutral starts; 3 search iterations/evaluation; no LLM cold-start training or GRPO.' \
    'Default wall-clock cap: 24 hours. Partial results saved; no guarantee all stages finish.' \
    'Use a separate allocated H100, or wait for the existing GPU job to finish.'
  exit 0
fi
[[ $# == 0 ]] || { echo 'Usage: bash scripts/run_posttrain_overnight.sh [--print-plan]' >&2; exit 2; }
for name in WORLD_SIZE LOCAL_WORLD_SIZE GROUP_WORLD_SIZE ROLE_WORLD_SIZE; do
  [[ "${!name:-1}" == 1 ]] || { echo "Requires one worker: $name=${!name}" >&2; exit 2; }
done
for name in RANK LOCAL_RANK GROUP_RANK ROLE_RANK; do
  [[ "${!name:-0}" == 0 ]] || { echo "Requires rank zero: $name=${!name}" >&2; exit 2; }
done
unset RANK LOCAL_RANK WORLD_SIZE LOCAL_WORLD_SIZE GROUP_RANK GROUP_WORLD_SIZE ROLE_RANK ROLE_WORLD_SIZE MASTER_ADDR MASTER_PORT
ENV_PREFIX="$BASE/envs/qihc-${QIHC_CUDA_WHEEL:-cu128}"
set +u
source "$BASE/miniconda3/etc/profile.d/conda.sh"
conda activate "$ENV_PREFIX"
set -u
export CUDA_HOME="$ENV_PREFIX" CUDA_DEVICE_ORDER=PCI_BUS_ID
export HF_HOME="$BASE/cache/huggingface" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export PIP_NO_INDEX=1 PYTHONNOUSERSITE=1 PYTHONPATH="$CODE" WANDB_MODE=disabled TOKENIZERS_PARALLELISM=false MPLBACKEND=Agg
export TORCH_EXTENSIONS_DIR="$BASE/cache/torch_extensions" TRITON_CACHE_DIR="$BASE/cache/triton" TMPDIR="$BASE/tmp"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}" OPENBLAS_NUM_THREADS=1 MAX_JOBS=4
mkdir -p "$TORCH_EXTENSIONS_DIR" "$TRITON_CACHE_DIR" "$TMPDIR" "$BASE/logs"
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)-$$"
exec > >(tee -a "$BASE/logs/overnight-$RUN_ID.log") 2>&1
cd "$CODE"
python -c 'import torch; assert torch.cuda.is_available(), "CUDA unavailable"; free,total=torch.cuda.mem_get_info(0); print("GPU:",torch.cuda.get_device_name(0),"free GiB:",free/2**30); assert free >= 60*2**30, "Need a separately allocated H100 with >=60 GiB free; do not share with the running formal job"'
python scripts/refresh_non_thinking_preparation.py --manifest "$BASE/artifacts/prepared.json" --code-dir "$CODE" --env-prefix "$ENV_PREFIX"
python experiments/server_readiness.py verify --manifest "$BASE/artifacts/prepared.json" --code-dir "$CODE" --env-prefix "$ENV_PREFIX"
exec python scripts/posttrain_overnight.py --manifest "$BASE/artifacts/prepared.json" \
  --output "$BASE/results/posttrain_overnight_$RUN_ID" --hours "${QIHC_OVERNIGHT_HOURS:-24}"
