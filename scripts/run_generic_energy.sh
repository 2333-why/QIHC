#!/usr/bin/env bash
# Single worker / single H100. Reuse prepared environment and weights, offline.
set -Eeuo pipefail
CODE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
BASE="${QIHC_BASE:-/inspire/hdd/global_user/czxs25210022/why}"
if [[ "${1:-}" == '--print-plan' ]]; then
  printf '%s\n' 'Generic binary NL optimization: knapsack, vertex cover, balanced cut, assignment.' \
    'Default 192 instances; disjoint train/validation/test; balanced-cut family excluded from all training.' \
    'Two-stage LLM objective/constraint IR; compiler + random-start pbit; independent gold verification.' \
    'CP-SAT, random-scan fair bits, Metropolis, fixed/adaptive pbit, direct LLM, base/SFT/DPO + matched no-feedback ablations.' \
    'Single H100 for real SFT/DPO; CPU float64 pbit emulator. No FPGA or energy-efficiency claim.' \
    'Existing prepared model reused; no download. Default 36-hour cap. Separate free GPU required.'
  exit 0
fi
[[ $# == 0 ]] || { echo 'Usage: bash scripts/run_generic_energy.sh [--print-plan]' >&2; exit 2; }
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
exec > >(tee -a "$BASE/logs/generic-energy-$RUN_ID.log") 2>&1
cd "$CODE"
python -c 'import torch; assert torch.cuda.is_available(), "CUDA unavailable"; free,total=torch.cuda.mem_get_info(0); print("GPU:",torch.cuda.get_device_name(0),"free GiB:",free/2**30); assert free >= 60*2**30, "Need >=60 GiB free on a separately allocated H100"'
python scripts/refresh_non_thinking_preparation.py --manifest "$BASE/artifacts/prepared.json" --code-dir "$CODE" --env-prefix "$ENV_PREFIX"
python experiments/server_readiness.py verify --manifest "$BASE/artifacts/prepared.json" --code-dir "$CODE" --env-prefix "$ENV_PREFIX"
exec python scripts/generic_energy_pipeline.py --manifest "$BASE/artifacts/prepared.json" \
  --output "$BASE/results/generic_energy_$RUN_ID" --hours "${QIHC_GENERIC_HOURS:-36}" \
  --per-family "${QIHC_GENERIC_CASES:-48}" --steps "${QIHC_GENERIC_STEPS:-4096}" --chains "${QIHC_GENERIC_CHAINS:-64}"
