#!/usr/bin/env bash
# Single-GPU entry point; reuse the verified offline pipeline unchanged.
set -Eeuo pipefail
CODE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
if [[ "${1:-}" == '--help' ]]; then
  printf '%s\n' \
    'Usage: bash scripts/run_single_h100.sh [--preflight-only|--print-plan]' \
    'Run inside an allocated single-H100 Linux job after script 01 succeeds.' \
    'Automatically activates the prepared environment and runs the formal 9B suite.' \
    'Forces QIHC_NPROC=1; other offline pipeline overrides remain available.'
  exit 0
fi
# A one-worker platform job may already export torchrun-style metadata.
# Reject a genuine multi-process launch before clearing that metadata, so this
# entry point cannot accidentally run the full suite once per worker.
for name in WORLD_SIZE LOCAL_WORLD_SIZE GROUP_WORLD_SIZE ROLE_WORLD_SIZE; do
  value="${!name:-1}"
  [[ "$value" == 1 ]] || { printf 'Single-H100 entry requires one worker/process: %s=%s\n' "$name" "$value" >&2; exit 2; }
done
for name in RANK LOCAL_RANK GROUP_RANK ROLE_RANK; do
  value="${!name:-0}"
  [[ "$value" == 0 ]] || { printf 'Single-H100 entry requires rank zero: %s=%s\n' "$name" "$value" >&2; exit 2; }
done
unset RANK LOCAL_RANK WORLD_SIZE LOCAL_WORLD_SIZE GROUP_RANK GROUP_WORLD_SIZE ROLE_RANK ROLE_WORLD_SIZE
unset MASTER_ADDR MASTER_PORT

# nvidia-smi is informational, not a CUDA availability test. Some GPU job
# containers expose CUDA through PyTorch without installing this CLI. Export
# a narrowly scoped fallback to the unchanged child launcher; its mandatory
# PyTorch/BF16 and NCCL/ZeRO-3 checks remain in force.
if ! command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi() {
    printf '%s\n' 'INFO: nvidia-smi unavailable; mandatory PyTorch CUDA/BF16 checks follow.'
  }
  export -f nvidia-smi
fi
export QIHC_NPROC=1
if [[ "${1:-}" != '--print-plan' ]]; then
  BASE="${QIHC_BASE:-/inspire/hdd/global_user/czxs25210022/why}"
  ENV_PREFIX="$BASE/envs/qihc-${QIHC_CUDA_WHEEL:-cu128}"
  if [[ -x "$ENV_PREFIX/bin/python" && -f "$BASE/artifacts/prepared.json" ]]; then
    # Revalidation imports DeepSpeed even for its tiny CPU training test.
    # Activate the toolkit before that import, not only in the child launcher.
    set +u
    source "$BASE/miniconda3/etc/profile.d/conda.sh"
    conda activate "$ENV_PREFIX"
    set -u
    export CUDA_HOME="$ENV_PREFIX" CUDA_DEVICE_ORDER=PCI_BUS_ID
    export HF_HOME="$BASE/cache/huggingface" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
    export WANDB_MODE=disabled PIP_NO_INDEX=1 TOKENIZERS_PARALLELISM=false MPLBACKEND=Agg
    export PYTHONNOUSERSITE=1 PYTHONPATH="${QIHC_CODE_DIR:-$CODE}"
    export TORCH_EXTENSIONS_DIR="$BASE/cache/torch_extensions" TRITON_CACHE_DIR="$BASE/cache/triton"
    export TMPDIR="$BASE/tmp" MAX_JOBS="${QIHC_BUILD_JOBS:-4}"
    export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}" OPENBLAS_NUM_THREADS=1
    mkdir -p "$TORCH_EXTENSIONS_DIR" "$TRITON_CACHE_DIR" "$TMPDIR"
    PYTHONNOUSERSITE=1 "$ENV_PREFIX/bin/python" "$CODE/scripts/refresh_non_thinking_preparation.py" \
      --manifest "$BASE/artifacts/prepared.json" --code-dir "${QIHC_CODE_DIR:-$CODE}" --env-prefix "$ENV_PREFIX"
  fi
fi
exec bash "$CODE/scripts/server/02_run_offline_gpu.sh" "$@"
