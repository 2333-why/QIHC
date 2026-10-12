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
exec bash "$CODE/scripts/server/02_run_offline_gpu.sh" "$@"
