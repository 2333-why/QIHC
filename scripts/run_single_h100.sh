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
export QIHC_NPROC=1
exec bash "$CODE/scripts/server/02_run_offline_gpu.sh" "$@"
