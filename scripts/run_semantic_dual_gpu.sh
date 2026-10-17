#!/usr/bin/env bash
# One worker with TWO visible H100 80GB cards. Run once, not inside torchrun.
set -Eeuo pipefail
CODE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
if [[ "${1:-}" == '--print-plan' ]]; then
  printf '%s\n' \
    'One worker / 2 H100 80GB; internal torchrun + ZeRO-3, global train batch 4.' \
    '768 synthetic cases (192/family); 384 training cases excluding all balanced-cut cases.' \
    'Train sizes 12/20/32 bits or 3/4/5 assignment; eval includes 40 bits / 6x6 assignment.' \
    'Joint objective+constraints SFT; witnessed semantic DPO + SFT error mining.' \
    'Base, SFT, policy-only DPO, semantic-only DPO, mixed DPO; identical inference prompt and search budget.' \
    'Three DPO arms start independently from the SAME SFT adapter, matched 128 optimizer steps.' \
    'Real longest-example SFT/DPO memory smoke; no model downloads or first-step rerun.' \
    'Default 96-hour cap (not runtime promise). CPU pbit emulator, no FPGA energy claim.'
  exit 0
fi
[[ $# == 0 ]] || { echo 'Usage: bash scripts/run_semantic_dual_gpu.sh [--print-plan]' >&2; exit 2; }
export QIHC_GENERIC_NPROC=2 QIHC_GENERIC_SEMANTIC=1
export QIHC_GENERIC_CASES="${QIHC_GENERIC_CASES:-192}"
export QIHC_GENERIC_HOURS="${QIHC_GENERIC_HOURS:-96}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
exec bash "$CODE/scripts/run_generic_energy.sh"
