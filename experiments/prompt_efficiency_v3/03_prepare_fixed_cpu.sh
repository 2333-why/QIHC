#!/usr/bin/env bash
set -Eeuo pipefail
BASE="${QIHC_BASE:-/inspire/hdd/global_user/czxs25210022/why}"
export QIHC_PROMPT_V3_RUN="${QIHC_PROMPT_V3_RUN:-$BASE/results/E001-1.3_prompt_fixed}"
bash "$(dirname -- "${BASH_SOURCE[0]}")/00_prepare_cpu.sh" "$@"
