#!/usr/bin/env bash
set -euo pipefail
REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "${REPO_DIR}"
exec python experiments/run_research_suite.py \
  --profile formal \
  --model-path "${MODEL_DIR:-${REPO_DIR}/deployment/models/Qwen--Qwen3.5-35B-A3B}" \
  --nproc "${NPROC_PER_NODE:-8}" \
  --deepspeed-config configs/deepspeed-zero3-bf16.json \
  "$@"
