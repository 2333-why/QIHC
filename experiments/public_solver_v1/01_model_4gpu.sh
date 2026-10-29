#!/usr/bin/env bash
set -Eeuo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/env.sh"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Ignore single-worker platform rank metadata: this launcher owns two GPU pairs.
# Actual duplicate launches are prevented by the process-lifetime filesystem lock.
exec "$PY" -u -m experiments.public_solver_v1.modeling launch "$@"
