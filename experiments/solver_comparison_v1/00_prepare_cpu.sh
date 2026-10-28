#!/usr/bin/env bash
set -Eeuo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/env.sh"
# Isolated additions only; never upgrade the shared Torch/Transformers environment.
bash "$CODE/experiments/solver_comparison_v1/prepare_solver_deps.sh"
bash experiments/prompt_efficiency_v3/03_prepare_fixed_cpu.sh
# The child preparation sets its own PYTHONPATH; this shell retains solver deps.
"$PY" -m experiments.solver_comparison_v1.run prepare
printf '%s\n' 'E001-1.3 + E002-1.1 prepared. CPU can exit; no weights redownloaded.'
