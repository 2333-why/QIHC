#!/usr/bin/env bash
set -Eeuo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/env.sh"
# Isolated additions only; never upgrade the shared Torch/Transformers environment.
mkdir -p "$BASE/deps"
exec 8>"$SOLVERS.prepare.lock"
flock -n 8 || { echo 'Another solver preparation is active'; exit 2; }
if [[ ! -d "$SOLVERS" ]]; then
  TASK_TMP="$(mktemp -d "$BASE/deps/.e002-solvers-XXXXXX")"
  "$PY" -m pip install --no-deps --only-binary=:all: --target "$TASK_TMP" \
    --index-url "${QIHC_PIP_INDEX:-https://pypi.tuna.tsinghua.edu.cn/simple}" \
    highspy==1.15.1 pyscipopt==6.3.0
  PYTHONPATH="$CODE:$TASK_TMP:$FORMAT" "$PY" -m experiments.solver_comparison_v1.smoke
  mv -T -- "$TASK_TMP" "$SOLVERS"
fi
"$PY" -m experiments.solver_comparison_v1.smoke
bash experiments/prompt_efficiency_v3/03_prepare_fixed_cpu.sh
# The child preparation sets its own PYTHONPATH; this shell retains solver deps.
"$PY" -m experiments.solver_comparison_v1.run prepare
printf '%s\n' 'E001-1.3 + E002-1.1 prepared. CPU can exit; no weights redownloaded.'
