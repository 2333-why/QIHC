#!/usr/bin/env bash
set -Eeuo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/env.sh"
# E002 dependencies only: never prepare or launch E001.
mkdir -p "$(dirname -- "$SOLVERS")"
exec 8>"$SOLVERS.prepare.lock"
flock -n 8 || { echo 'Another solver preparation is active'; exit 2; }
if [[ ! -d "$SOLVERS" ]]; then
  TASK_TMP="$(mktemp -d "$(dirname -- "$SOLVERS")/.e002-solvers-XXXXXX")"
  "$PY" -m pip install --no-deps --only-binary=:all: --target "$TASK_TMP" \
    --index-url "${QIHC_PIP_INDEX:-https://pypi.tuna.tsinghua.edu.cn/simple}" \
    highspy==1.15.1 pyscipopt==6.3.0
  PYTHONPATH="$CODE:$TASK_TMP:$FORMAT" "$PY" -m experiments.solver_comparison_v1.smoke
  mv -T -- "$TASK_TMP" "$SOLVERS"
fi
"$PY" -m experiments.solver_comparison_v1.smoke
printf '%s\n' 'E002 solver environment READY. E001 was not started. Run 06 after its frozen modeling outputs are available.'
