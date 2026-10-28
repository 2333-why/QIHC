#!/usr/bin/env bash
set -Eeuo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/env.sh"
mkdir -p "$(dirname -- "$SOLVERS")"
exec 8>"$SOLVERS.prepare.lock"
flock -n 8 || { echo 'Another solver preparation is active'; exit 2; }
if [[ ! -d "$SOLVERS" ]]; then
  TASK_TMP="$("$PY" -m experiments.solver_comparison_v1.staging "$(dirname -- "$SOLVERS")")"
  if [[ -z "$TASK_TMP" ]]; then
    TASK_TMP="$(mktemp -d "$(dirname -- "$SOLVERS")/.e002-solvers-XXXXXX")"
    "$PY" -m pip install --no-deps --only-binary=:all: --target "$TASK_TMP" \
      --index-url "${QIHC_PIP_INDEX:-https://pypi.tuna.tsinghua.edu.cn/simple}" \
      highspy==1.15.1 pyscipopt==6.3.0
  else
    printf 'Reusing downloaded solver staging: %s\n' "$TASK_TMP"
  fi
  PYTHONPATH="$CODE:$TASK_TMP:$FORMAT" "$PY" -m experiments.solver_comparison_v1.smoke
  mv -T -- "$TASK_TMP" "$SOLVERS"
fi
"$PY" -m experiments.solver_comparison_v1.smoke
