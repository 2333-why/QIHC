#!/usr/bin/env bash
set -Eeuo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/env.sh"
# Reuse existing isolated dependencies and weights; never upgrade shared torch.
bash experiments/solver_comparison_v1/prepare_solver_deps.sh
if [[ ! -d "$FORMAT" ]]; then
  mkdir -p "$(dirname -- "$FORMAT")"
  exec 9>"$FORMAT.prepare.lock"
  flock -n 9 || { echo 'Another format preparation is active'; exit 2; }
  TASK_TMP="$(mktemp -d "$(dirname -- "$FORMAT")/.e004-format-XXXXXX")"
  "$PY" -m pip install --no-deps --only-binary=:all: --target "$TASK_TMP" \
    --index-url "${QIHC_PIP_INDEX:-https://pypi.tuna.tsinghua.edu.cn/simple}" \
    lm-format-enforcer==0.11.3 interegular==0.3.3
  PYTHONPATH="$CODE:$TASK_TMP" "$PY" -c 'from qihc.modeling.formatting import check_runtime; print(check_runtime())'
  mv -T -- "$TASK_TMP" "$FORMAT"
fi
"$PY" -u -m experiments.feedback_study_v1.smoke
"$PY" -u -m experiments.feedback_study_v1.prepare "$@"
