#!/usr/bin/env bash
set -Eeuo pipefail
CODE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)"
BASE="${QIHC_BASE:-/inspire/hdd/global_user/czxs25210022/why}"
PY="${QIHC_ENV_PREFIX:-$BASE/envs/qihc-cu128}/bin/python"
TARGET="${QIHC_FORMAT_DEPS:-$BASE/deps/modeling-v5-format-0113}"
export PYTHONNOUSERSITE=1 MPLBACKEND=Agg OMP_NUM_THREADS=1
export PYTHONPATH="$CODE:$TARGET"
mkdir -p -- "$(dirname -- "$TARGET")"
exec 9>"$TARGET.prepare.lock"
flock -n 9 || { echo 'Another format preparation is active'; exit 2; }
if [[ ! -e "$TARGET" ]]; then
  TASK_TMP="$(mktemp -d "$(dirname -- "$TARGET")/.prompt-format-XXXXXX")"
  "$PY" -m pip install --no-deps --only-binary=:all: --target "$TASK_TMP" \
    --index-url "${QIHC_PIP_INDEX:-https://pypi.tuna.tsinghua.edu.cn/simple}" \
    lm-format-enforcer==0.11.3 interegular==0.3.3
  PYTHONPATH="$CODE:$TASK_TMP" "$PY" -c 'from qihc.modeling.formatting import check_runtime; print(check_runtime())'
  mv -T -- "$TASK_TMP" "$TARGET"
fi
cd "$CODE"
"$PY" -u -m experiments.prompt_efficiency_v1.run prepare "$@"
