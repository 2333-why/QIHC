#!/usr/bin/env bash
set -Eeuo pipefail
CODE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd -P)"
BASE="${QIHC_BASE:-/inspire/hdd/global_user/czxs25210022/why}"
PY="${QIHC_ENV_PREFIX:-$BASE/envs/qihc-cu128}/bin/python"
TARGET="${QIHC_FORMAT_DEPS:-$BASE/deps/modeling-v5-format-0113}"
export PYTHONNOUSERSITE=1 MPLBACKEND=Agg OMP_NUM_THREADS=1
mkdir -p -- "$(dirname -- "$TARGET")"
exec 9>"$TARGET.prepare.lock"
flock -n 9 || { echo 'Another format dependency preparation is active'; exit 2; }
if [[ ! -e "$TARGET" ]]; then
  TASK_TMP="$(mktemp -d "$(dirname -- "$TARGET")/.format-v6-XXXXXX")"
  "$PY" -m pip install --no-deps --only-binary=:all: --target "$TASK_TMP" \
    --index-url "${QIHC_PIP_INDEX:-https://pypi.tuna.tsinghua.edu.cn/simple}" \
    lm-format-enforcer==0.11.3 interegular==0.3.3
  PYTHONPATH="$CODE:$TASK_TMP" "$PY" -c 'from tools.runtime.modeling_v6.formatting import check_dependencies,parser; check_dependencies(); parser(); from lmformatenforcer.integrations.transformers import build_transformers_prefix_allowed_tokens_fn; print("Format dependency and Transformers imports passed")'
  mv -T -- "$TASK_TMP" "$TARGET"
fi
PYTHONPATH="$CODE:$TARGET" "$PY" -c 'from tools.runtime.modeling_v6.formatting import check_dependencies,parser; check_dependencies(); parser(); from lmformatenforcer.integrations.transformers import build_transformers_prefix_allowed_tokens_fn; print("V6 format environment READY; existing v5 dependency directory reused without upgrades")'
PYTHONPATH="$CODE:$TARGET" "$PY" -m tools.runtime.modeling_v6.preflight
