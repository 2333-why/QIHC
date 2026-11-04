#!/usr/bin/env bash
set -euo pipefail
export PIP_INDEX_URL="${PIP_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}"
export PIP_DEFAULT_TIMEOUT="${PIP_DEFAULT_TIMEOUT:-120}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ENV="${QIHC_ENV:-/inspire/hdd/global_user/czxs25210022/why/envs/qihc-cu128}"
export QIHC_ENV="$ENV"
python3 "$ROOT/experiments/feedback_active_v3/bootstrap_env.py" "$ENV"
source "$ROOT/experiments/feedback_active_v3/pip_install.sh"
qihc_pip_install "$ENV/bin/python" -r "$ROOT/experiments/feedback_active_v3/requirements-cpu.txt" -r "$ROOT/experiments/feedback_active_v3/requirements-gpu.txt"
"$ENV/bin/python" -m pip check
"$ENV/bin/python" -m pip freeze > "$ENV/requirements.resolved.txt"
touch "$ENV/.e004-setup-complete"
echo "Environment ready: $ENV"
