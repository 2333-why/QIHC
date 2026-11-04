#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ENV="${QIHC_ENV:-${CONDA_PREFIX:-${VIRTUAL_ENV:-/inspire/hdd/global_user/czxs25210022/why/.venv-qihc}}}"
export QIHC_ENV="$ENV"
python3 "$ROOT/experiments/feedback_active_v3/bootstrap_env.py" "$ENV"
"$ENV/bin/python" -m pip install --upgrade pip
"$ENV/bin/python" -m pip install -r "$ROOT/experiments/feedback_active_v3/requirements-cpu.txt" -r "$ROOT/experiments/feedback_active_v3/requirements-gpu.txt"
"$ENV/bin/python" -m pip check
"$ENV/bin/python" -m pip freeze > "$ENV/requirements.resolved.txt"
touch "$ENV/.e004-setup-complete"
echo "Environment ready: $ENV"
