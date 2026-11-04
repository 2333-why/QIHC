#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ENV="${QIHC_ACTIVE_ENV:-$ROOT/.venv-e004-active}"
if [[ ! -x "$ENV/bin/python" ]]; then
  python3 -m venv "$ENV"
fi
if ! "$ENV/bin/python" -m pip --version >/dev/null 2>&1; then
  "$ENV/bin/python" -m ensurepip --upgrade
fi
"$ENV/bin/python" -m pip install --upgrade pip
"$ENV/bin/python" -m pip install -r "$ROOT/experiments/feedback_active_v3/requirements-cpu.txt" -r "$ROOT/experiments/feedback_active_v3/requirements-gpu.txt"
"$ENV/bin/python" -m pip check
"$ENV/bin/python" -m pip freeze > "$ENV/requirements.resolved.txt"
touch "$ENV/.e004-setup-complete"
echo "Environment ready: $ENV"
