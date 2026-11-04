#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ENV="${QIHC_ACTIVE_ENV:-$ROOT/.venv-e004-active}"
python3 -m venv "$ENV"
"$ENV/bin/python" -m pip install --upgrade pip
"$ENV/bin/python" -m pip install -r "$ROOT/experiments/feedback_active_v3/requirements-cpu.txt" -r "$ROOT/experiments/feedback_active_v3/requirements-gpu.txt"
"$ENV/bin/python" -m pip check
"$ENV/bin/python" -m pip freeze > "$ENV/requirements.resolved.txt"
echo "Environment ready: $ENV"
