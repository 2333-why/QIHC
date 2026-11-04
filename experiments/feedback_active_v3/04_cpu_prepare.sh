#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
ENV="${QIHC_CPU_ENV:-$ROOT/.venv-e004-cpu}"
if [[ ! -x "$ENV/bin/python" ]]; then
  python3 -m venv "$ENV"
fi
# An interrupted installation may leave a working Python without dependencies.
if ! "$ENV/bin/python" -m pip --version >/dev/null 2>&1; then
  "$ENV/bin/python" -m ensurepip --upgrade
fi
"$ENV/bin/python" -m pip install -r experiments/feedback_active_v3/requirements-cpu.txt
"$ENV/bin/python" -m pip check
"$ENV/bin/python" -m experiments.feedback_active_v3.assets restore
"$ENV/bin/python" -m experiments.feedback_active_v3.validate_data
echo "CPU preparation complete: dependencies and public data validated."
