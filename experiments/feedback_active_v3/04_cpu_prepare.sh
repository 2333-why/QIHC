#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
ENV="${QIHC_ENV:-${CONDA_PREFIX:-${VIRTUAL_ENV:-/inspire/hdd/global_user/czxs25210022/why/.venv-qihc}}}"
export QIHC_ENV="$ENV"
python3 "$ROOT/experiments/feedback_active_v3/bootstrap_env.py" "$ENV"
"$ENV/bin/python" -m pip install -r experiments/feedback_active_v3/requirements-cpu.txt
"$ENV/bin/python" -m pip check
"$ENV/bin/python" -m experiments.feedback_active_v3.assets restore
"$ENV/bin/python" -m experiments.feedback_active_v3.validate_data
echo "CPU preparation complete: dependencies and public data validated."
