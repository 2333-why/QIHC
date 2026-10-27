#!/usr/bin/env bash
set -Eeuo pipefail
CODE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)"
BASE="${QIHC_BASE:-/inspire/hdd/global_user/czxs25210022/why}"
export PYTHONPATH="$CODE:${QIHC_FORMAT_DEPS:-$BASE/deps/modeling-v5-format-0113}" PYTHONNOUSERSITE=1 MPLBACKEND=Agg
cd "$CODE"
exec "${QIHC_ENV_PREFIX:-$BASE/envs/qihc-cu128}/bin/python" -m experiments.prompt_efficiency_v3.run report "$@"
