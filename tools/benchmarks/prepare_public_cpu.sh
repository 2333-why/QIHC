#!/usr/bin/env bash
# Only downloads public test data. Does not train, download models or reseal old runs.
set -Eeuo pipefail
CODE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)"
BASE="${QIHC_BASE:-/inspire/hdd/global_user/czxs25210022/why}"
PYTHON="${QIHC_DOWNLOAD_PYTHON:-$BASE/envs/qihc-cu128/bin/python}"
if [[ ! -x "$PYTHON" ]]; then
  PYTHON="$(command -v python3 || command -v python)"
fi
mkdir -p "$BASE/logs"
exec > >(tee -a "$BASE/logs/public-download-$(date -u +%Y%m%dT%H%M%SZ)-$$.log") 2>&1
exec "$PYTHON" "$CODE/tools/benchmarks/download_public.py" \
  --output "${QIHC_PUBLIC_DATA:-$BASE/data/public_benchmarks_v1}" "$@"
