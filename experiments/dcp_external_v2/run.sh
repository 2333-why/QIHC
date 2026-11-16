#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
export QIHC_BASE="${QIHC_BASE:-/inspire/hdd/global_user/czxs25210022/why}"
export QIHC_ENV="${QIHC_ENV:-$QIHC_BASE/envs/qihc-cu128}"
export QIHC_DCP_RUN="${QIHC_DCP_RUN:-$QIHC_BASE/results/E005-1.17_dcp235}"
export QIHC_DCP_UPSTREAM="${QIHC_DCP_UPSTREAM:-$QIHC_BASE/DCP-Bench-Open-v0.3.0}"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
PY="$QIHC_ENV/bin/python"
test -x "$PY"
case "${1:-}" in
  prepare)
    if ! "$PY" -c 'from importlib.metadata import version; assert version("cpmpy")=="1.0.0"' 2>/dev/null; then
      source experiments/feedback_active_v3/pip_install.sh
      qihc_pip_install "$PY" --no-deps cpmpy==1.0.0
    fi
    "$PY" -m experiments.feedback_active_v3.check_env --gpu
    if [[ ! -d "$QIHC_DCP_UPSTREAM/.git" ]]; then
      git clone --depth 1 --branch v0.3.0 https://github.com/DCP-Bench/DCP-Bench-Open.git "$QIHC_DCP_UPSTREAM"
    fi
    prior="$QIHC_BASE/results/E005-1.16_dcp/dcp-bench-open-v0.3.0.jsonl"
    if [[ -z "${QIHC_DCP_ASSET:-}" && ! -f "$prior" ]]; then
      export QIHC_DCP_ASSET="$QIHC_DCP_RUN/source/dcp-bench-open-v0.3.0.jsonl"
      mkdir -p "$QIHC_DCP_RUN/source"
      if [[ ! -f "$QIHC_DCP_ASSET" ]]; then
        tmp=$(mktemp "$QIHC_DCP_RUN/source/download.XXXXXXXX")
        trap 'rm -f -- "$tmp"' EXIT
        curl --fail --location --retry 3 -o "$tmp" https://github.com/DCP-Bench/DCP-Bench-Open/releases/download/v0.3.0/dcp-bench-open.jsonl
        echo "bf4f50f435c03a5600194306937e37039cdfa2e002dfa1c6548cd815c8601385  $tmp" | sha256sum -c -
        mv "$tmp" "$QIHC_DCP_ASSET"
      fi
    fi
    exec "$PY" -m experiments.dcp_external_v2.cpu prepare ;;
  gpu) exec "$PY" -m experiments.dcp_external_v2.gpu ;;
  finish) exec "$PY" -m experiments.dcp_external_v2.cpu finish ;;
  report) exec "$PY" -m experiments.dcp_external_v2.report ;;
  *) echo "Usage: bash experiments/dcp_external_v2/run.sh {prepare|gpu|finish|report}" >&2; exit 2 ;;
esac
