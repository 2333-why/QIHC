#!/usr/bin/env bash
# Existing QIHC environment is reused; old weights/manifest are untouched.
set -Eeuo pipefail
CODE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
BASE="${QIHC_BASE:-/inspire/hdd/global_user/czxs25210022/why}"
ENV_PREFIX="${QIHC_ENV_PREFIX:-$BASE/envs/qihc-cu128}"
[[ $# == 0 ]] || { echo 'Usage: bash scripts/prepare_large_cpu.sh'; exit 2; }
[[ -f "$BASE/miniconda3/etc/profile.d/conda.sh" ]] || { echo 'Existing Miniconda/environment required; run original preparation on a fresh machine.'; exit 2; }
set +u
source "$BASE/miniconda3/etc/profile.d/conda.sh"
conda activate "$ENV_PREFIX"
set -u
export PYTHONNOUSERSITE=1 PYTHONPATH="$CODE" HF_HOME="$BASE/cache/huggingface"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}" HF_HUB_DISABLE_XET=1
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE HF_DATASETS_OFFLINE
mkdir -p "$BASE/logs"
exec > >(tee -a "$BASE/logs/prepare-large-$(date -u +%Y%m%dT%H%M%SZ)-$$.log") 2>&1
cd "$CODE"
python scripts/large_model_pipeline.py prepare --base "$BASE" --count "${QIHC_LARGE_CASES:-240}"
