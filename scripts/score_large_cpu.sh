#!/usr/bin/env bash
# Run on an allocated CPU compute node, not inside the four-GPU job.
set -Eeuo pipefail
CODE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
BASE="${QIHC_BASE:-/inspire/hdd/global_user/czxs25210022/why}"
[[ $# == 1 ]] || { echo 'Usage: bash scripts/score_large_cpu.sh /absolute/results/run_directory'; exit 2; }
RUN_DIR="$1"
[[ -f "$RUN_DIR/run.json" ]] || { echo 'Existing run.json required'; exit 2; }
set +u
source "$BASE/miniconda3/etc/profile.d/conda.sh"
conda activate "${QIHC_ENV_PREFIX:-$BASE/envs/qihc-cu128}"
set -u
export PYTHONPATH="$CODE" PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=""
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
mkdir -p "$BASE/logs"
exec > >(tee -a "$BASE/logs/large-cpu-score-$(date -u +%Y%m%dT%H%M%SZ)-$$.log") 2>&1
cd "$CODE"
exec python scripts/large_model_pipeline.py cpu-score --base "$BASE" --output "$RUN_DIR"
