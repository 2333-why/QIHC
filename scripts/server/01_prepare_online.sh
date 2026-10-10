#!/usr/bin/env bash
# Run on an ONLINE Linux x86_64 CPU node. No sudo or shell-profile edits.
set -Eeuo pipefail
umask 077
trap 'printf "Preparation failed at line %s. Offline launch is NOT ready.\n" "$LINENO" >&2' ERR
BASE="${QIHC_BASE:-/inspire/hdd/global_user/czxs25210022/why}"
CODE="${QIHC_CODE_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)}"
CUDA_FLAVOR="${QIHC_CUDA_WHEEL:-cu128}"
MODEL_ID="${QIHC_MODEL_ID:-Qwen/Qwen3.5-9B}"
REVISION="${QIHC_MODEL_REVISION:-main}"
CONDA_ROOT="${BASE}/miniconda3"
ENV_PREFIX="${BASE}/envs/qihc-${CUDA_FLAVOR}"
MODEL_DIR="${QIHC_MODEL_DIR:-${BASE}/models/${MODEL_ID##*/}}"
if [[ "${1:-}" == '--help' ]]; then
  printf '%s\n' 'Usage: bash scripts/server/01_prepare_online.sh [--print-plan]' \
    'Defaults: shared why directory, Python 3.11, Qwen3.5-9B, CUDA 12.8 wheels/toolkit.' \
    'Overrides: QIHC_BASE, QIHC_CODE_DIR, QIHC_MODEL_ID, QIHC_MODEL_REVISION,' \
    'QIHC_MODEL_DIR, QIHC_CUDA_WHEEL=cu126|cu128, QIHC_CASES_PER_RANGE=12.'
  exit 0
fi
if [[ "${1:-}" == '--print-plan' ]]; then
  printf 'ONLINE PREPARATION\nbase=%s\ncode=%s\nenv=%s\nmodel=%s\nmodel_id=%s\nrevision=%s\n' "$BASE" "$CODE" "$ENV_PREFIX" "$MODEL_DIR" "$MODEL_ID" "$REVISION"
  printf '%s\n' 'Install verified Miniconda; conda Python/compiler/CUDA toolkit; GPU-enabled torch;' \
    'install locked dependencies; download pinned model and benchmarks;' \
    'offline model metadata check; pytest; actual tiny SFT/DPO/GRPO; CPU suite smoke; seal manifest.'
  exit 0
fi
[[ $# == 0 ]] || { echo 'Unknown argument. Use --help.' >&2; exit 2; }
[[ "$(uname -s)" == Linux && "$(uname -m)" == x86_64 ]] || { echo 'Linux x86_64 required.' >&2; exit 2; }
case "$CUDA_FLAVOR" in cu128) TOOLKIT=12.8 ;; cu126) TOOLKIT=12.6 ;; *) echo 'Use cu128 or cu126.' >&2; exit 2 ;; esac
for command in curl sha256sum flock; do command -v "$command" >/dev/null || { echo "Missing $command" >&2; exit 2; }; done
[[ -f "$CODE/experiments/run_research_suite.py" && -f "$CODE/configs/server-requirements.lock.txt" ]] || { echo 'Clone the complete updated QIHC checkout first.' >&2; exit 2; }
mkdir -p "$BASE/artifacts" "$BASE/logs" "$BASE/downloads" "$BASE/cache/conda" "$BASE/tmp" "$BASE/models" "$BASE/envs"
exec 9>"$BASE/artifacts/online-prepare.lock"
flock -n 9 || { echo 'Another online preparation is running.' >&2; exit 2; }
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)-$$"
REPORT_DIR="$BASE/artifacts/preparation-$RUN_ID"
mkdir -p "$REPORT_DIR"
if [[ -f "$BASE/artifacts/prepared.json" ]]; then
  mv -- "$BASE/artifacts/prepared.json" "$REPORT_DIR/previous_prepared.json"
fi
exec > >(tee -a "$BASE/logs/online-$RUN_ID.log") 2>&1
export CONDA_PKGS_DIRS="$BASE/cache/conda" PIP_CACHE_DIR="$BASE/cache/pip" TMPDIR="$BASE/tmp"
export HF_HOME="$BASE/cache/huggingface" HF_HUB_DOWNLOAD_TIMEOUT=600 HF_HUB_ETAG_TIMEOUT=120
export HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 HF_DATASETS_OFFLINE=0 PIP_NO_INDEX=0
export TOKENIZERS_PARALLELISM=false MPLBACKEND=Agg OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONNOUSERSITE=1 PYTHONPATH="$CODE"
# Fixed installer and published checksum from repo.anaconda.com/miniconda.
INSTALLER="$BASE/downloads/Miniconda3-py311_26.7.1-1-Linux-x86_64.sh"
INSTALLER_SHA=a6f98e6e19d5b7897ae887cd6af931eb863459f86ffd1a09cc370124cab0993e
if [[ ! -x "$CONDA_ROOT/bin/conda" ]]; then
  [[ ! -e "$CONDA_ROOT" ]] || { echo 'Incomplete miniconda3 directory exists; inspect it before retrying.' >&2; exit 2; }
  curl --fail --location --retry 5 --continue-at - --output "$INSTALLER" "https://repo.anaconda.com/miniconda/$(basename "$INSTALLER")"
  printf '%s  %s\n' "$INSTALLER_SHA" "$INSTALLER" | sha256sum --check -
  bash "$INSTALLER" -b -p "$CONDA_ROOT"
fi
# Only conda-forge and NVIDIA channels: no implicit defaults-channel ToS action.
# Third-party conda/compiler hooks can read unset variables (e.g. SYS_SYSROOT).
# Suspend nounset only across hook execution; retain errexit and pipefail.
set +u
source "$CONDA_ROOT/etc/profile.d/conda.sh"
set -u
if [[ -x "$ENV_PREFIX/bin/python" ]]; then CONDA_ACTION=install; else CONDA_ACTION=create; fi
# `conda install` may also reactivate the calling environment after its transaction.
set +u
conda "$CONDA_ACTION" --yes --prefix "$ENV_PREFIX" --override-channels -c conda-forge -c nvidia \
  python=3.11 pip setuptools wheel gcc_linux-64=12 gxx_linux-64=12 cmake "cuda-toolkit=$TOOLKIT"
set -u
set +u
conda activate "$ENV_PREFIX"
set -u
export CUDA_HOME="$ENV_PREFIX" DS_BUILD_OPS=0 MAX_JOBS="${QIHC_BUILD_JOBS:-4}"
export TORCH_EXTENSIONS_DIR="$BASE/cache/torch_extensions" TRITON_CACHE_DIR="$BASE/cache/triton"
python -m pip install --upgrade pip
python -m pip install torch==2.9.1 torchvision==0.24.1 --index-url "https://download.pytorch.org/whl/$CUDA_FLAVOR"
python -m pip install --no-build-isolation -r "$CODE/configs/server-requirements.lock.txt"
python -m pip install --no-deps --no-build-isolation -e "$CODE"
python -m pip check
python -c 'import sys,torch; assert torch.version.cuda == sys.argv[1], (torch.__version__, torch.version.cuda); print("GPU-enabled torch installed:", torch.__version__)' "$TOOLKIT"
nvcc --version
cd "$CODE"
python experiments/server_readiness.py download-model --repo "$MODEL_ID" --revision "$REVISION" \
  --model-path "$MODEL_DIR" --output "$REPORT_DIR/model_manifest.json"
DATA_DIR="$BASE/data/research_v4"
mkdir -p "$DATA_DIR"
for range in medium hard; do
  if [[ "$range" == medium ]]; then LOW=200; HIGH=399; else LOW=400; HIGH=1000; fi
  python experiments/download_hard_cvrplib.py "$DATA_DIR/CVRPLIB/X" --min-customers "$LOW" --max-customers "$HIGH" \
    --limit "${QIHC_CASES_PER_RANGE:-12}" --selection spread --verify-cached --manifest-name "${range}_download_manifest.json"
  python experiments/prepare_hard_nlcvrp.py "$DATA_DIR/CVRPLIB/X" "$DATA_DIR/${range}.jsonl" \
    --min-customers "$LOW" --max-customers "$HIGH" --limit "${QIHC_CASES_PER_RANGE:-12}" --selection spread --clauses-per-type 3
done
# From here on, checks must work without Hub access. Full 35B forward is deferred.
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
python experiments/check_research_environment.py --model-path "$MODEL_DIR" --nproc 1 --training --cpu-only --output "$REPORT_DIR/cpu_environment.json"
python -m compileall -q qihc experiments
python -m pytest -q
python experiments/check_training_integration.py --output "$REPORT_DIR/training_api"
python experiments/run_research_suite.py --profile smoke --output "$REPORT_DIR/cpu_suite_smoke"
python experiments/server_readiness.py seal --base "$BASE" --env-prefix "$ENV_PREFIX" --code-dir "$CODE" \
  --cuda-flavor "$CUDA_FLAVOR" --model-manifest "$REPORT_DIR/model_manifest.json" --cpu-report "$REPORT_DIR/cpu_environment.json" \
  --data "$DATA_DIR/medium.jsonl" "$DATA_DIR/hard.jsonl" --training-smoke "$REPORT_DIR/training_api/summary.json" \
  --suite-smoke "$REPORT_DIR/cpu_suite_smoke/research_results.json" --output "$BASE/artifacts/prepared.json"
python -m pip freeze > "$REPORT_DIR/pip-freeze.txt"
conda list --explicit > "$REPORT_DIR/conda-explicit.txt"
printf '\nONLINE PREPARATION COMPLETE. GPU/driver/NCCL/full-model checks remain for script 02.\n'
