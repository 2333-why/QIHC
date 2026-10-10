#!/usr/bin/env bash
# Run inside a scheduler-allocated OFFLINE single-node Linux GPU job.
set -Eeuo pipefail
umask 077
trap 'printf "Offline run failed at line %s. Inspect the run logs.\n" "$LINENO" >&2' ERR
BASE="${QIHC_BASE:-/inspire/hdd/global_user/czxs25210022/why}"
CODE="${QIHC_CODE_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)}"
CUDA_FLAVOR="${QIHC_CUDA_WHEEL:-cu128}"
ENV_PREFIX="$BASE/envs/qihc-$CUDA_FLAVOR"
PROFILE="${QIHC_PROFILE:-formal}"
NPROC="${QIHC_NPROC:-2}"
GRPO_BATCH="${QIHC_GRPO_GENERATION_BATCH_SIZE:-4}"
CHECK_ONLY=0
if [[ "${1:-}" == '--help' ]]; then
  printf '%s\n' 'Usage: bash scripts/server/02_run_offline_gpu.sh [--preflight-only|--print-plan]' \
    'Defaults: Qwen3.5-9B prepared by script 01, 2 GPUs, formal suite, GRPO generation batch 4.' \
    'Run inside an allocated GPU job; respects CUDA_VISIBLE_DEVICES. Single-node multi-GPU.' \
    'Overrides: QIHC_BASE, QIHC_CODE_DIR, QIHC_CUDA_WHEEL, QIHC_NPROC,' \
    'QIHC_PROFILE=pilot|formal, QIHC_OUTPUT, QIHC_TRAINING_STEPS=200, QIHC_GRPO_GENERATION_BATCH_SIZE=4.'
  exit 0
fi
if [[ "${1:-}" == '--print-plan' ]]; then
  printf 'OFFLINE GPU RUN\nbase=%s\ncode=%s\nenv=%s\nprofile=%s\nnproc=%s\ngrpo_generation_batch=%s\n' "$BASE" "$CODE" "$ENV_PREFIX" "$PROFILE" "$NPROC" "$GRPO_BATCH"
  printf '%s\n' 'Verify prepared model/data/source/environment; CUDA and BF16 check; torchrun NCCL + ZeRO-3 smoke;' \
    'launch run_research_suite.py with LOCAL --data and --deepspeed-config;' \
    'full-model generation gate; evaluations; teacher data; SFT/DPO/GRPO; post-training evaluations.'
  exit 0
fi
if [[ "${1:-}" == '--preflight-only' ]]; then CHECK_ONLY=1; shift; fi
[[ $# == 0 ]] || { echo 'Unknown argument. Use --help.' >&2; exit 2; }
[[ "$(uname -s)" == Linux && "$(uname -m)" == x86_64 ]] || { echo 'Linux x86_64 required.' >&2; exit 2; }
[[ "$PROFILE" == pilot || "$PROFILE" == formal ]] || { echo 'Use pilot or formal.' >&2; exit 2; }
[[ -f "$BASE/artifacts/prepared.json" && -x "$ENV_PREFIX/bin/python" ]] || { echo 'Run script 01 successfully first with the same QIHC_CUDA_WHEEL.' >&2; exit 2; }
source "$BASE/miniconda3/etc/profile.d/conda.sh"
conda activate "$ENV_PREFIX"
export CUDA_HOME="$ENV_PREFIX" CUDA_DEVICE_ORDER=PCI_BUS_ID
export HF_HOME="$BASE/cache/huggingface" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export WANDB_MODE=disabled PIP_NO_INDEX=1 TOKENIZERS_PARALLELISM=false MPLBACKEND=Agg
export PYTHONNOUSERSITE=1 PYTHONPATH="$CODE"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}" OPENBLAS_NUM_THREADS=1
export TORCH_EXTENSIONS_DIR="$BASE/cache/torch_extensions" TRITON_CACHE_DIR="$BASE/cache/triton"
export TMPDIR="$BASE/tmp" MAX_JOBS="${QIHC_BUILD_JOBS:-4}" NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
# Do not override scheduler networking, CUDA_VISIBLE_DEVICES, NCCL socket/IB selection.
[[ "${SLURM_NNODES:-1}" == 1 ]] || { echo 'This launcher is single-node multi-GPU, not a multi-node launcher.' >&2; exit 2; }
for name in RANK LOCAL_RANK WORLD_SIZE; do
  [[ -z "${!name:-}" ]] || { echo 'Run script 02 once, not inside torchrun/srun-per-GPU.' >&2; exit 2; }
done
cd "$CODE"
[[ "$NPROC" =~ ^[1-9][0-9]*$ ]] || { echo 'No visible GPUs or invalid QIHC_NPROC.' >&2; exit 2; }
python -c 'import sys; from experiments.train_s2e_lora import resolve_generation_batch; print("GRPO generation batch:", resolve_generation_batch(int(sys.argv[1]), int(sys.argv[2]), 4, 8))' "$GRPO_BATCH" "$NPROC"
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)-$$"
CHECK_DIR="$BASE/artifacts/gpu-check-$RUN_ID"
mkdir -p "$CHECK_DIR" "$BASE/logs" "$TMPDIR"
exec > >(tee -a "$BASE/logs/offline-$RUN_ID.log") 2>&1
python experiments/server_readiness.py verify --manifest "$BASE/artifacts/prepared.json" --code-dir "$CODE" --env-prefix "$ENV_PREFIX"
MODEL_DIR="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["model_path"])' "$BASE/artifacts/prepared.json")"
MEDIUM="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["data"][0])' "$BASE/artifacts/prepared.json")"
HARD="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["data"][1])' "$BASE/artifacts/prepared.json")"
nvidia-smi
python -m pip check
python experiments/check_research_environment.py --model-path "$MODEL_DIR" --nproc "$NPROC" --training --output "$CHECK_DIR/environment.json"
python -m torch.distributed.run --standalone --nproc_per_node "$NPROC" experiments/server_readiness.py distributed-smoke \
  --deepspeed-config "$CODE/configs/deepspeed-zero3-bf16.json" --output "$CHECK_DIR/distributed.json"
if [[ "$CHECK_ONLY" == 1 ]]; then
  # Full-model inference is necessary even in preflight-only mode.
  python experiments/server_readiness.py model-smoke --model-path "$MODEL_DIR" --output "$CHECK_DIR/model.json"
  printf '\nGPU PREFLIGHT PASSED. No formal experiment/training started.\n'
  exit 0
fi
OUTPUT="${QIHC_OUTPUT:-$BASE/results/qihc_v4_${PROFILE}_${RUN_ID}}"
python experiments/run_research_suite.py --profile "$PROFILE" --model-path "$MODEL_DIR" --nproc "$NPROC" \
  --data "$MEDIUM" "$HARD" --benchmark-dir "$BASE/data/research_v4/CVRPLIB/X" \
  --deepspeed-config "$CODE/configs/deepspeed-zero3-bf16.json" \
  --grpo-generation-batch-size "$GRPO_BATCH" \
  --training-steps "${QIHC_TRAINING_STEPS:-200}" --output "$OUTPUT"
printf '\nOFFLINE PIPELINE COMPLETE: %s\n' "$OUTPUT"
