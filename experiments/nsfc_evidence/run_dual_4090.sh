#!/usr/bin/env bash
# =============================================================================
# 联网 2×RTX 4090 — 申请书全量实验（无需 H200）
#
# 说明：
#   - 7B 单卡约 15GB，一张 4090 即可推理；本脚本用 2 卡并行缩短总时间
#   - 默认跳过 14B Case H（24GB 单卡装不下 fp16 14B）
#   - 联网自动 pip install + 下载模型/BBH
#
# 用法：
#   cd /path/to/QIHC
#   chmod +x experiments/nsfc_evidence/run_dual_4090.sh
#   bash experiments/nsfc_evidence/run_dual_4090.sh
#
# 后台：
#   nohup bash experiments/nsfc_evidence/run_dual_4090.sh \
#     > experiments/outputs/nsfc_evidence/dual_4090_launcher.log 2>&1 &
#
# 环境变量：
#   CUDA_VISIBLE_DEVICES=0,1
#   MODEL_NAME=Qwen/Qwen2.5-7B-Instruct
#   HF_HOME=$PWD/.cache/huggingface
#   SEEDS="0 1 2 3 4"
#   HF_LIMIT_PER_TASK=50          # HF 每子任务题数；赶时间可设 25
#   SKIP_SETUP=0                  # 1=跳过 pip/下载（环境已就绪时）
#   SKIP_MODEL_DOWNLOAD=0         # 1=跳过模型下载（已有 local_dir 时）
#   MODEL_DOWNLOAD_BACKEND=auto   # auto | modelscope | huggingface
#   SKIP_CASE_H=1                 # 默认 1；双 4090 仍建议跳过 14B
#   USE_CN_MIRROR=1               # 默认 1；清华 PyPI + HF 镜像
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

# shellcheck disable=SC1091
source "${SCRIPT_DIR}/cn_mirror_env.sh"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-7B-Instruct}"
export HF_HOME="${HF_HOME:-${REPO_ROOT}/.cache/huggingface}"
export SEEDS="${SEEDS:-0 1 2 3 4}"
export HF_LIMIT_PER_TASK="${HF_LIMIT_PER_TASK:-50}"
export SKIP_CASE_H="${SKIP_CASE_H:-1}"
export SKIP_SETUP="${SKIP_SETUP:-0}"
export SKIP_MODEL_DOWNLOAD="${SKIP_MODEL_DOWNLOAD:-0}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# BUNDLE_ROOT used below for model path / modelscope cache
BUNDLE_ROOT="${BUNDLE_ROOT:-${REPO_ROOT}/offline_bundle}"
export MODELSCOPE_CACHE="${MODELSCOPE_CACHE:-${BUNDLE_ROOT}/.modelscope}"
MODEL_LOCAL_DIR="${BUNDLE_ROOT}/models/Qwen2.5-7B-Instruct"

if [[ -f "${MODEL_LOCAL_DIR}/config.json" ]]; then
  MODEL="${MODEL_LOCAL_DIR}"
else
  MODEL="${MODEL_NAME}"
fi

TS="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${REPO_ROOT}/experiments/outputs/nsfc_evidence/dual_4090_${TS}"
PYTHON="${PYTHON:-python3}"

mkdir -p "${HF_HOME}" "${RUN_DIR}"

exec > >(tee -a "${RUN_DIR}/console.log") 2>&1

echo "=============================================="
echo " QIHC · 联网 2×4090 全量实验"
echo " Repo:     ${REPO_ROOT}"
echo " GPUs:     ${CUDA_VISIBLE_DEVICES}"
echo " Model:    ${MODEL}"
echo " HF_HOME:  ${HF_HOME}"
echo " Run dir:  ${RUN_DIR}"
echo " SEEDS:    ${SEEDS}"
echo " HF limit: ${HF_LIMIT_PER_TASK} per task"
echo " PyPI:     ${PIP_INDEX_URL}"
echo " HF:       ${HF_ENDPOINT}"
echo " Model dir:${MODEL_LOCAL_DIR}"
echo " DL backend: ${MODEL_DOWNLOAD_BACKEND:-auto}"
echo "=============================================="

nvidia-smi -L || nvidia-smi || true

if [[ "${SKIP_SETUP}" != "1" ]]; then
  echo "[$(date -Iseconds)] 安装依赖..."
  if [[ ! -d "${REPO_ROOT}/.venv" ]]; then
    "${PYTHON}" -m venv "${REPO_ROOT}/.venv"
  fi
  # shellcheck disable=SC1091
  source "${REPO_ROOT}/.venv/bin/activate"
  pip install -q -U pip "${PIP_INSTALL_ARGS[@]}"
  pip install -q "${PIP_INSTALL_ARGS[@]}" -e ".[dev,hf,llm]"
  pip install -q "${PIP_INSTALL_ARGS[@]}" modelscope || echo "WARN: modelscope install failed"

  if [[ "${SKIP_MODEL_DOWNLOAD}" != "1" ]] && [[ ! -f "${MODEL_LOCAL_DIR}/config.json" ]]; then
    echo "[$(date -Iseconds)] 预下载 7B → ${MODEL_LOCAL_DIR} (ModelScope 优先)..."
    mkdir -p "${MODEL_LOCAL_DIR}"
    "${PYTHON}" experiments/nsfc_evidence/download_model_hf.py \
      --repo "${MODEL_NAME}" \
      --local-dir "${MODEL_LOCAL_DIR}" \
      --backend "${MODEL_DOWNLOAD_BACKEND:-auto}"
    MODEL="${MODEL_LOCAL_DIR}"
  elif [[ -f "${MODEL_LOCAL_DIR}/config.json" ]]; then
    echo "[skip] 模型已存在: ${MODEL_LOCAL_DIR}"
    MODEL="${MODEL_LOCAL_DIR}"
  fi

  echo "[$(date -Iseconds)] 预取 BBH..."
  "${PYTHON}" experiments/download_bbh_hf.py --limit-per-task "${HF_LIMIT_PER_TASK}"

  echo "[$(date -Iseconds)] 合成 200 题缓存..."
  "${PYTHON}" -c "
from qihc.orchestrator.constrained_data import load_synthetic_tasks
load_synthetic_tasks(n_tasks=200, seed=42, regenerate=False)
print('synthetic ok')
"

  pytest -q tests/test_vci.py tests/test_bbh.py || echo "WARN: pytest 部分失败"
else
  # shellcheck disable=SC1091
  source "${REPO_ROOT}/.venv/bin/activate"
fi

run_gpu() {
  local gpu="$1"
  local name="$2"
  shift 2
  local log="${RUN_DIR}/${name}.log"
  echo "[launch GPU${gpu}] ${name}"
  (
    export CUDA_VISIBLE_DEVICES="${gpu}"
    "$@" > "${log}" 2>&1
    echo "[done GPU${gpu}] ${name}"
  ) &
}

echo "[$(date -Iseconds)] === Wave 1：bundled 主表 + handoff ==="
run_gpu 0 "w1_unified_bundled_7b" \
  "${PYTHON}" experiments/nsfc_evidence/run_unified_ablation.py \
  --dataset bundled --use-llm --model-name "${MODEL}" \
  --budget-steps 400 --n-samples 50 --seeds ${SEEDS} \
  --output-dir "${RUN_DIR}/unified_bundled_7b"

run_gpu 1 "w1_handoff_7b" \
  "${PYTHON}" experiments/run_vci_handoff.py \
  --source bundled --logits llm --model-name "${MODEL}" \
  --limit 40 --budget-steps 400 --noise-scales-dense \
  --seeds 0 1 2 --include-cr-baselines \
  --output-dir "${RUN_DIR}/handoff_7b_dense"
wait

echo "[$(date -Iseconds)] === Wave 2：HF CR + Case G ==="
run_gpu 0 "w2_cr_hf_bbh" \
  "${PYTHON}" experiments/nsfc_evidence/run_cr_bbh.py \
  --source hf --use-llm --model-name "${MODEL}" \
  --n-samples 50 --limit-per-task "${HF_LIMIT_PER_TASK}" --budget-steps 400 \
  --output-dir "${RUN_DIR}/cr_hf_bbh"

run_gpu 1 "w2_case_g" \
  "${PYTHON}" experiments/nsfc_evidence/run_case_g_constrained_bbh.py \
  --use-llm --model-name "${MODEL}" \
  --limit-per-task "${HF_LIMIT_PER_TASK}" --n-samples 50 --budget-steps 400 \
  --output-dir "${RUN_DIR}/case_g"
wait

echo "[$(date -Iseconds)] === Wave 3：Pareto + synthetic 主表 ==="
run_gpu 0 "w3_pareto_7b" \
  "${PYTHON}" experiments/run_vci_pareto.py \
  --budgets 100 200 300 400 600 --include-cr \
  --use-llm --model-name "${MODEL}" \
  --output-dir "${RUN_DIR}/pareto_bundled_7b"

run_gpu 1 "w3_unified_synthetic" \
  "${PYTHON}" experiments/nsfc_evidence/run_unified_ablation.py \
  --dataset synthetic --n-tasks 200 \
  --budget-steps 400 --n-samples 50 --seeds ${SEEDS} \
  --output-dir "${RUN_DIR}/unified_synthetic"
wait

echo "[$(date -Iseconds)] === Wave 4：机制实验（CPU/轻量，串行）==="
"${PYTHON}" experiments/nsfc_evidence/run_case_f_vci_ablation.py \
  --n-tasks 200 --n-samples 50 --budget-steps 400 \
  --output-dir "${RUN_DIR}/case_f" \
  2>&1 | tee "${RUN_DIR}/case_f.log"

"${PYTHON}" experiments/nsfc_evidence/run_case_b_dual_synthetic.py \
  --n-tasks 200 --n-samples 50 --budget-steps 400 \
  --output-dir "${RUN_DIR}/case_b" \
  2>&1 | tee "${RUN_DIR}/case_b.log"

"${PYTHON}" experiments/nsfc_evidence/run_sampler_ablation.py \
  --steps 400 --output-dir "${RUN_DIR}/sampler_ablation" \
  2>&1 | tee "${RUN_DIR}/sampler_ablation.log"

"${PYTHON}" experiments/nsfc_evidence/run_f_trajectories.py \
  --budget-steps 400 --output-dir "${RUN_DIR}/f_trajectories" \
  2>&1 | tee "${RUN_DIR}/f_trajectories.log"

"${PYTHON}" experiments/run_sampler_scaling.py \
  --nodes 50 100 200 500 --trials 5 --measure-tts \
  --output-dir "${RUN_DIR}/scaling_tts" \
  2>&1 | tee "${RUN_DIR}/scaling.log"

if [[ "${SKIP_CASE_H}" != "1" ]]; then
  echo "[$(date -Iseconds)] Case H 14B（单卡，可能 OOM，默认已跳过）..."
  CUDA_VISIBLE_DEVICES=0 "${PYTHON}" experiments/nsfc_evidence/run_case_h_model_compare.py \
    --n-tasks 200 --model-7b "${MODEL}" \
    --model-14b "${MODEL_NAME_14B:-Qwen/Qwen2.5-14B-Instruct}" \
    --budget-steps 400 --output-dir "${RUN_DIR}/case_h" \
    2>&1 | tee "${RUN_DIR}/case_h.log" || echo "WARN: Case H 失败（4090 可忽略）"
fi

echo "[$(date -Iseconds)] 汇总同算力表..."
"${PYTHON}" experiments/nsfc_evidence/run_case_c_compute_budget.py \
  --run-dir "${RUN_DIR}" --output-dir "${RUN_DIR}/compute_budget" || true

ARCHIVE="${RUN_DIR}/dual_4090_results.tar.gz"
tar -czf "${ARCHIVE}" -C "${RUN_DIR}" .

echo "=============================================="
echo " 2×4090 全量实验完成"
echo " 结果: ${RUN_DIR}"
echo " 打包: ${ARCHIVE}"
echo "----------------------------------------------"
echo " 主表: ${RUN_DIR}/unified_bundled_7b/unified_ablation.json"
echo " handoff: ${RUN_DIR}/handoff_7b_dense/handoff.json"
echo " HF CR: ${RUN_DIR}/cr_hf_bbh/cr_protocol.json"
echo "=============================================="
