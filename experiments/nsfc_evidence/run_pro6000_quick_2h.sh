#!/usr/bin/env bash
# =============================================================================
# 单卡 Pro 6000 · 约 2 小时快速实验（冒烟 + 7B 核心表）
#
# 适用：只有 1 张空闲 GPU（默认 GPU 0），先验证流水线再跑全量。
#
# 用法：
#   cd /hdd/why/QIHC
#   bash experiments/nsfc_evidence/run_pro6000_quick_2h.sh
#
# 后台：
#   nohup bash experiments/nsfc_evidence/run_pro6000_quick_2h.sh \
#     > experiments/outputs/nsfc_evidence/pro6000_quick_launcher.log 2>&1 &
#
# 环境变量（可缩短/加长）：
#   CUDA_VISIBLE_DEVICES=0        # 空闲的那张卡
#   QUICK_N_TASKS=10              # bundled 题数（默认 10）
#   QUICK_N_SAMPLES=8             # CR 采样次数（默认 8，全量是 50）
#   QUICK_BUDGET=300              # p-bit 步数
#   QUICK_HANDOFF_LIMIT=10        # handoff 题数
#   SKIP_SETUP=1                  # 环境/模型已就绪
#   SKIP_MODEL_DOWNLOAD=1
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

# shellcheck disable=SC1091
source "${SCRIPT_DIR}/cn_mirror_env.sh"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-7B-Instruct}"
export HF_HOME="${HF_HOME:-${REPO_ROOT}/.cache/huggingface}"
export SKIP_SETUP="${SKIP_SETUP:-0}"
export SKIP_MODEL_DOWNLOAD="${SKIP_MODEL_DOWNLOAD:-0}"

export QUICK_N_TASKS="${QUICK_N_TASKS:-10}"
export QUICK_N_SAMPLES="${QUICK_N_SAMPLES:-8}"
export QUICK_BUDGET="${QUICK_BUDGET:-300}"
export QUICK_HANDOFF_LIMIT="${QUICK_HANDOFF_LIMIT:-10}"
export QUICK_SYNTH_N="${QUICK_SYNTH_N:-50}"

BUNDLE_ROOT="${BUNDLE_ROOT:-${REPO_ROOT}/offline_bundle}"
MODEL_LOCAL="${BUNDLE_ROOT}/models/Qwen2.5-7B-Instruct"
if [[ -f "${MODEL_LOCAL}/config.json" ]]; then
  MODEL="${MODEL_LOCAL}"
else
  MODEL="${MODEL_NAME}"
fi

TS="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${REPO_ROOT}/experiments/outputs/nsfc_evidence/pro6000_quick_${TS}"
PYTHON="${PYTHON:-python3}"

mkdir -p "${RUN_DIR}"
exec > >(tee -a "${RUN_DIR}/console.log") 2>&1

echo "=============================================="
echo " QIHC · Pro 6000 单卡快速实验 (~2h)"
echo " Repo:      ${REPO_ROOT}"
echo " GPU:       ${CUDA_VISIBLE_DEVICES}"
echo " Model:     ${MODEL}"
echo " Tasks:     ${QUICK_N_TASKS} bundled + handoff ${QUICK_HANDOFF_LIMIT}"
echo " Samples:   ${QUICK_N_SAMPLES} (full run uses 50)"
echo " Run dir:   ${RUN_DIR}"
echo "=============================================="
nvidia-smi -L || nvidia-smi || true

if [[ "${SKIP_SETUP}" != "1" ]]; then
  [[ -d "${REPO_ROOT}/.venv" ]] || "${PYTHON}" -m venv "${REPO_ROOT}/.venv"
  # shellcheck disable=SC1091
  source "${REPO_ROOT}/.venv/bin/activate"
  pip install -q -U pip "${PIP_INSTALL_ARGS[@]}"
  pip install -q "${PIP_INSTALL_ARGS[@]}" -e ".[dev,hf,llm]"
  pip install -q "${PIP_INSTALL_ARGS[@]}" modelscope || true
  if [[ "${SKIP_MODEL_DOWNLOAD}" != "1" ]] && [[ ! -f "${MODEL_LOCAL}/config.json" ]]; then
    mkdir -p "${MODEL_LOCAL}"
    "${PYTHON}" experiments/nsfc_evidence/download_model_hf.py \
      --repo "${MODEL_NAME}" --local-dir "${MODEL_LOCAL}" --backend modelscope
    MODEL="${MODEL_LOCAL}"
  fi
else
  # shellcheck disable=SC1091
  source "${REPO_ROOT}/.venv/bin/activate"
fi

python -c "import torch; print('CUDA:', torch.cuda.is_available(), 'nGPU:', torch.cuda.device_count())" || true

echo ""
echo "[$(date -Iseconds)] === A. Mock 流水线自检 (~3 min) ==="
"${PYTHON}" experiments/nsfc_evidence/run_unified_ablation.py \
  --dataset bundled --budget-steps "${QUICK_BUDGET}" --n-samples "${QUICK_N_SAMPLES}" \
  --seeds 0 --output-dir "${RUN_DIR}/A_mock_bundled"

echo ""
echo "[$(date -Iseconds)] === B. 7B 主表：bundled ${QUICK_N_TASKS} 题 × 5 模式 (~60–90 min) ==="
"${PYTHON}" experiments/nsfc_evidence/run_cr_bbh.py \
  --source bundled --use-llm --model-name "${MODEL}" \
  --limit "${QUICK_N_TASKS}" --n-samples "${QUICK_N_SAMPLES}" \
  --budget-steps "${QUICK_BUDGET}" --seed 0 \
  --output-dir "${RUN_DIR}/B_cr_bundled_7b"

echo ""
echo "[$(date -Iseconds)] === C. Handoff ${QUICK_HANDOFF_LIMIT} 题 (~20–40 min) ==="
"${PYTHON}" experiments/run_vci_handoff.py \
  --source bundled --logits llm --model-name "${MODEL}" \
  --limit "${QUICK_HANDOFF_LIMIT}" --budget-steps "${QUICK_BUDGET}" \
  --seeds 0 --noise-scales-dense --include-cr-baselines \
  --output-dir "${RUN_DIR}/C_handoff_7b"

echo ""
echo "[$(date -Iseconds)] === D. 机制快测 (~10 min, mostly CPU) ==="
"${PYTHON}" experiments/nsfc_evidence/run_case_f_vci_ablation.py \
  --n-tasks "${QUICK_SYNTH_N}" --n-samples "${QUICK_N_SAMPLES}" \
  --budget-steps "${QUICK_BUDGET}" --output-dir "${RUN_DIR}/D_case_f"

"${PYTHON}" experiments/nsfc_evidence/run_sampler_ablation.py \
  --steps "${QUICK_BUDGET}" --output-dir "${RUN_DIR}/D_sampler"

"${PYTHON}" experiments/nsfc_evidence/run_f_trajectories.py \
  --budget-steps "${QUICK_BUDGET}" --output-dir "${RUN_DIR}/D_f_traj"

"${PYTHON}" experiments/nsfc_evidence/run_case_c_compute_budget.py \
  --run-dir "${RUN_DIR}" --output-dir "${RUN_DIR}/compute_budget" || true

ARCHIVE="${RUN_DIR}/pro6000_quick_results.tar.gz"
tar -czf "${ARCHIVE}" -C "${RUN_DIR}" .

echo "=============================================="
echo " 快速实验完成（单卡 Pro 6000）"
echo " 结果: ${RUN_DIR}"
echo " 打包: ${ARCHIVE}"
echo "----------------------------------------------"
echo " 主表: ${RUN_DIR}/B_cr_bundled_7b/cr_protocol.json"
echo " handoff: ${RUN_DIR}/C_handoff_7b/handoff.json"
echo " mock: ${RUN_DIR}/A_mock_bundled/unified_ablation.json"
echo "=============================================="
