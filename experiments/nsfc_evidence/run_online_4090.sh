#!/usr/bin/env bash
# =============================================================================
# 联网 4090 工作站 — 下载离线包 + 7B 轻量验证实验
#
# 用途：
#   1. 安装依赖、下载模型与 BBH 数据集
#   2. 打包 offline_bundle/ 供不联网 H200 机器使用
#   3. 在 4090 上跑申请书 P0 轻量实验（7B，题量适中）
#
# 用法：
#   cd /path/to/QIHC
#   chmod +x experiments/nsfc_evidence/run_online_4090.sh
#   bash experiments/nsfc_evidence/run_online_4090.sh
#
# 可选环境变量：
#   CUDA_VISIBLE_DEVICES=0          # 单卡 4090
#   HF_HOME=$PWD/offline_bundle/hf_home
#   BUNDLE_ROOT=$PWD/offline_bundle # 离线传输目录
#   MODEL_HUB_7B=Qwen/Qwen2.5-7B-Instruct
#   MODEL_HUB_14B=Qwen/Qwen2.5-14B-Instruct
#   DOWNLOAD_14B=1                  # 是否一并下载 14B（约 30GB）
#   SKIP_EXPERIMENTS=0              # 设为 1 则只下载打包，不跑实验
#   SKIP_WHEELS=0                   # 设为 1 跳过 pip wheel 打包
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export HF_HOME="${HF_HOME:-${REPO_ROOT}/offline_bundle/hf_home}"
export BUNDLE_ROOT="${BUNDLE_ROOT:-${REPO_ROOT}/offline_bundle}"
export MODEL_HUB_7B="${MODEL_HUB_7B:-Qwen/Qwen2.5-7B-Instruct}"
export MODEL_HUB_14B="${MODEL_HUB_14B:-Qwen/Qwen2.5-14B-Instruct}"
export DOWNLOAD_14B="${DOWNLOAD_14B:-1}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

MODEL_LOCAL_7B="${BUNDLE_ROOT}/models/Qwen2.5-7B-Instruct"
MODEL_LOCAL_14B="${BUNDLE_ROOT}/models/Qwen2.5-14B-Instruct"
WHEELS_DIR="${BUNDLE_ROOT}/wheels"
MANIFEST="${BUNDLE_ROOT}/manifest.env"
TS="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${REPO_ROOT}/experiments/outputs/nsfc_evidence/online_4090_${TS}"
PYTHON="${PYTHON:-python3}"

mkdir -p "${HF_HOME}" "${BUNDLE_ROOT}/models" "${WHEELS_DIR}" "${RUN_DIR}"

exec > >(tee -a "${RUN_DIR}/console.log") 2>&1

echo "=============================================="
echo " QIHC · 联网 4090 流水线"
echo " Repo:         ${REPO_ROOT}"
echo " GPU:          ${CUDA_VISIBLE_DEVICES}"
echo " HF_HOME:      ${HF_HOME}"
echo " BUNDLE_ROOT:  ${BUNDLE_ROOT}"
echo " Run dir:      ${RUN_DIR}"
echo "=============================================="

nvidia-smi || true

echo "[$(date -Iseconds)] 创建 venv..."
if [[ ! -d "${REPO_ROOT}/.venv" ]]; then
  "${PYTHON}" -m venv "${REPO_ROOT}/.venv"
fi
# shellcheck disable=SC1091
source "${REPO_ROOT}/.venv/bin/activate"

echo "[$(date -Iseconds)] 安装依赖（需联网）..."
pip install -q -U pip
pip install -q -e ".[dev,hf,llm]"
pip install -q huggingface_hub

download_model() {
  local hub_id="$1"
  local local_dir="$2"
  if [[ -f "${local_dir}/config.json" ]]; then
    echo "[skip] 已存在: ${local_dir}"
    return 0
  fi
  echo "[download] ${hub_id} -> ${local_dir}"
  if command -v huggingface-cli >/dev/null 2>&1; then
    huggingface-cli download "${hub_id}" --local-dir "${local_dir}"
  else
    "${PYTHON}" - <<PY
from huggingface_hub import snapshot_download
snapshot_download(repo_id="${hub_id}", local_dir="${local_dir}")
PY
  fi
}

echo "[$(date -Iseconds)] 下载 7B 模型..."
download_model "${MODEL_HUB_7B}" "${MODEL_LOCAL_7B}"

if [[ "${DOWNLOAD_14B}" == "1" ]]; then
  echo "[$(date -Iseconds)] 下载 14B 模型（供 H200 离线 Case H）..."
  download_model "${MODEL_HUB_14B}" "${MODEL_LOCAL_14B}"
else
  echo "[skip] DOWNLOAD_14B=0，不下载 14B"
fi

echo "[$(date -Iseconds)] 预取 BBH 数据集到 HF_HOME..."
"${PYTHON}" experiments/download_bbh_hf.py --limit-per-task 50

if [[ "${SKIP_WHEELS:-0}" != "1" ]]; then
  echo "[$(date -Iseconds)] 打包 pip wheels（供 H200 完全离线安装）..."
  pip download -q -d "${WHEELS_DIR}" -e ".[dev,hf,llm]" || echo "WARN: pip download 部分失败，H200 可改用已有 venv"
fi

echo "[$(date -Iseconds)] 生成合成 200 题缓存..."
"${PYTHON}" -c "
from qihc.orchestrator.constrained_data import load_synthetic_tasks
load_synthetic_tasks(n_tasks=200, seed=42, regenerate=False)
print('synthetic ok')
"

echo "[$(date -Iseconds)] 写入离线 manifest: ${MANIFEST}"
cat > "${MANIFEST}" <<EOF
# QIHC offline bundle — 由 run_online_4090.sh 生成
# 在 H200 上: source offline_bundle/manifest.env
export QIHC_REPO_ROOT="${REPO_ROOT}"
export HF_HOME="${HF_HOME}"
export BUNDLE_ROOT="${BUNDLE_ROOT}"
export MODEL_LOCAL_7B="${MODEL_LOCAL_7B}"
export MODEL_LOCAL_14B="${MODEL_LOCAL_14B}"
export WHEELS_DIR="${WHEELS_DIR}"
export MODEL_HUB_7B="${MODEL_HUB_7B}"
export MODEL_HUB_14B="${MODEL_HUB_14B}"
EOF

echo "[$(date -Iseconds)] pytest 冒烟..."
pytest -q tests/test_vci.py tests/test_bbh.py || echo "WARN: pytest 有失败项"

if [[ "${SKIP_EXPERIMENTS:-0}" == "1" ]]; then
  echo "[skip] SKIP_EXPERIMENTS=1，跳过 4090 实验"
else
  echo "[$(date -Iseconds)] === 4090 轻量实验（7B，申请书 P0）==="

  # 实验 1：mock 流水线自检（CPU/GPU 均可）
  "${PYTHON}" experiments/nsfc_evidence/run_unified_ablation.py \
    --dataset bundled --budget-steps 400 --n-samples 50 --seeds 0 \
    --output-dir "${RUN_DIR}/smoke_bundled_mock"

  # 实验 2：bundled + 真实 7B logits（主表，3 seeds）
  "${PYTHON}" experiments/nsfc_evidence/run_unified_ablation.py \
    --dataset bundled --use-llm \
    --model-name "${MODEL_LOCAL_7B}" \
    --budget-steps 400 --n-samples 50 --seeds 0 1 2 \
    --output-dir "${RUN_DIR}/unified_bundled_7b"

  # 实验 3：handoff（7B 真实语义）
  "${PYTHON}" experiments/run_vci_handoff.py \
    --source bundled --logits llm \
    --model-name "${MODEL_LOCAL_7B}" \
    --limit 40 --budget-steps 400 --noise-scales-dense \
    --seeds 0 1 2 --include-cr-baselines \
    --output-dir "${RUN_DIR}/handoff_7b"

  # 实验 4：HF CR 子集（先 25 题/任务，4090 可承受）
  "${PYTHON}" experiments/nsfc_evidence/run_cr_bbh.py \
    --source hf --use-llm \
    --model-name "${MODEL_LOCAL_7B}" \
    --n-samples 50 --limit-per-task 25 --budget-steps 400 \
    --output-dir "${RUN_DIR}/cr_hf_bbh_25"

  # 实验 5：Pareto（mock logits，快）
  "${PYTHON}" experiments/run_vci_pareto.py \
    --budgets 100 200 300 400 600 --include-cr \
    --output-dir "${RUN_DIR}/pareto_mock"

  "${PYTHON}" experiments/nsfc_evidence/run_case_c_compute_budget.py \
    --run-dir "${RUN_DIR}" --output-dir "${RUN_DIR}/compute_budget" || true
fi

ARCHIVE="${BUNDLE_ROOT}/qihc_offline_bundle_${TS}.tar.gz"
echo "[$(date -Iseconds)] 打包离线传输包（不含整个 repo，仅 bundle）..."
tar -czf "${ARCHIVE}" \
  -C "${BUNDLE_ROOT}" \
  manifest.env models hf_home wheels 2>/dev/null \
  || tar -czf "${ARCHIVE}" -C "${BUNDLE_ROOT}" manifest.env models hf_home

RESULTS_ARCHIVE="${RUN_DIR}/online_4090_results.tar.gz"
tar -czf "${RESULTS_ARCHIVE}" -C "${RUN_DIR}" . || true

echo "=============================================="
echo " 联网 4090 流水线完成"
echo "----------------------------------------------"
echo " 离线包目录:   ${BUNDLE_ROOT}"
echo " 离线 manifest: ${MANIFEST}"
echo " 传输压缩包:   ${ARCHIVE}"
echo " 4090 实验结果: ${RUN_DIR}"
echo " 结果压缩包:   ${RESULTS_ARCHIVE}"
echo "----------------------------------------------"
echo " 下一步（H200 不联网机器）："
echo "   1) 将整个 QIHC/ 代码目录 + offline_bundle/ 拷到 H200"
echo "   2) cd QIHC && bash experiments/nsfc_evidence/run_offline_h200.sh"
echo "=============================================="
