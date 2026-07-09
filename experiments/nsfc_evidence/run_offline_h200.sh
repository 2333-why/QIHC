#!/usr/bin/env bash
# =============================================================================
# 不联网 4×H200 — 批量实验（读取 offline_bundle/manifest.env）
#
# 前置条件：
#   已在联网 4090 上运行 run_online_4090.sh，并将以下内容拷到本机：
#     - 整个 QIHC/ 代码仓库
#     - offline_bundle/（含 models/, hf_home/, manifest.env）
#
# 用法：
#   cd /path/to/QIHC
#   chmod +x experiments/nsfc_evidence/run_offline_h200.sh
#   bash experiments/nsfc_evidence/run_offline_h200.sh
#
# 后台运行：
#   nohup bash experiments/nsfc_evidence/run_offline_h200.sh \
#     > experiments/outputs/nsfc_evidence/offline_h200_launcher.log 2>&1 &
#
# 可选环境变量：
#   BUNDLE_ROOT=$PWD/offline_bundle     # manifest.env 所在目录
#   CUDA_VISIBLE_DEVICES=0,1,2,3
#   SKIP_CASE_H=0                       # 1=跳过 14B 对比（未下载 14B 时）
#   HF_LIMIT_PER_TASK=50                # HF CR 每任务题数
#   SEEDS="0 1 2 3 4"
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

# shellcheck disable=SC1091
source "${SCRIPT_DIR}/cn_mirror_env.sh"

export BUNDLE_ROOT="${BUNDLE_ROOT:-${REPO_ROOT}/offline_bundle}"
MANIFEST="${BUNDLE_ROOT}/manifest.env"

if [[ ! -f "${MANIFEST}" ]]; then
  echo "ERROR: 找不到 ${MANIFEST}"
  echo "请先在联网 4090 运行: bash experiments/nsfc_evidence/run_online_4090.sh"
  exit 1
fi

# shellcheck disable=SC1090
source "${MANIFEST}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export HF_HOME="${HF_HOME:-${BUNDLE_ROOT}/hf_home}"
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

MODEL_7B="${MODEL_LOCAL_7B}"
MODEL_14B="${MODEL_LOCAL_14B}"
SEEDS="${SEEDS:-0 1 2 3 4}"
HF_LIMIT="${HF_LIMIT_PER_TASK:-50}"
SKIP_CASE_H="${SKIP_CASE_H:-0}"
TS="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${REPO_ROOT}/experiments/outputs/nsfc_evidence/offline_h200_${TS}"
PYTHON="${PYTHON:-python3}"
WHEELS_DIR="${WHEELS_DIR:-${BUNDLE_ROOT}/wheels}"

mkdir -p "${RUN_DIR}"

if [[ ! -f "${MODEL_7B}/config.json" ]]; then
  echo "ERROR: 7B 模型不在 ${MODEL_7B}"
  echo "请确认 offline_bundle/models/ 已完整拷贝"
  exit 1
fi

exec > >(tee -a "${RUN_DIR}/console.log") 2>&1

echo "=============================================="
echo " QIHC · 离线 4×H200 批量实验"
echo " Repo:       ${REPO_ROOT}"
echo " GPUs:       ${CUDA_VISIBLE_DEVICES}"
echo " HF_HOME:    ${HF_HOME}"
echo " Model 7B:   ${MODEL_7B}"
echo " Model 14B:  ${MODEL_14B}"
echo " Run dir:    ${RUN_DIR}"
echo " OFFLINE:    TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1"
echo "=============================================="

nvidia-smi || true

echo "[$(date -Iseconds)] 创建/激活 venv..."
if [[ ! -d "${REPO_ROOT}/.venv" ]]; then
  "${PYTHON}" -m venv "${REPO_ROOT}/.venv"
fi
# shellcheck disable=SC1091
source "${REPO_ROOT}/.venv/bin/activate"

if [[ -d "${WHEELS_DIR}" ]] && ls "${WHEELS_DIR}"/*.whl >/dev/null 2>&1; then
  echo "[$(date -Iseconds)] 离线安装依赖（wheels）..."
  pip install -q -U pip "${PIP_INSTALL_ARGS[@]}"
  pip install -q --no-index --find-links="${WHEELS_DIR}" -e ".[dev,hf,llm]" || {
    echo "WARN: 离线 pip 失败，尝试清华源在线安装..."
    pip install -q "${PIP_INSTALL_ARGS[@]}" -e ".[dev,hf,llm]" 2>/dev/null || true
  }
else
  echo "[$(date -Iseconds)] 未找到 wheels，使用清华源在线安装..."
  pip install -q "${PIP_INSTALL_ARGS[@]}" -e ".[dev,hf,llm]" 2>/dev/null || true
fi

echo "[$(date -Iseconds)] 检查 BBH 缓存..."
"${PYTHON}" experiments/download_bbh_hf.py --limit-per-task "${HF_LIMIT}" || {
  echo "ERROR: BBH 缓存不可用，请从 4090 拷贝 offline_bundle/hf_home"
  exit 1
}

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
  echo $! > "${RUN_DIR}/${name}.pid"
}

echo "[$(date -Iseconds)] === Wave 1：4 卡并行（核心 GPU 实验）==="

# GPU 0: bundled 主表 7B 五模式 × 5 seeds
run_gpu 0 "w1_unified_bundled_7b" \
  "${PYTHON}" experiments/nsfc_evidence/run_unified_ablation.py \
  --dataset bundled --use-llm --model-name "${MODEL_7B}" \
  --budget-steps 400 --n-samples 50 --seeds ${SEEDS} \
  --output-dir "${RUN_DIR}/unified_bundled_7b"

# GPU 1: handoff 密集 σ 扫描
run_gpu 1 "w1_handoff_7b" \
  "${PYTHON}" experiments/run_vci_handoff.py \
  --source bundled --logits llm --model-name "${MODEL_7B}" \
  --limit 40 --budget-steps 400 --noise-scales-dense \
  --seeds 0 1 2 --include-cr-baselines \
  --output-dir "${RUN_DIR}/handoff_7b_dense"

# GPU 2: HF CR 全量子集（limit-per-task 可配）
run_gpu 2 "w1_cr_hf_bbh" \
  "${PYTHON}" experiments/nsfc_evidence/run_cr_bbh.py \
  --source hf --use-llm --model-name "${MODEL_7B}" \
  --n-samples 50 --limit-per-task "${HF_LIMIT}" --budget-steps 400 \
  --output-dir "${RUN_DIR}/cr_hf_bbh"

# GPU 3: Case G 约束版 BBH
run_gpu 3 "w1_case_g" \
  "${PYTHON}" experiments/nsfc_evidence/run_case_g_constrained_bbh.py \
  --use-llm --model-name "${MODEL_7B}" \
  --limit-per-task "${HF_LIMIT}" --n-samples 50 --budget-steps 400 \
  --output-dir "${RUN_DIR}/case_g"

wait
echo "[$(date -Iseconds)] Wave 1 完成"

echo "[$(date -Iseconds)] === Wave 2：4 卡并行（扩展 + 机制）==="

run_gpu 0 "w2_pareto_7b" \
  "${PYTHON}" experiments/run_vci_pareto.py \
  --budgets 100 200 300 400 600 --include-cr \
  --use-llm --model-name "${MODEL_7B}" \
  --output-dir "${RUN_DIR}/pareto_bundled_7b"

run_gpu 1 "w2_case_f_ablation" \
  "${PYTHON}" experiments/nsfc_evidence/run_case_f_vci_ablation.py \
  --n-tasks 200 --n-samples 50 --budget-steps 400 \
  --output-dir "${RUN_DIR}/case_f"

run_gpu 2 "w2_unified_synthetic" \
  "${PYTHON}" experiments/nsfc_evidence/run_unified_ablation.py \
  --dataset synthetic --n-tasks 200 \
  --budget-steps 400 --n-samples 50 --seeds ${SEEDS} \
  --output-dir "${RUN_DIR}/unified_synthetic"

run_gpu 3 "w2_sampler_scaling" \
  "${PYTHON}" experiments/run_sampler_scaling.py \
  --nodes 50 100 200 500 --trials 5 --measure-tts \
  --output-dir "${RUN_DIR}/scaling_tts"

wait
echo "[$(date -Iseconds)] Wave 2 完成"

echo "[$(date -Iseconds)] === Wave 3：CPU/轻量 + 可选 14B ==="

# 不占 GPU 的 mock / 合成实验
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

if [[ "${SKIP_CASE_H}" != "1" ]] && [[ -f "${MODEL_14B}/config.json" ]]; then
  echo "[$(date -Iseconds)] Case H: 7B vs 14B（单卡串行，用 GPU 0）..."
  CUDA_VISIBLE_DEVICES=0 "${PYTHON}" experiments/nsfc_evidence/run_case_h_model_compare.py \
    --n-tasks 200 --model-7b "${MODEL_7B}" --model-14b "${MODEL_14B}" \
    --budget-steps 400 --output-dir "${RUN_DIR}/case_h" \
    2>&1 | tee "${RUN_DIR}/case_h.log"
else
  echo "[skip] Case H（SKIP_CASE_H=1 或未找到 14B 模型）"
fi

echo "[$(date -Iseconds)] 汇总同算力表..."
"${PYTHON}" experiments/nsfc_evidence/run_case_c_compute_budget.py \
  --run-dir "${RUN_DIR}" --output-dir "${RUN_DIR}/compute_budget" || true

ARCHIVE="${RUN_DIR}/offline_h200_results.tar.gz"
tar -czf "${ARCHIVE}" -C "${RUN_DIR}" .

cat > "${RUN_DIR}/RUN_SUMMARY.txt" <<EOF
QIHC offline H200 run
started: ${TS}
repo: ${REPO_ROOT}
model_7b: ${MODEL_7B}
model_14b: ${MODEL_14B}
gpus: ${CUDA_VISIBLE_DEVICES}
seeds: ${SEEDS}
hf_limit_per_task: ${HF_LIMIT}

Wave 1: unified_bundled_7b, handoff, cr_hf, case_g
Wave 2: pareto_7b, case_f, unified_synthetic, scaling
Wave 3: case_b, sampler_ablation, f_trajectories, case_h(optional)

Pack results:
  ${ARCHIVE}
EOF

echo "=============================================="
echo " 离线 H200 批量实验完成"
echo " 结果目录: ${RUN_DIR}"
echo " 压缩包:   ${ARCHIVE}"
echo "----------------------------------------------"
echo " 拷回本地分析："
echo "   scp user@h200:${ARCHIVE} ."
echo "=============================================="
