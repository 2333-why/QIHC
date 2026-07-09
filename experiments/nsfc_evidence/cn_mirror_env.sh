#!/usr/bin/env bash
# 国内网络镜像 — 在实验脚本中 source，勿直接执行
#
# 用法（已内置于 run_dual_4090.sh 等）：
#   source "${SCRIPT_DIR}/cn_mirror_env.sh"
#
# 覆盖默认源：
#   PIP_INDEX_URL=https://pypi.org/simple bash run_dual_4090.sh
#   USE_CN_MIRROR=0 bash run_dual_4090.sh

if [[ "${USE_CN_MIRROR:-1}" == "0" ]]; then
  PIP_INSTALL_ARGS=()
  return 0 2>/dev/null || exit 0
fi

# 清华 PyPI 源
export PIP_INDEX_URL="${PIP_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}"
export PIP_TRUSTED_HOST="${PIP_TRUSTED_HOST:-pypi.tuna.tsinghua.edu.cn}"

# HuggingFace 镜像（模型/数据集下载，与 pip 无关但常一起用）
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

# 禁用 Xet/CAS 后端（国内/镜像环境常 401 Unauthorized）
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-0}"
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-600}"
export HF_HUB_ETAG_TIMEOUT="${HF_HUB_ETAG_TIMEOUT:-120}"

# ModelScope（国内服务器下载 Qwen 等模型更稳）
export USE_MODELSCOPE="${USE_MODELSCOPE:-1}"
export MODEL_DOWNLOAD_BACKEND="${MODEL_DOWNLOAD_BACKEND:-auto}"

# pip 统一参数（可被脚本展开）
PIP_INSTALL_ARGS=(-i "${PIP_INDEX_URL}" --trusted-host "${PIP_TRUSTED_HOST}")
