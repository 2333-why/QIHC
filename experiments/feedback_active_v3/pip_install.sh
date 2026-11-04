#!/usr/bin/env bash
# Source this helper; retry the same constraints against a domestic mirror.
qihc_pip_install() {
  local interpreter="$1" mirror
  shift
  local -a mirrors=("${PIP_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}" "https://mirrors.aliyun.com/pypi/simple")
  for mirror in "${mirrors[@]}"; do
    echo "Installing dependencies from: $mirror"
    if "$interpreter" -m pip install --index-url "$mirror" --retries 2 "$@"; then
      return 0
    fi
    echo "Installation failed on this source; retrying with the same package constraints." >&2
  done
  echo "Both domestic sources failed; retain the pip error log for diagnosis." >&2
  return 1
}
