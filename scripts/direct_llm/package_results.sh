#!/usr/bin/env bash
# Package a completed direct-LLM run without including models, caches or source data outside RUN_ROOT.
set -euo pipefail

RUN_ROOT="${1:?usage: package_results.sh RUN_ROOT [ARCHIVE_PATH]}"
RUN_ROOT="$(realpath "${RUN_ROOT}")"
test -d "${RUN_ROOT}"
test -s "${RUN_ROOT}/direct_llm_report.json"
ARCHIVE_PATH="${2:-$(dirname "${RUN_ROOT}")/$(basename "${RUN_ROOT}").tar.gz}"
ARCHIVE_PATH="$(realpath -m "${ARCHIVE_PATH}")"

if [[ "${ARCHIVE_PATH}" == "${RUN_ROOT}"/* ]]; then
  echo "Archive must not be placed inside RUN_ROOT." >&2
  exit 2
fi

tar -C "$(dirname "${RUN_ROOT}")" -czf "${ARCHIVE_PATH}" "$(basename "${RUN_ROOT}")"
sha256sum "${ARCHIVE_PATH}" > "${ARCHIVE_PATH}.sha256"
echo "archive=${ARCHIVE_PATH}"
echo "sha256=${ARCHIVE_PATH}.sha256"
