#!/usr/bin/env bash
# Download, resume, update and verify the local model used by QIHC.
set -euo pipefail

WORK_ROOT="${WORK_ROOT:-/hdd/wl2}"
REPO_DIR="${REPO_DIR:-${WORK_ROOT}/QIHC}"
GLOBAL_ROOT="${GLOBAL_ROOT:-${WORK_ROOT}}"
MODEL_ID="${MODEL_ID:-Qwen/Qwen3-Coder-30B-A3B-Instruct}"
MODEL_REVISION="${MODEL_REVISION:-main}"
MODEL_SLUG="${MODEL_ID//\//--}"
MODEL_DIR="${MODEL_DIR:-${GLOBAL_ROOT}/models/${MODEL_SLUG}}"
MODEL_DOWNLOAD_BACKEND="${MODEL_DOWNLOAD_BACKEND:-auto}"
UPDATE_MODEL="${UPDATE_MODEL:-0}"

source "${REPO_DIR}/scripts/s2e/activate_qihc.sh"
mkdir -p "${MODEL_DIR}" "${HF_HOME}/hub"

ARGS=(
  --repo "${MODEL_ID}"
  --revision "${MODEL_REVISION}"
  --local-dir "${MODEL_DIR}"
  --cache-dir "${HF_HOME}/hub"
  --backend "${MODEL_DOWNLOAD_BACKEND}"
  --write-manifest
)
if [[ "${UPDATE_MODEL}" == "1" ]]; then
  # snapshot_download and ModelScope both reuse valid local/cache files; force
  # means check the remote revision, not delete and redownload every shard.
  ARGS+=(--force)
fi

for attempt in 1 2 3 4 5; do
  if python "${REPO_DIR}/experiments/nsfc_evidence/download_model_hf.py" "${ARGS[@]}"; then
    break
  fi
  if [[ "${attempt}" == "5" ]]; then
    echo "Model download/update failed after ${attempt} resumable attempts." >&2
    exit 1
  fi
  echo "Download attempt ${attempt}/5 failed; retrying existing partial files." >&2
  sleep 10
done

python "${REPO_DIR}/experiments/nsfc_evidence/download_model_hf.py" \
  --repo "${MODEL_ID}" --revision "${MODEL_REVISION}" \
  --local-dir "${MODEL_DIR}" --verify-only

python - "${MODEL_DIR}" <<'PY'
import json
import sys
from pathlib import Path

model_dir = Path(sys.argv[1])
config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
manifest = json.loads((model_dir / "qihc_model_manifest.json").read_text(encoding="utf-8"))
print("model_dir=", model_dir)
print("model_type=", config.get("model_type"))
print("architectures=", config.get("architectures"))
print("requested_revision=", manifest.get("requested_revision"))
print("resolved_revision=", manifest.get("resolved_revision"))
print("weight_GiB=", round(manifest["inventory"]["weight_bytes"] / 2**30, 2))
PY

echo "MODEL READY: ${MODEL_DIR}"
echo "For reproducible formal runs: export MODEL_DIR='${MODEL_DIR}'"
