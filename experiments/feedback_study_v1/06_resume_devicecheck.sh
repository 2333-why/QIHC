#!/usr/bin/env bash
set -Eeuo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/env.sh"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
"$PY" -u -m experiments.feedback_study_v1.recover_startup
exec bash experiments/feedback_study_v1/01_run_4gpu.sh "$@"
