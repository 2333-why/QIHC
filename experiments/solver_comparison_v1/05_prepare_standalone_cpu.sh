#!/usr/bin/env bash
set -Eeuo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/env.sh"
# E002 dependencies only: never prepare or launch E001.
bash "$CODE/experiments/solver_comparison_v1/prepare_solver_deps.sh"
printf '%s\n' 'E002 solver environment READY. E001 was not started. Run 06 after its frozen modeling outputs are available.'
