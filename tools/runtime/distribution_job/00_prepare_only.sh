#!/usr/bin/env bash
# No search worker, no model loading, no GPU needed. Reuses downloaded assets.
set -Eeuo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
exec bash "$HERE/01_run_4gpu80.sh" --prepare-only "$@"
