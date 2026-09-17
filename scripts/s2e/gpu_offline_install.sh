#!/usr/bin/env bash
# Backward-compatible entry point. Offline installation is no longer required.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "INFO: using the unified online 2xPRO6000 setup; no offline bundle is created."
exec bash "${SCRIPT_DIR}/pro6000_online_setup.sh" "$@"
