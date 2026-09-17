#!/usr/bin/env bash
# Backward-compatible entry point. The deployment is now one online 2xPRO6000 host.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/pro6000_online_setup.sh" "$@"
