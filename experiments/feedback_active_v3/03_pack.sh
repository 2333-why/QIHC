#!/usr/bin/env bash
set -euo pipefail
RUN="${QIHC_ACTIVE_RUN:-/inspire/hdd/global_user/czxs25210022/why/results/E004_2.1_active_feedback}"
test -f "$RUN/manifest.json"
OUT="${QIHC_PACK_OUTPUT:-${RUN}_analysis_$(date +%Y%m%dT%H%M%S).tar.gz}"
tar --exclude='jobs/inputs' --exclude='evaluation/jobs/inputs' --exclude='logs/vllm*' \
  -czf "$OUT" -C "$RUN" .
sha256sum "$OUT" > "$OUT.sha256"
echo "$OUT"
