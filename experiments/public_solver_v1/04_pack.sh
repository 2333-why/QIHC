#!/usr/bin/env bash
set -Eeuo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/env.sh"
bash experiments/public_solver_v1/03_report.sh
OUT="$BASE/E003-1.1_analysis_$(TZ=Asia/Shanghai date +%Y%m%dT%H%M%S).tar.gz"
FILES=(manifest.json cases.CPU_ONLY.json summary.json report.md models scores references)
for ITEM in preflight.json modeling_status.json solver_status.json logs; do
  [[ ! -e "$QIHC_E003_RUN/$ITEM" ]] || FILES+=("$ITEM")
done
tar -czf "$OUT" -C "$QIHC_E003_RUN" "${FILES[@]}"
echo "$OUT"
