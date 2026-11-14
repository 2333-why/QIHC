#!/usr/bin/env bash
# Run from a shared checkout on the node appropriate to the selected stage.
source "$(dirname "${BASH_SOURCE[0]}")/../common.sh"

run() { "$PY" -m experiments.pbit_joint_v1.run "$@"; }

case "${1:-status}" in
  status)
    "$PY" - "$QIHC_JOINT_RUN" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
if not (root / 'manifest.json').is_file():
    sys.exit(f'Missing prepared run: {root}')
for arm in ('base', 'plain', 'pbit'):
    folder = root / 'models' / arm
    count = sum(1 for _ in (folder / 'predictions').glob('*.json'))
    print(f'{arm}: {count}/870 predictions; done={(folder / "done.json").exists()}')
print('Saved label files:', sum(1 for _ in (root / 'control_labels').glob('*.json')))
gate = root / 'control_gate.json'
print('Control gate:', json.loads(gate.read_text()) if gate.exists() else 'pending')
PY
    ;;
  predict)
    arm=${2:?Specify plain or pbit}
    case "$arm" in plain|pbit) ;; *) exit 2 ;; esac
    if [[ ! -f "$QIHC_JOINT_RUN/models/$arm/done.json" ]]; then run model "$arm"; fi
    ;;
  cpu)
    run model base
    for arm in plain pbit; do
      if [[ ! -f "$QIHC_JOINT_RUN/models/$arm/done.json" ]]; then
        echo "Resume the predict $arm GPU job first." >&2
        exit 1
      fi
    done
    for arm in base plain pbit; do
      for policy in schedule rule; do run evaluate "$arm" "$policy"; done
    done
    run labels
    ;;
  learned)
    run controller
    for arm in base plain pbit; do run evaluate "$arm" learned; done
    run report
    ;;
  *) echo 'Usage: resume.sh status | predict plain | predict pbit | cpu | learned' >&2; exit 2 ;;
esac
