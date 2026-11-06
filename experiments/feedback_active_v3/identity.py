"""Fingerprint actual uncommitted release files, not merely the Git HEAD."""
import hashlib
from pathlib import Path

from qihc.experiment_runtime.storage import value_digest


def source_identity():
    root=Path(__file__).resolve().parents[2]
    paths=list((root/'qihc').rglob('*.py'))
    paths += [p for p in (root/'experiments/feedback_active_v3').glob('*') if p.is_file() and p.suffix in ('.py','.sh','.json','.txt')]
    paths += [p for p in (root/'experiments/feedback_active_v4').glob('*') if p.is_file()]
    values={p.relative_to(root).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}
    return dict(sha256=value_digest(values),files=values)
