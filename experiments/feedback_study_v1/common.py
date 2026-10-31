"""Identity, atomic records and immutable configuration for E004 only."""
import hashlib
import importlib.metadata
import os
from pathlib import Path
import subprocess
from qihc.experiment_runtime import resilient_storage as s

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BASE = Path(os.environ.get('QIHC_BASE', '/inspire/hdd/global_user/czxs25210022/why'))


def default_root():
    return Path(os.environ.get('QIHC_E004_RUN', str(BASE/'results/E004-1.1_feedback')))


def fingerprint():
    h = hashlib.sha256()
    for folder in ('qihc/modeling', 'qihc/generic', 'qihc/experiment_runtime', 'experiments/feedback_study_v1'):
        for p in sorted((ROOT/folder).rglob('*')):
            if p.suffix in ('.py', '.json', '.sh'):
                h.update(p.relative_to(ROOT).as_posix().encode())
                h.update(p.read_bytes().replace(b'\r\n', b'\n'))
    return h.hexdigest()


def validate(root):
    m = s.load(root/'manifest.json')
    if m['source_hash'] != fingerprint():
        raise ValueError('E004 source changed. Use the pinned worktree; never mix versions in one run.')
    if s.value_digest(m['config']) != m['config_digest']:
        raise ValueError('Prepared configuration identity changed')
    if not m.get('smoke') and m.get('dependencies')!=versions():
        raise ValueError('Pinned environment versions changed; restore the prepared environment')
    if s.digest(root/'cases.public.json') != m['public_digest']:
        raise ValueError('Public case identity mismatch')
    return m


def versions():
    return {name:importlib.metadata.version(name) for name in
            ('numpy','ortools','pyscipopt','torch','transformers','lm-format-enforcer','interegular')}


def commit():
    return subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()


def identity(case_id, arm, seed):
    return s.key(case_id, arm, str(seed))


def configurations(case, cfg):
    if case['split'].startswith('development'):
        return [(a, 0) for a in cfg['public_arms']]
    arms = cfg['public_arms'] if case['split'] == 'public' else cfg['extension_arms']
    if case.get('diagnostic_subset'):
        arms = arms + cfg['diagnostic_arms']
    return [(a, seed) for a in arms for seed in cfg['seeds']]


def backend(arm):
    return 'metropolis' if arm == 'B1' else 'scip' if arm == 'B2' else 'pbit'


def immutable(path, value):
    if path.exists():
        if s.load(path) != value:
            raise ValueError('Immutable artifact changed: '+str(path))
    else:
        s.save(path, value)
