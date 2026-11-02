"""Immutable experiment identity and bounded subprocess execution."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from qihc.experiment_runtime import resilient_storage as s
from experiments.feedback_study_v1.common import fingerprint as core_fingerprint

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
ENTRY = 'experiments.feedback_witness_v1'
BASE = Path(os.environ.get('QIHC_BASE', '/inspire/hdd/global_user/czxs25210022/why'))


def fingerprint():
    h = hashlib.sha256(core_fingerprint().encode())
    for folder in (HERE, ROOT/'experiments/feedback_pilot_v1', ROOT/'experiments/feedback_comparison_v1'):
        for p in sorted(folder.glob('*')):
            if p.suffix in ('.py', '.json', '.sh'):
                h.update(p.relative_to(ROOT).as_posix().encode())
                h.update(p.read_bytes().replace(b'\r\n', b'\n'))
    return h.hexdigest()


def immutable(path, value):
    if path.exists():
        if s.load(path) != value:
            raise ValueError('Immutable artifact differs: '+str(path))
    else:
        s.save(path, value)


def validate(root):
    m = s.load(root/'manifest.json')
    if m['experiment_id'] != 'E004-1.4' or m['source_hash'] != fingerprint():
        raise ValueError('Wrong experiment or modified source; use frozen worktree')
    if s.value_digest(m['config']) != m['config_digest']:
        raise ValueError('Configuration changed')
    if not m.get('smoke'):
        import importlib.metadata
        if any(importlib.metadata.version(name)!=version for name,version in m['dependencies'].items()):
            raise ValueError('Prepared dependency versions changed')
    for name, digest in m['files'].items():
        if s.digest(root/name) != digest:
            raise ValueError('Frozen input changed: '+name)
    return m


def job_paths(root, job):
    key = s.value_digest(job)
    return root/'jobs'/(key+'.json'), root/'outputs'/(key+'.json')


def cpu_job(root, job, timeout, deadline):
    inp, out = job_paths(root, job)
    immutable(inp, job)
    if out.exists():
        result = s.load(out)
        if result.get('job_sha256') != s.digest(inp):
            raise ValueError('Cached CPU result identity mismatch')
        return result
    claim = inp.with_suffix('.started.json')
    if claim.exists():
        result = {'status': 'interrupted_no_retry', 'x': None}
    elif time.time() >= deadline:
        result = {'status': 'deadline_not_run', 'x': None}
    else:
        s.save(claim, {'started_at': time.time()})
        started = time.monotonic()
        log = inp.with_suffix('.log')
        try:
            with log.open('wb') as stream:
                proc = subprocess.run([sys.executable, '-m', ENTRY+'.cpu', str(inp), str(out)],
                                      stdout=stream, stderr=subprocess.STDOUT,
                                      timeout=max(.01, min(timeout, deadline-time.time())))
            result = s.load(out) if out.exists() else {'status': 'worker_failed', 'returncode': proc.returncode, 'x': None}
        except subprocess.TimeoutExpired:
            result = {'status': 'budget_timeout' if time.time() < deadline else 'deadline_interrupted', 'x': None}
        result['process_seconds'] = time.monotonic()-started
    result['job_sha256'] = s.digest(inp)
    s.save(out, result)
    return result


def public_case(c):
    return {k: c[k] for k in ('id', 'question', 'variable_names')}
