"""Frozen identity and single-flight, bounded subprocess jobs."""
import hashlib
import errno
import os
from pathlib import Path
import subprocess
import sys
import time
from qihc.experiment_runtime import resilient_storage as s
from experiments.feedback_witness_v1.common import immutable, public_case, job_paths
from experiments.feedback_witness_v1.common import fingerprint as prior_fingerprint

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
ENTRY = 'experiments.feedback_witness_v2'
BASE = Path(os.environ.get('QIHC_BASE', '/inspire/hdd/global_user/czxs25210022/why'))


def fingerprint():
    h = hashlib.sha256(prior_fingerprint().encode())
    for p in sorted(HERE.iterdir()):
        if p.suffix in ('.py', '.json', '.sh'):
            h.update(p.relative_to(ROOT).as_posix().encode())
            h.update(p.read_bytes().replace(b'\r\n', b'\n'))
    return h.hexdigest()


def validate(root):
    m = s.load(root/'manifest.json')
    if m['experiment_id'] != 'E004-1.5' or m['source_hash'] != fingerprint():
        raise ValueError('Wrong experiment or changed source; use frozen worktree')
    if s.value_digest(m['config']) != m['config_digest']:
        raise ValueError('Configuration changed')
    if not m.get('smoke'):
        import importlib.metadata
        if any(importlib.metadata.version(k) != v for k, v in m['dependencies'].items()):
            raise ValueError('Prepared dependency versions changed')
    for name, digest in m['files'].items():
        if s.digest(root/name) != digest:
            raise ValueError('Frozen input changed: '+name)
    return m


def cpu_job(root, job, timeout, deadline):
    inp, out = job_paths(root, job)
    lock = s.worker_lock(inp.with_suffix('.lock'))
    # A live owner is never classified as interrupted. Wait for its canonical result.
    while True:
        try:
            lock.__enter__()
            break
        except OSError as exc:
            if exc.errno not in (errno.EACCES, errno.EAGAIN, errno.EWOULDBLOCK):
                raise
            if time.time() >= deadline:
                # Caller-local result only: never overwrite the live owner's output.
                return {'status': 'deadline_waiting_for_job', 'x': None,
                        'job_key': s.value_digest(job)}
            time.sleep(.025)
            lock = s.worker_lock(inp.with_suffix('.lock'))
    try:
        immutable(inp, job)
        if out.exists():
            row = s.load(out)
            if row.get('job_sha256') != s.digest(inp):
                raise ValueError('Cached CPU result identity mismatch')
            return row
        claim = inp.with_suffix('.started.json')
        if claim.exists():
            # Only the exclusive lock owner can now diagnose an interrupted earlier owner.
            row = {'status': 'interrupted_no_retry', 'x': None}
        elif time.time() >= deadline:
            row = {'status': 'deadline_not_run', 'x': None}
        else:
            s.save(claim, {'started_at': time.time(), 'owner_pid': os.getpid()})
            started = time.monotonic()
            try:
                with inp.with_suffix('.log').open('wb') as stream:
                    proc = subprocess.run([sys.executable, '-m', ENTRY+'.cpu', str(inp), str(out)],
                                          stdout=stream, stderr=subprocess.STDOUT,
                                          timeout=max(.01, min(timeout, deadline-time.time())))
                row = s.load(out) if out.exists() else {'status': 'worker_failed', 'returncode': proc.returncode, 'x': None}
            except subprocess.TimeoutExpired:
                row = {'status': 'budget_timeout' if time.time() < deadline else 'deadline_interrupted', 'x': None}
            row['process_seconds'] = time.monotonic()-started
        row['job_sha256'] = s.digest(inp)
        s.save(out, row)
        return row
    finally:
        lock.__exit__(None, None, None)
