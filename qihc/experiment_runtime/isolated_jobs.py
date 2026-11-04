"""Bounded asynchronous CPU subprocesses with native-library isolation."""
from __future__ import annotations

import asyncio
import os
from pathlib import Path
import signal
import sys
import time

from .storage import load, save, value_digest


class JobPool:
    def __init__(self, directory, workers, module):
        self.directory = Path(directory); self.slots = asyncio.Semaphore(workers)
        self.module = module; self.locks = {}; self.active = set()

    async def run(self, job, timeout):
        key = value_digest(job)
        async with self.locks.setdefault(key, asyncio.Lock()):
            ip = self.directory / 'inputs' / (key + '.json'); op = self.directory / 'outputs' / (key + '.json')
            if op.exists():
                row = load(op)
                if row.get('job_hash') != key: raise ValueError('CPU cache identity mismatch')
                if row.get('status') != 'interrupted': return row
                save(self.directory / 'interrupted_attempts' / (key + '_' + str(time.time_ns()) + '.json'), row)
            async with self.slots:
                save(ip, job); started = time.monotonic()
                env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1')
                proc = await asyncio.create_subprocess_exec(sys.executable, '-m', self.module, str(ip), str(op),
                    stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.PIPE, env=env,
                    start_new_session=os.name != 'nt')
                self.active.add(proc)
                try:
                    _, errors = await asyncio.wait_for(proc.communicate(), timeout)
                    if proc.returncode or not op.exists():
                        save(op, dict(status='worker_failed', error=errors.decode(errors='replace')[-2400:], job_hash=key, x=None))
                except (asyncio.TimeoutError, asyncio.CancelledError) as exc:
                    if proc.returncode is None:
                        if os.name != 'nt':
                            try: os.killpg(proc.pid, signal.SIGKILL)
                            except ProcessLookupError: pass
                        else: proc.kill()
                        await proc.wait()
                    save(op, dict(status='worker_timeout' if isinstance(exc, asyncio.TimeoutError) else 'interrupted',
                                  job_hash=key, x=None, seconds=time.monotonic() - started))
                    if isinstance(exc, asyncio.CancelledError): raise
                finally:
                    self.active.discard(proc)
                return load(op)
