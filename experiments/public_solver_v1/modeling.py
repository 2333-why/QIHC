"""GPU-only productive queue: no CPU solving, no semantic gate, exits when drained."""
import argparse
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from qihc.experiment_runtime import resilient_storage as s
from .common import default_root, validate

STOP = False


def stop(*_):
    global STOP
    STOP = True


def failure(root, path, error):
    if (root/'responses'/path.name).exists():
        return
    s.save(root/'responses'/path.name, {'status': 'generation_error', 'mathematical_validity': 'invalid',
        'response': '', 'validation_error': error, 'request_sha256': s.digest(path),
        'semantic_correctness': 'unverified', 'seconds': None,
        'cost_scope': 'see builder_steps and inference_events; failed attempt cost may be incomplete'})


def items(root, m, rank):
    result = []
    for identity, sha in m['requests'].items():
        if int(identity, 16) % 2 != rank or (root/'responses'/(identity+'.json')).exists():
            continue
        p = root/'requests'/(identity+'.json'); r = s.load(p)
        if s.value_digest(r) != sha:
            raise ValueError('Request identity mismatch')
        result.append((p, r))
    return sorted(result, key=lambda v: (v[1]['queue_order'], v[1]['queue_group']))


def worker(root, rank):
    from qihc.modeling.backend import LocalModel
    from qihc.modeling.typed_workflow import execute
    m = validate(root, model=True)
    with s.worker_lock(root/f'gpu-{rank}.lock'):
        pending = items(root, m, rank)
        if not pending:
            return
        s.save(root/f'gpu-{rank}.json', {'status': 'loading', 'time': time.time()})
        generator = LocalModel(m['model'], device='balanced', temperature=0.)
        placement = getattr(generator.model, 'hf_device_map', {})
        if not placement or any(str(v) in ('cpu', 'disk') for v in placement.values()):
            raise RuntimeError('Model must fit on its assigned GPU pair')
        # No search/quality acceptance: every completed format attempt is persisted.
        for start in range(0, len(pending), m['config']['batch_size']):
            if STOP:
                break
            batch = pending[start:start+m['config']['batch_size']]
            batch = [(p, r) for p, r in batch if not (root/'responses'/p.name).exists()]
            if not batch:
                continue
            s.save(root/f'gpu-{rank}.json', {'status': 'generating', 'time': time.time(),
                'requests': [p.stem for p, _ in batch]})
            try:
                execute(generator, batch, root, rank, f'{rank}-{time.time_ns()}')
            except Exception as exc:
                # Retry a failed batch as single requests; execute replays saved substeps.
                # Fatal device faults are restarted by the supervisor, not hidden as semantics.
                if isinstance(exc, RuntimeError) and not isinstance(exc, generator.torch.cuda.OutOfMemoryError):
                    raise
                for p, r in batch:
                    if (root/'responses'/p.name).exists():
                        continue
                    try:
                        execute(generator, [(p, r)], root, rank, f'{rank}-{time.time_ns()}-single')
                    except Exception as single:
                        if isinstance(single, RuntimeError) and not isinstance(single, generator.torch.cuda.OutOfMemoryError):
                            raise
                        failure(root, p, repr(single))
                        generator.torch.cuda.empty_cache()
            s.save(root/f'gpu-{rank}.json', {'status': 'batch_complete', 'time': time.time()})
            print('MODEL PROGRESS', rank, len(list((root/'responses').glob('*.json'))), '/', len(m['requests']), flush=True)
        s.save(root/f'gpu-{rank}.json', {'status': 'stopped_checkpointed' if STOP else 'complete', 'time': time.time()})


def launch(root):
    import torch
    from qihc.modeling.formatting import check_runtime
    m = validate(root, model=True); cfg = m['config']
    preflight = s.load(root/'preflight.json')
    if preflight.get('status') != 'complete' or preflight.get('source_hash') != m['source_hash']:
        raise ValueError('Run CPU format preparation for this pinned checkout first')
    print(check_runtime(), flush=True)
    if torch.cuda.device_count() != 4:
        raise RuntimeError('Use one worker with four visible GPUs; do not wrap in torchrun')
    ids = os.environ.get('CUDA_VISIBLE_DEVICES', '0,1,2,3').split(',')
    if len(ids) != 4 or len(set(ids)) != 4:
        raise ValueError('Invalid visible GPU list')
    children = {}; restarts = {0: 0, 1: 0}; finished = set(); start = time.monotonic(); last = 0.
    initial = len(list((root/'responses').glob('*.json')))
    def spawn(rank):
        logpath = root/'logs'/f'gpu-{rank}-{time.time_ns()}.log'; logpath.parent.mkdir(exist_ok=True)
        log = logpath.open('ab', buffering=0)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=','.join(ids[2*rank:2*rank+2]))
        proc = subprocess.Popen([sys.executable, '-u', '-m', 'experiments.public_solver_v1.modeling',
            'worker', '--run', str(root), '--rank', str(rank)], env=env, stdout=log, stderr=subprocess.STDOUT)
        children[rank] = proc, log, time.time()
        print('GPU START', rank, str(logpath), flush=True)
    with s.worker_lock(root/'gpu-launch.lock'):
        try:
            for rank in range(2):
                if items(root, m, rank):
                    spawn(rank)
                else:
                    finished.add(rank)
            while len(finished) < 2 and not STOP:
                for rank, (proc, log, launched) in list(children.items()):
                    hb = root/f'gpu-{rank}.json'
                    state = s.load(hb) if hb.exists() else {}
                    # workflow updates substep heartbeats, including long multi-attempt calls.
                    step = root/f'gpu_replica_{rank}.json'
                    latest = max(launched, state.get('time', 0), s.load(step).get('time', 0) if step.exists() else 0)
                    timedout = time.time()-latest > cfg['gpu_watchdog_seconds']
                    code = proc.poll()
                    if timedout or code is not None:
                        if timedout and code is None:
                            proc.kill(); proc.wait()
                        log.close(); del children[rank]
                        if code == 0 and not items(root, m, rank):
                            finished.add(rank); continue
                        # A hung/current bad request may be skipped, but model-loading failure
                        # with no current request remains a visible infrastructure failure.
                        if state.get('status') == 'generating' and state.get('time', 0) >= launched:
                            for identity in state.get('requests', []):
                                failure(root, root/'requests'/(identity+'.json'), f'GPU worker exit={code}, watchdog={timedout}; inspect logs')
                        if restarts[rank] >= cfg['max_gpu_restarts']:
                            raise RuntimeError('Repeated GPU infrastructure failure; saved responses retained. Inspect worker logs and resume.')
                        restarts[rank] += 1
                        spawn(rank)
                if time.monotonic()-last > 30:
                    ready = len(list((root/'responses').glob('*.json'))); elapsed = time.monotonic()-start
                    state = {'status': 'generating', 'responses': ready, 'expected': len(m['requests']),
                        'eta_seconds': elapsed/(ready-initial)*(len(m['requests'])-ready) if ready > initial else None,
                        'time': time.time(), 'restarts': restarts}
                    s.save(root/'modeling_status.json', state); print(state, flush=True); last = time.monotonic()
                time.sleep(1)
            s.save(root/'modeling_status.json', {'status': 'stopped_checkpointed' if STOP else 'complete',
                'responses': len(list((root/'responses').glob('*.json'))), 'time': time.time(),
                'meaning': 'terminal attempts, not semantic success'})
        except Exception as exc:
            s.save(root/'modeling_status.json', {'status': 'infrastructure_failed', 'error': repr(exc), 'time': time.time()})
            raise
        finally:
            for proc, log, launched in children.values():
                if proc.poll() is None:
                    proc.terminate()
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill(); proc.wait()
                log.close()
    print('GPU queue drained; release GPU allocation. Run/resume 02_solve_cpu.sh for scoring.', flush=True)
    return 130 if STOP else 0


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('action', choices=['launch', 'worker'])
    p.add_argument('--run', type=Path, default=default_root()); p.add_argument('--rank', type=int, choices=[0,1], default=0)
    a = p.parse_args(); signal.signal(signal.SIGTERM, stop); signal.signal(signal.SIGINT, stop)
    if a.action == 'worker':
        worker(a.run, a.rank)
    else:
        raise SystemExit(launch(a.run))
