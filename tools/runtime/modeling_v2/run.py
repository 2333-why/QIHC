"""Single-allocation orchestration; scientific code and live legacy runs stay untouched."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from contextlib import ExitStack
import multiprocessing
import hashlib
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from tools.benchmarks import comparison as c
from tools.runtime.modeling_v2 import experiment as e
from tools.runtime.modeling_v2.health import health, gate_check

STOP = False


def stop(signum, frame):
    global STOP
    STOP = True


def complete(run):
    path = run/'summary.json'
    return path.exists() and c.load(path).get('status') == 'complete'


def process_case(run, case, cfg):
    """Same search/checkpoints; release fixed-seed evidence without waiting for other seeds."""
    result = e.process_case(run, case, cfg)
    initial = e.response(run, case, 'model')
    if initial is not None:
        folder = run/'progress'/c.key(case['id'], 'case')
        for arm in cfg['arms']:
            path = folder/f'{arm}.initial.0.json'
            if path.exists():
                row = e.check_checkpoint(path, initial['response'], arm, 0)
                c.enqueue(run, case, 'review_'+arm,
                          e.review_prompt(case, initial['response'], row['diagnostic'], arm))
    return result


def cpu(args):
    # Parent owns cpu.lock for the whole lifetime. Spawn prevents pool children
    # from inheriting that lock, CUDA state, or the parent's signal handlers.
    manifest = e.validate(args.run)
    cases, cfg = c.load(args.run/'cases.CPU_ONLY.json'), manifest['settings']
    active, cursor, last_report = {}, 0, 0.
    with ProcessPoolExecutor(args.workers, mp_context=multiprocessing.get_context('spawn')) as pool:
        try:
            while not STOP:
                for future in list(active):
                    if future.done():
                        case = active.pop(future)
                        try:
                            future.result()
                        except BaseException as exc:
                            c.save(args.run/'errors'/(c.key(case['id'], 'case')+'.json'),
                                   {'case': case['id'], 'error': repr(exc), 'time': time.time()})
                            raise
                ids = {case['id'] for case in active.values()}
                for _ in range(len(cases)):
                    if len(active) >= args.workers:
                        break
                    case = cases[cursor % len(cases)]
                    cursor += 1
                    if case['id'] in ids or (args.run/'final'/(c.key(case['id'], 'case')+'.json')).exists():
                        continue
                    active[pool.submit(process_case, args.run, case, cfg)] = case
                    ids.add(case['id'])
                if time.monotonic()-last_report >= 15:
                    report = e.summarize(args.run, cases, cfg)
                    c.save(args.run/'cpu_status.json', {'status': 'running', 'time': time.time(),
                           'active_cases': list(ids), 'completed': report['completed']})
                    print(f"CPU complete={report['completed']}/{len(cases)} active={len(active)}", flush=True)
                    last_report = time.monotonic()
                    if report['status'] == 'complete':
                        break
                time.sleep(.25)
        finally:
            # Graceful stop completes at most one seed per active worker.
            for future in active:
                future.cancel()
    report = e.summarize(args.run, cases, cfg)
    c.save(args.run/'cpu_status.json', {'status': 'complete' if report['status'] == 'complete'
                                      else 'stopped_checkpointed', 'time': time.time(),
                                      'completed': report['completed']})


def gpu(args):
    from tools.runtime.modeling_v2 import inference
    signal.signal(signal.SIGTERM, inference.stop)
    signal.signal(signal.SIGINT, inference.stop)
    # Parent supervises queue starvation and exits on CPU failure. Keep weights
    # loaded across legitimate CPU phases, rather than reload every 120 seconds.
    args.idle_exit = 86400 * 365
    inference.worker(args)


def pending_count(run):
    return sum(not (run/'responses'/p.name).exists() for p in (run/'requests').glob('*.json'))


def device_groups(visible, count, replicas):
    if count != 4 or replicas != 2:
        raise ValueError('This entry requires exactly 4 visible GPUs and 2 model replicas')
    devices = visible.split(',') if visible else [str(i) for i in range(count)]
    if len(devices) != count or len(set(devices)) != count:
        raise ValueError('CUDA visible-device inventory mismatch')
    return [','.join(devices[i:i+2]) for i in range(0, count, 2)]


def child_env(gpus=''):
    return dict(os.environ, CUDA_VISIBLE_DEVICES=gpus, PYTHONPATH=str(ROOT),
                PYTHONNOUSERSITE='1', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                MKL_NUM_THREADS='1', TOKENIZERS_PARALLELISM='false', HF_HUB_OFFLINE='1',
                TRANSFORMERS_OFFLINE='1', HF_DATASETS_OFFLINE='1', PIP_NO_INDEX='1',
                PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')


def shutdown(children, grace=120):
    """Terminate only our process groups, including CPU pool grandchildren."""
    for proc in children:
        if proc.poll() is None:
            proc.terminate()  # Let CPU supervisor finish seed checkpoints first.
    deadline = time.monotonic()+grace
    while any(p.poll() is None for p in children) and time.monotonic() < deadline:
        time.sleep(.2)
    for proc in children:
        if os.name == 'posix':
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        elif proc.poll() is None:
            proc.kill()
        proc.wait()


def check_children(children, finished):
    for name, proc in children.items():
        code = proc.poll()
        # Reporter reads final files directly and may finish before the CPU's
        # next 15-second summary tick. A clean report exit is not a failure.
        if name == 'report' and code == 0:
            continue
        if code is not None and (code != 0 or not finished):
            raise RuntimeError(f'{name} exited {code} before successful completion; inspect its log, then resume')


def supervise(args, groups):
    children, logs = {}, []
    stamp = time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())+f'-{os.getpid()}'
    logdir = args.run/'job_logs'
    logdir.mkdir(exist_ok=True)
    base = [sys.executable, '-u', str(Path(__file__).resolve())]

    def start(name, command, env):
        handle = (logdir/f'{stamp}-{name}.log').open('ab', buffering=0)
        logs.append(handle)
        children[name] = subprocess.Popen(command, env=env, stdout=handle, stderr=subprocess.STDOUT,
                                           start_new_session=True)
        print(f'START {name}: {handle.name}', flush=True)

    idle_since, last_status = None, 0.
    result, error = 'stopped_checkpointed', None
    try:
        start('cpu', base+['cpu', '--run', str(args.run), '--workers', str(args.workers)], child_env())
        for rank, devices in enumerate(groups):
            start(f'gpu-{rank}', base+['gpu', '--run', str(args.run), '--rank', str(rank),
                  '--batch-size', str(args.batch_size)], child_env(devices))
        start('report', [sys.executable, '-u', str(ROOT/'tools/analysis/distribution_feedback/report.py'),
                        '--run', str(args.run), '--output', str(args.output), '--cpu-workers', str(args.workers),
                        '--replicas', '2', '--watch-seconds', '60'], child_env())
        while not STOP:
            finished = complete(args.run)
            check_children(children, finished)
            if finished:
                result = 'complete'
                break
            now = time.monotonic()
            pending = pending_count(args.run)
            idle_since = None if pending else (now if idle_since is None else idle_since)
            idle = 0. if idle_since is None else now-idle_since
            if now-last_status >= 30:
                quality = health(args.run)
                c.save(args.run/'health.json', quality)
                if quality['trip']:
                    raise RuntimeError('QUALITY CIRCUIT BREAKER: >=80% invalid among last 12..24 distinct models; inspect health.json')
                for rank in range(2):
                    path = args.run/f'gpu_replica_{rank}.json'
                    if path.exists():
                        g = c.load(path)
                        if g.get('status') == 'generating' and time.time()-g.get('time', time.time()) > 1800:
                            raise RuntimeError('GPU batch exceeded 30-minute watchdog; inspect replica log')
                state = {'status': 'running', 'time': time.time(), 'pending_requests': pending,
                         'queue_empty_seconds': round(idle), 'workers': args.workers,
                         'replicas': 2, 'batch_size': args.batch_size,
                         'warning': 'CPU is producing feedback; no artificial GPU work' if idle >= 180 else None}
                c.save(args.run/'job_status.json', state)
                print('JOB '+str(state), flush=True)
                last_status = now
            if idle >= args.max_idle:
                raise RuntimeError(f'GPU queue empty for {args.max_idle}s; checkpointing and releasing allocation. '
                                   'Inspect CPU logs. No synthetic keep-alive work. Resume the same command.')
            time.sleep(1)
    except BaseException as exc:
        error = repr(exc)
        raise
    finally:
        shutdown(list(children.values()))
        for handle in logs:
            handle.close()
        c.save(args.run/'job_status.json', {'status': result, 'time': time.time(),
                                          'error': error,
                                          'note': 'Completed atomic checkpoints retained; same command resumes.'})
    if STOP:
        return 130
    # Refresh after workers stop, so the final report cannot be left one tick behind.
    subprocess.run([sys.executable, str(ROOT/'tools/analysis/distribution_feedback/report.py'),
                    '--run', str(args.run), '--output', str(args.output), '--cpu-workers', str(args.workers),
                    '--replicas', '2'], env=child_env(), check=True)
    return 0


def launch(args):
    if os.name != 'posix' and not args.prepare_only:
        raise ValueError('Unified launcher targets Linux compute jobs')
    for key in ('WORLD_SIZE', 'LOCAL_WORLD_SIZE', 'GROUP_WORLD_SIZE', 'ROLE_WORLD_SIZE'):
        if os.environ.get(key, '1') != '1':
            raise ValueError('Launch once: 1 worker with 4 GPUs, not torchrun')
    for key in ('RANK', 'LOCAL_RANK', 'GROUP_RANK', 'ROLE_RANK'):
        if os.environ.get(key, '0') != '0':
            raise ValueError('Only rank zero is supported')
    for key in ('RANK', 'LOCAL_RANK', 'WORLD_SIZE', 'LOCAL_WORLD_SIZE', 'GROUP_RANK', 'GROUP_WORLD_SIZE',
                'ROLE_RANK', 'ROLE_WORLD_SIZE', 'MASTER_ADDR', 'MASTER_PORT'):
        os.environ.pop(key, None)
    groups = None
    if not args.prepare_only:
        import torch
        groups = device_groups(os.environ.get('CUDA_VISIBLE_DEVICES'), torch.cuda.device_count(), 2)
        available = len(os.sched_getaffinity(0))
        if args.workers > available:
            raise ValueError(f'{args.workers} CPU workers exceed allocated CPU affinity {available}')
    if args.output.resolve() == args.run.resolve() or args.run.resolve() in args.output.resolve().parents:
        raise ValueError('Report output must be outside experiment directory')
    # Same locks as old split workers. An active run is refused, never killed or migrated.
    with ExitStack() as locks:
        locks.enter_context(c.worker_lock(args.run.parent/(args.run.name+'.unified.lock')))
        locks.enter_context(c.worker_lock(args.run.parent/(args.run.name+'.prepare.lock')))
        existing = (args.run/'manifest.json').exists()
        if existing:
            locks.enter_context(c.worker_lock(args.run/'cpu.lock'))
            locks.enter_context(c.worker_lock(args.run/'gpu.lock'))
        e.prepare(argparse.Namespace(run=args.run, from_run=args.from_run, budget=512 if args.gate else 8192,
                                    chains=4 if args.gate else 16, seeds=1 if args.gate else 3,
                                    solver_seconds=5., limit=0, arms=['pbit_selected'] if args.gate else list(e.ARMS), gate=args.gate))
        if args.prepare_only:
            print('PREPARED ONLY: no CPU search or GPU inference started; online CPU may exit.', flush=True)
            return 0
        if STOP:
            return 130
        if not existing:
            locks.enter_context(c.worker_lock(args.run/'cpu.lock'))
            locks.enter_context(c.worker_lock(args.run/'gpu.lock'))
        c.save(args.run/'execution'/f'{time.time_ns()}.json', {
            'scheduler': 'unified_4gpu80_v1', 'scientific_source': e.source_hash(),
            'runtime_source_sha256': hashlib.sha256(Path(__file__).read_bytes().replace(b'\r\n', b'\n')).hexdigest(),
            'workers': args.workers, 'replicas': 2, 'batch_size': args.batch_size,
            'early_feedback': 'fixed seed zero only; no change to search budgets or prompts',
            'note': 'Batch composition may change numerical generation; execution history retained.',
            'time': time.time()})
        if complete(args.run):
            subprocess.run([sys.executable, str(ROOT/'tools/analysis/distribution_feedback/report.py'),
                            '--run', str(args.run), '--output', str(args.output), '--cpu-workers', str(args.workers),
                            '--replicas', '2'], env=child_env(), check=True)
            return 0
        return supervise(args, groups)


def main():
    base = Path(os.environ.get('QIHC_BASE', '/inspire/hdd/global_user/czxs25210022/why'))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['launch', 'cpu', 'gpu'], nargs='?', default='launch')
    parser.add_argument('--run', type=Path, default=Path(os.environ.get('QIHC_DISTRIBUTION_RUN', str(base/'results/distribution_feedback_modeling_v2'))))
    parser.add_argument('--from-run', type=Path, default=Path(os.environ.get('QIHC_DISTRIBUTION_SOURCE', str(base/'results/co_comparison_v1'))))
    parser.add_argument('--output', type=Path)
    parser.add_argument('--workers', type=int, default=int(os.environ.get('QIHC_DISTRIBUTION_CPU_WORKERS', '32')))
    parser.add_argument('--batch-size', type=int, default=int(os.environ.get('QIHC_DISTRIBUTION_BATCH', '4')))
    parser.add_argument('--max-idle', type=int, default=900)
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--gate', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    args.output = args.output or Path(str(args.run)+'_reports')
    args.replicas = 2
    if not 1 <= args.workers <= 64 or not 1 <= args.batch_size <= 32 or args.max_idle < 180 or args.rank not in (0, 1):
        parser.error('Invalid worker/batch/idle/rank settings')
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    if args.stage == 'launch' and not args.prepare_only and not args.gate:
        gate_run = Path(str(args.run)+'_gate')
        command = [sys.executable, '-u', str(Path(__file__).resolve()), 'launch', '--gate',
                   '--run', str(gate_run), '--from-run', str(args.from_run), '--workers', str(args.workers),
                   '--batch-size', str(args.batch_size), '--max-idle', str(args.max_idle)]
        proc = subprocess.Popen(command, start_new_session=True)
        try:
            while proc.poll() is None and not STOP:
                time.sleep(1)
            if STOP:
                return 130
            if proc.returncode:
                raise RuntimeError('Real-model startup gate failed; full experiment was NOT launched')
        finally:
            shutdown([proc], grace=150)
        checked = gate_check(gate_run)
        c.save(gate_run/'gate_acceptance.json', checked)
        if not checked['passed']:
            raise RuntimeError('Real-model gate did not search every family in both stages; full experiment blocked')
        print('REAL MODEL GATE PASSED: initial and revised search verified. Starting full 12-arm experiment.', flush=True)
    return {'launch': launch, 'cpu': cpu, 'gpu': gpu}[args.stage](args) or 0


if __name__ == '__main__':
    raise SystemExit(main())
