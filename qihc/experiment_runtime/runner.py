"""One allocation: two 2-GPU replicas, CPU evaluation pool and automatic reports."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone, timedelta

from qihc.experiment_runtime import storage as s

STOP = False


def stop(*_):
    global STOP
    STOP = True


def record_event(root, message):
    stamp = datetime.now(timezone(timedelta(hours=8))).isoformat(timespec='seconds')
    with (root/'record.md').open('a', encoding='utf-8') as f:
        f.write(f'\n- {stamp}: {message}\n')


def pending(directory, rank, batch):
    rows = []
    for p in (directory/'requests').glob('*.json'):
        if int(p.stem, 16) % 2 != rank or (directory/'responses'/p.name).exists():
            continue
        row = s.load(p)
        if hashlib.sha256(row['question'].encode()).hexdigest() != row['question_sha256']:
            raise ValueError('Source text integrity mismatch')
        rows.append((p, row))
    # Opt-in stratified ordering. Historical requests without queue_group keep
    # their exact ordering; new runs interleave datasets rather than exhaust A.
    rows.sort(key=lambda x: (x[1]['case_id'], x[1]['phase']))
    if rows and any('queue_group' in r for _, r in rows):
        rows.sort(key=lambda x: (x[1].get('queue_order', 0), x[1].get('queue_group', ''),
                                 x[1]['case_id'], x[1]['phase']))
    if not rows:
        return []
    kind = rows[0][1]['frontend']
    return [(p, r) for p, r in rows if r['frontend'] == kind][:batch]


def worker(root, rank, study):
    from qihc.modeling.backend import LocalModel
    from qihc.modeling.workflow import execute
    cfg = study.validate(root, model=True)
    with s.worker_lock(root/f'gpu-{rank}.lock'):
        print(f'GPU {rank}: loading {cfg["model"]}', flush=True)
        s.save(root/f'gpu-{rank}.json', {'status': 'loading', 'time': time.time()})
        generator = LocalModel(cfg['model'], device='balanced', temperature=0.)
        placement = getattr(generator.model, 'hf_device_map', {})
        if not placement or any(str(v) in ('cpu', 'disk') for v in placement.values()):
            raise RuntimeError('Model must fit in assigned GPUs; CPU/disk offload detected')
        print(f'GPU {rank}: weights loaded; placement={placement}', flush=True)
        while not STOP:
            phase = s.load(root/'phase.json')['phase']
            if phase == 'done':
                break
            directory = root/phase
            items = pending(directory, rank, cfg['config']['batch_size'])
            for path, row in items:
                if s.digest(path) != cfg['request_hashes'].get(row['id']):
                    raise ValueError('Prepared request hash mismatch')
            if not items:
                s.save(root/f'gpu-{rank}.json', {'status': 'idle', 'phase': phase, 'time': time.time()})
                time.sleep(2)
                continue
            batch = f'{rank}-{time.time_ns()}'
            s.save(root/f'gpu-{rank}.json', {'status': 'generating', 'phase': phase, 'time': time.time(),
                    'requests': [r['id'] for _, r in items]})
            getattr(study, "execute", execute)(generator, items, directory, rank, batch)
            s.save(root/f'gpu-{rank}.json', {'status': 'batch_complete', 'phase': phase, 'time': time.time()})


def gate_check(report):
    # Execution acceptance only. Poor feasibility/coverage is a measured outcome.
    failures = []
    for arm, row in report['overall'].items():
        if not row['compilable_fraction_of_generated'] or any(b['searched_runs'] < 1 for b in row['backends'].values()):
            failures.append(arm)
    return {'passed': report['status'] == 'complete' and not failures, 'failed_arms': failures,
            'scope': 'Each arm must generate at least one executable model searched by both backends. Not semantic certification.'}


def child_failure(children, logs):
    for rank, proc in enumerate(children):
        code = proc.poll()
        if code is not None:
            path = Path(logs[rank].name)
            with path.open('rb') as f:
                f.seek(max(0, path.stat().st_size-16000))
                tail = f.read().decode('utf-8', errors='replace')
            print(f'GPU {rank} log: {path}\n{tail}', flush=True)
            raise RuntimeError(f'GPU {rank} exited {code}; root exception printed above; log={path}')


def launch(root, study):
    from qihc.modeling.formatting import check_runtime
    cfg = study.validate(root, model=True)
    readiness = s.load(root/'preflight.json')
    if readiness.get('source_hash') != cfg['source_hash'] or readiness.get('status') != 'passed':
        raise ValueError('Run CPU preparation for this exact checkout first')
    import torch
    print(json.dumps(check_runtime()), flush=True)
    if torch.cuda.device_count() != 4:
        raise RuntimeError('This experiment requires four visible GPUs in one worker')
    devices = os.environ.get('CUDA_VISIBLE_DEVICES', '0,1,2,3').split(',')
    if len(devices) != 4 or len(set(devices)) != 4:
        raise ValueError('Invalid visible GPU inventory')
    settings = cfg['config']; children = []; logs = []; active = {}
    started = time.monotonic(); idle_since = None
    with s.worker_lock(root/'launch.lock'):
        try:
            record_event(root, 'Launch/resume; commit='+cfg['code_commit']+'; source='+cfg['source_hash'])
            phase = 'gate'
            s.save(root/'phase.json', {'phase': phase})
            logdir = root/'logs'; logdir.mkdir(exist_ok=True)
            for rank in range(2):
                path = logdir/f'{time.strftime("%Y%m%dT%H%M%SZ",time.gmtime())}-{os.getpid()}-gpu-{rank}.log'
                f = path.open('ab', buffering=0); logs.append(f)
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=','.join(devices[rank*2:rank*2+2]),
                           PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
                children.append(subprocess.Popen([sys.executable, '-u', '-m', study.ENTRY_MODULE,
                            'worker', '--run', str(root), '--rank', str(rank)], env=env, stdout=f,
                            stderr=subprocess.STDOUT, start_new_session=True))
                print(f'START gpu-{rank}: {path}', flush=True)
            with ProcessPoolExecutor(settings['cpu_workers'], mp_context=multiprocessing.get_context('spawn')) as pool:
                for phase in ('gate', 'formal'):
                    record_event(root, 'Enter phase: '+phase)
                    directory = root/phase
                    if s.digest(directory/'cases.CPU_ONLY.json') != cfg['phase_cases_sha256'][phase]:
                        raise ValueError('CPU-only phase cases changed')
                    cases = s.load(directory/'cases.CPU_ONLY.json')
                    # Rebuild exact requests on CPU to verify cases/queue integrity before use.
                    for case in cases:
                        for arm in settings['arms']:
                            req = study.request(case, arm)
                            if s.load(directory/'requests'/(req['id']+'.json')) != req:
                                raise ValueError('Request was modified after preparation')
                    s.save(root/'phase.json', {'phase': phase})
                    last_report = 0.; idle_since = None
                    initial_responses = len(list((directory/'responses').glob('*.json')))
                    phase_started = time.monotonic()
                    while not STOP:
                        child_failure(children, logs)
                        for name, future in list(active.items()):
                            if future.done():
                                future.result(); del active[name]
                        for name, args in study.tasks(directory, cases, settings):
                            if len(active) >= settings['cpu_workers']:
                                break
                            if name not in active:
                                active[name] = pool.submit(study.evaluate_one, *args)
                        if time.monotonic()-last_report >= 30:
                            report = study.summarize(root, phase)
                            ready = len(list((directory/'responses').glob('*.json')))
                            total = report['expected_case_arms']; pending_count = total-ready
                            now = time.monotonic()
                            idle_since = None if pending_count else (idle_since or now)
                            completed_this_launch = ready-initial_responses
                            eta = ((now-phase_started)/completed_this_launch*pending_count) if completed_this_launch >= 10 else None
                            state = {'status': 'running', 'phase': phase, 'responses': ready, 'requests': total,
                                     'completed_case_arms': report['completed_case_arms'], 'active_cpu': len(active),
                                     'generation_eta_seconds': eta, 'eta_scope': 'current-phase measured throughput; excludes remaining CPU tail and future phase',
                                     'elapsed_seconds': now-started, 'time': time.time()}
                            s.save(root/'status.json', state); print(json.dumps(state), flush=True)
                            if report['status'] == 'complete':
                                break
                            for rank in range(2):
                                path = root/f'gpu-{rank}.json'
                                if path.exists():
                                    gpu = s.load(path)
                                    if gpu['status'] in ('generating', 'loading') and time.time()-gpu['time'] > settings['generation_watchdog_seconds']:
                                        raise RuntimeError(f'GPU {rank} watchdog expired; inspect its log')
                            if idle_since and now-idle_since > settings['max_idle_seconds']:
                                raise RuntimeError('GPU queue drained while CPU unfinished; checkpointed. No synthetic resource occupancy.')
                            last_report = now
                        time.sleep(1)
                    if STOP:
                        break
                    if phase == 'gate':
                        accepted = getattr(study, 'gate_check', gate_check)(report)
                        s.save(root/'gate_acceptance.json', accepted)
                        record_event(root, 'Execution acceptance: '+json.dumps(accepted))
                        if not accepted['passed']:
                            print('GATE FAILURE DETAILS '+json.dumps({a: report['overall'][a] for a in accepted['failed_arms']}), flush=True)
                            raise RuntimeError('Execution gate failed: '+str(accepted['failed_arms'])+'; inspect gate/report.md and builder_steps')
                        print('EXECUTION GATE PASSED; starting formal prompt comparisons', flush=True)
            s.save(root/'phase.json', {'phase': 'done'})
            s.save(root/'status.json', {'status': 'stopped_checkpointed' if STOP else 'complete',
                                       'elapsed_seconds_this_launch': time.monotonic()-started, 'time': time.time()})
            record_event(root, 'Stopped with checkpoints' if STOP else 'Completed; see formal/report.md and summary.json')
        except BaseException as exc:
            s.save(root/'status.json', {'status': 'failed', 'error': repr(exc),
                                       'elapsed_seconds_this_launch': time.monotonic()-started, 'time': time.time()})
            record_event(root, 'Failed: '+repr(exc))
            raise
        finally:
            for proc in children:
                if proc.poll() is None:
                    proc.terminate()
            deadline = time.monotonic()+15
            while any(p.poll() is None for p in children) and time.monotonic() < deadline:
                time.sleep(.2)
            for proc in children:
                if proc.poll() is None:
                    proc.kill()
                proc.wait()
            for f in logs:
                f.close()
    return 130 if STOP else 0


def main(study):
    base = Path(os.environ.get('QIHC_BASE', '/inspire/hdd/global_user/czxs25210022/why'))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'launch', 'worker', 'report', 'status'])
    parser.add_argument('--run', type=Path, default=Path(os.environ.get(study.RUN_ENV, str(base/'results'/study.NAME))))
    parser.add_argument('--source', type=Path, default=Path(os.environ.get('QIHC_DISTRIBUTION_SOURCE', str(base/'results'/study.DEFAULT_SOURCE))))
    parser.add_argument('--rank', type=int, choices=[0, 1], default=0)
    args = parser.parse_args(); root = args.run.resolve()
    signal.signal(signal.SIGTERM, stop); signal.signal(signal.SIGINT, stop)
    if args.action == 'prepare':
        study.prepare_runtime(root, args.source)
    elif args.action == 'launch':
        return launch(root, study)
    elif args.action == 'worker':
        worker(root, args.rank, study)
    elif args.action == 'report':
        for phase in ('gate', 'formal'):
            study.summarize(root, phase)
        print(root/'formal/report.md')
    else:
        for name in ('status.json', 'gate_acceptance.json', 'gpu-0.json', 'gpu-1.json'):
            if (root/name).exists():
                print(name, json.dumps(s.load(root/name), ensure_ascii=False))
    return 0
