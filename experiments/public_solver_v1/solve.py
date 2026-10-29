"""Bounded subprocess bundles: native crashes/hangs affect one case/backend only."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import os
import platform
from pathlib import Path
import signal
import subprocess
import sys
import time

from qihc.experiment_runtime import resilient_storage as s
from experiments.solver_comparison_v1.run import score_one, reference_one
from .common import default_root, validate, import_responses
from .report import report

STOP = False


def stop(*_):
    global STOP
    STOP = True


def score_path(root, identity, backend, seconds, seed):
    return root/'scores'/identity/f'{backend}-{seconds}-{seed}.json'


def error_score(root, case, identity, backend, seconds, seed, status, error):
    record = s.load(root/'models'/(identity+'.json'))
    s.save(score_path(root, identity, backend, seconds, seed), {
        'case_id': case['id'], 'dataset': case['dataset'], 'source_arm': record['arm'],
        'backend': backend, 'budget_seconds': seconds, 'seed': seed,
        'model_sha256': record.get('model_sha256'), 'response_sha256': record['response_sha256'],
        'status': status, 'verified_feasible': False, 'predicted_model_feasible': False,
        'error': error, 'raw_metrics': {}, 'scope': 'failure_preserved_in_denominator'})


def bundle(root, jobfile):
    job = s.load(jobfile); cfg = s.load(root/'manifest.json')['config']; case = job['case']
    frozen = s.load(root/'models'/(job['identity']+'.json'))
    if frozen['status'] == 'compiled' and frozen['model_sha256'] != s.value_digest(frozen['model']):
        raise ValueError('Frozen mathematics checksum mismatch')
    for seconds in cfg['seconds']:
        for seed in cfg['seeds']:
            path = score_path(root, job['identity'], job['backend'], seconds, seed)
            if path.exists():
                continue
            s.save(jobfile.with_suffix('.heartbeat.json'), {'time': time.time(), 'seconds': seconds})
            try:
                score_one(root, case, job['identity'], job['backend'], seconds, seed, cfg['chains'])
            except Exception as exc:
                error_score(root, case, job['identity'], job['backend'], seconds, seed, 'execution_error', repr(exc))
    if job['arm'] == 'canonical' and job['backend'] == 'cp_sat':
        path = root/'references'/(s.key(case['id'])+'.json')
        if not path.exists():
            s.save(jobfile.with_suffix('.heartbeat.json'), {'time': time.time(), 'seconds': cfg['reference_seconds']})
            try:
                reference_one(root, case, cfg['reference_seconds'])
            except Exception as exc:
                reference_error(root, case, repr(exc))


def reference_error(root, case, error):
    s.save(root/'references'/(s.key(case['id'])+'.json'), {'case_id': case['id'],
        'gold_sha256': s.value_digest(case['gold']), 'published_reference': case.get('reference_answer'),
        'result': {'feasible': False, 'optimality_proven': False, 'error': error},
        'scope': 'reference_failed_published_value_only'})


def missing(root, job, cfg):
    return (any(not score_path(root, job['identity'], job['backend'], sec, seed).exists()
                for sec in cfg['seconds'] for seed in cfg['seeds']) or
            job['arm'] == 'canonical' and job['backend'] == 'cp_sat' and
            not (root/'references'/(s.key(job['case']['id'])+'.json')).exists())


def fail_remaining(root, job, cfg, status, error):
    for sec in cfg['seconds']:
        for seed in cfg['seeds']:
            if not score_path(root, job['identity'], job['backend'], sec, seed).exists():
                error_score(root, job['case'], job['identity'], job['backend'], sec, seed, status, error)
    if job['arm'] == 'canonical' and job['backend'] == 'cp_sat':
        path = root/'references'/(s.key(job['case']['id'])+'.json')
        if not path.exists():
            reference_error(root, job['case'], error)


def run(root, track='both', workers=24, follow=False):
    m = validate(root, cpu=True); cfg = m['config']; cases = s.load(root/'cases.CPU_ONLY.json')
    if not 1 <= workers <= 64:
        raise ValueError('CPU workers must be 1..64')
    active = {}; finished = set(); start = time.monotonic(); last = 0.; imported_at = 0.; initial = None
    reporting = ThreadPoolExecutor(max_workers=1); report_future = None; last_report = 0.; summary = None
    tracks = cfg['source_arms'] if track == 'both' else [track]
    ordered = sorted(cases, key=lambda c: (c['original_id'].rsplit('_', 1)[-1], c['dataset']))
    with s.worker_lock(root/'solver.lock'):
        s.save(root/'launches'/f'cpu-{time.time_ns()}.json', {'time': time.time(),
            'platform': platform.platform(), 'processor': platform.processor(),
            'logical_cpus': os.cpu_count(), 'workers': workers, 'track': track,
            'threads_per_solver': 1, 'source_hash': m.get('source_hash')})
        try:
            while not STOP:
                if time.monotonic()-imported_at > 30:
                    import_responses(root, m, cases); imported_at = time.monotonic()
                for key, (proc, log, job, jf, started) in list(active.items()):
                    hb = jf.with_suffix('.heartbeat.json')
                    state = s.load(hb) if hb.exists() else {'time': started, 'seconds': 60}
                    timeout = time.time()-state['time'] > max(120, state['seconds']*5+60)
                    code = proc.poll()
                    if timeout or code is not None:
                        if timeout and code is None:
                            proc.kill(); proc.wait()
                        if timeout or code != 0 or missing(root, job, cfg):
                            fail_remaining(root, job, cfg, 'hard_timeout' if timeout else 'worker_crash',
                                           f'child exit={code}; timeout={timeout}; log={log.name}')
                        log.close(); del active[key]; finished.add(key)
                for c in ordered:
                    if len(active) >= workers:
                        break
                    for arm in tracks:
                        identity = s.key(c['id'], arm)
                        if not (root/'models'/(identity+'.json')).exists():
                            continue
                        for backend in cfg['backends']:
                            job = {'case': c, 'arm': arm, 'identity': identity, 'backend': backend}
                            key = identity+'-'+backend
                            if key in active or key in finished:
                                continue
                            if not missing(root, job, cfg):
                                finished.add(key); continue
                            jf = root/'jobs'/(key+'.json'); s.save(jf, job)
                            # Reset only our heartbeat; prior score checkpoints are immutable.
                            s.save(jf.with_suffix('.heartbeat.json'), {'time': time.time(), 'seconds': 60})
                            logfile = root/'logs'/(key+'.log'); logfile.parent.mkdir(exist_ok=True)
                            log = logfile.open('ab', buffering=0)
                            env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
                            proc = subprocess.Popen([sys.executable, '-u', '-m', 'experiments.public_solver_v1.solve',
                                'bundle', '--run', str(root), '--job', str(jf)], stdout=log, stderr=subprocess.STDOUT, env=env)
                            active[key] = proc, log, job, jf, time.time()
                            if len(active) >= workers:
                                break
                        if len(active) >= workers:
                            break
                if report_future is not None and report_future.done():
                    summary = report_future.result(); report_future = None
                if not active:
                    if report_future is not None:
                        report_future.result(); report_future = None
                    summary = report(root)  # Final fresh snapshot, after all children finished.
                elif report_future is None and time.monotonic()-last_report >= cfg['report_interval_seconds']:
                    # Reporting 189k rows must not block dispatch/watchdogs on a shared filesystem.
                    report_future = reporting.submit(report, root); last_report = time.monotonic()
                if time.monotonic()-last >= 30 or not active:
                    completed = summary['completed_scores'] if summary else None
                    if initial is None and completed is not None:
                        initial = completed
                    ready = len(list((root/'models').glob('*.json')))
                    state = {'status': 'running' if active else 'partial_waiting_for_models',
                        'track': track, 'active': len(active), 'workers': workers, 'models_ready': ready,
                        'completed_scores_as_of_report': completed,
                        'expected_scores': len(cases)*len(cfg['source_arms'])*len(cfg['backends'])*len(cfg['seconds'])*len(cfg['seeds']),
                        'seconds_this_launch': time.monotonic()-start, 'time': time.time(),
                        'eta_seconds': ((time.monotonic()-start)/(completed-initial)*(summary['expected_scores']-completed)
                                        if completed is not None and initial is not None and completed > initial else None),
                        'eta_scope': 'observed terminal scoring throughput; excludes future LLM inference; failures count as terminal'}
                    s.save(root/'solver_status.json', state); print(state, flush=True); last = time.monotonic()
                if not active:
                    if not follow or track == 'canonical' or ready == len(cases)*2:
                        status = ('complete' if ready == len(cases)*2 and summary['all_terminal'] else
                                  'track_complete' if track != 'both' and summary['tracks'][track]['status'] == 'complete'
                                  else 'partial_waiting_for_models')
                        s.save(root/'solver_status.json', dict(state, status=status))
                        return 0
                    # Don't hold an idle CPU allocation indefinitely waiting for GPU jobs.
                    print('No available work. Resume this command when more model responses arrive.', flush=True)
                    return 0
                time.sleep(.5)
        except Exception as exc:
            s.save(root/'solver_status.json', {'status': 'infrastructure_failed', 'error': repr(exc), 'time': time.time()})
            raise
        finally:
            for proc, log, job, jf, started in active.values():
                if proc.poll() is None:
                    proc.terminate()
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill(); proc.wait()
                log.close()
            if STOP:
                s.save(root/'solver_status.json', {'status': 'stopped_checkpointed', 'time': time.time()})
            reporting.shutdown(wait=True)
    return 130


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('action', choices=['run', 'bundle', 'report'])
    p.add_argument('--run', type=Path, default=default_root()); p.add_argument('--job', type=Path)
    p.add_argument('--track', choices=['canonical', 'llm_typed_flat', 'both'], default='both')
    p.add_argument('--workers', type=int, default=24); p.add_argument('--follow', action='store_true')
    a = p.parse_args(); signal.signal(signal.SIGTERM, stop); signal.signal(signal.SIGINT, stop)
    if a.action == 'bundle':
        bundle(a.run, a.job)
    elif a.action == 'report':
        report(a.run)
    else:
        raise SystemExit(run(a.run, a.track, a.workers, a.follow))
