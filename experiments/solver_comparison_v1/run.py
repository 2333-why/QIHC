"""CPU-only streaming evaluator of frozen LLM outputs; never writes source run."""
import argparse
from collections import Counter
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import time

from qihc.experiment_runtime import storage as s
from qihc.modeling import protocol, typed_sources
from qihc.generic.solver_baselines.registry import BACKENDS, solve
from qihc.generic.solver_baselines.isolation import BackendPools

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
STOP = False


def stop(*_):
    global STOP
    STOP = True


def fingerprint():
    h = hashlib.sha256()
    for folder in ('qihc/generic', 'qihc/modeling', 'qihc/experiment_runtime', 'experiments/solver_comparison_v1'):
        for p in sorted((ROOT/folder).rglob('*')):
            if p.suffix in ('.py', '.json', '.sh'):
                h.update(p.relative_to(ROOT).as_posix().encode()); h.update(p.read_bytes().replace(b'\r\n', b'\n'))
    return h.hexdigest()


def dependencies():
    versions = {n: importlib.metadata.version(n) for n in ('numpy', 'ortools', 'highspy', 'pyscipopt')}
    if versions['highspy'] != '1.15.1' or versions['pyscipopt'] != '6.3.0':
        raise ValueError('Run E002 CPU preparation: pinned highspy 1.15.1 and pyscipopt 6.3.0 required')
    return versions


def prepare(root, source):
    if root.resolve() == source.resolve() or source.resolve() in root.resolve().parents:
        raise ValueError('E002 output must be separate from source experiment')
    cfg = s.load(HERE/'config.json'); src = s.load(source/'manifest.json')
    if (not cfg['backends'] or len(set(cfg['backends'])) != len(cfg['backends'])
            or any(b not in BACKENDS for b in cfg['backends'])
            or not cfg['source_arms'] or len(set(cfg['source_arms'])) != len(cfg['source_arms'])
            or not cfg['seeds'] or len(set(cfg['seeds'])) != len(cfg['seeds'])
            or any(type(seed) is not int or not 0 <= seed < 2**31 for seed in cfg['seeds'])
            or not cfg['seconds'] or len(set(cfg['seconds'])) != len(cfg['seconds'])
            or any(not math.isfinite(v) or v <= 0 for v in [*cfg['seconds'], cfg['reference_seconds']])
            or type(cfg['cpu_workers']) is not int or not 1 <= cfg['cpu_workers'] <= 64
            or type(cfg['chains']) is not int or not 1 <= cfg['chains'] <= 256
            or not 1 <= cfg['report_interval_seconds'] <= 3600):
        raise ValueError('Invalid E002 backend/seed/budget/parallel configuration')
    if any(a not in src['config']['arms'] for a in cfg['source_arms']):
        raise ValueError('Source must contain every predeclared source arm')
    cases_path = source/'formal/cases.CPU_ONLY.json'
    if s.digest(cases_path) != src['phase_cases_sha256']['formal']:
        raise ValueError('Source case identity mismatch')
    cases = s.load(cases_path)
    if len({c['id'] for c in cases}) != len(cases):
        raise ValueError('Duplicate cases')
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    manifest = {'experiment_id': 'E002-1.1', 'config': cfg, 'source_run': str(source.resolve()),
        'source_manifest_sha256': s.digest(source/'manifest.json'), 'source_commit': src['code_commit'],
        'source_model': src['model'], 'source_cases_sha256': s.digest(cases_path),
        'source_hash': fingerprint(), 'code_commit': commit, 'dependencies': dependencies(),
        'cases': len(cases), 'scope': 'Each source arm is a separate stratum. All solvers use identical frozen predicted mathematics.'}
    if (root/'manifest.json').exists() and s.load(root/'manifest.json') != manifest:
        raise ValueError('Source/config/code/dependencies changed. Preserve old run; choose a new result path.')
    s.save(root/'manifest.json', manifest)
    s.save(root/'cases.CPU_ONLY.json', cases)
    return manifest


def import_models(root, manifest, cases):
    source = Path(manifest['source_run']); src = s.load(source/'manifest.json')
    if s.digest(source/'manifest.json') != manifest['source_manifest_sha256']:
        raise ValueError('Source manifest changed during evaluation')
    count = 0
    for case in cases:
        for arm in manifest['config']['source_arms']:
            identity = s.key(case['id'], arm)
            response = source/'formal/responses'/(identity+'.json')
            if not response.exists():
                continue
            req = source/'formal/requests'/(identity+'.json'); rh = s.digest(response)
            saved = root/'models'/(identity+'.json')
            request_hash = s.digest(req)
            if request_hash != src['request_hashes'][identity]:
                raise ValueError('Source request identity mismatch')
            if saved.exists():
                if (s.load(saved)['response_sha256'] != rh or
                        s.load(saved)['request_sha256'] != request_hash):
                    raise ValueError('A frozen response changed; refusing to mix results')
                count += 1; continue
            row = s.load(response)
            if row['request_sha256'] != s.digest(req) or s.digest(req) != src['request_hashes'][identity]:
                raise ValueError('Source request/response identity mismatch')
            request = s.load(req)
            if request.get('question') is not None and request['question'] != case['question']:
                raise ValueError('Request original question mismatch')
            record = {'case_id': case['id'], 'dataset': case['dataset'], 'arm': arm,
                'response_sha256': rh, 'request_sha256': s.digest(req), 'response': row['response'],
                'status': 'invalid_model', 'model': None,
                'generation_cost': {k: row.get(k) for k in ('input_tokens', 'output_tokens', 'seconds', 'generation_attempts')}}
            try:
                if row.get('mathematical_validity') != 'valid':
                    raise ValueError(row.get('validation_error', row.get('status')))
                inspect = typed_sources.inspect if request['contract'] == 'typed' else protocol.inspect
                model, reason, audits = inspect(row['response'], case['question'])
                if model is None:
                    raise ValueError(reason)
                record.update(status='compiled', model=model, audits=audits, model_sha256=s.value_digest(model))
            except (ValueError, TypeError, KeyError, OverflowError) as exc:
                record['error'] = str(exc)
            s.save(saved, record); count += 1
    return count


def score_one(root, case, identity, backend, seconds, seed, chains):
    from qihc.experiment_runtime.evaluation import evaluate
    record = s.load(root/'models'/(identity+'.json'))
    result = {'case_id': case['id'], 'dataset': case['dataset'], 'source_arm': record['arm'],
        'backend': backend, 'budget_seconds': seconds, 'seed': seed,
        'model_sha256': record.get('model_sha256'), 'response_sha256': record['response_sha256'],
        'status': 'invalid_model', 'verified_feasible': False, 'predicted_model_feasible': False}
    if record['status'] == 'compiled':
        raw = solve(record['model'], backend, seconds, seed, chains)
        result.update(evaluate(case, record['model'], raw))
        result.update(raw_metrics={k: v for k, v in raw.items() if k != 'x'},
                      status='unsupported' if raw.get('status') == 'UNSUPPORTED' else 'searched',
                      backend=backend, scope='shared_predicted_model')
    else:
        result['error'] = record.get('error')
    path = root/'scores'/identity/f'{backend}-{seconds}-{seed}.json'
    s.save(path, result)
    return str(path)


def reference_one(root, case, seconds):
    # Independent CPU-only evaluator. Never passed to source LLM or solver input.
    raw = solve(case['gold'], 'cp_sat', seconds, 0)
    s.save(root/'references'/(s.key(case['id'])+'.json'), {'case_id': case['id'], 'gold_sha256': s.value_digest(case['gold']),
           'published_reference': case.get('reference_answer'), 'result': raw,
           'scope': 'CPU-only original-model reference; optimal only if proven'})


def tasks(root, cases, cfg):
    # Round-robin by budget, backend, seed, then dataset/case; no method can
    # consume the entire allocation before its comparators receive a turn.
    ordered = sorted(cases, key=lambda c: (s.key(c['id']), c['dataset']))
    for c in ordered:
        path = root/'references'/(s.key(c['id'])+'.json')
        if not path.exists():
            yield str(path), reference_one, (root, c, cfg['reference_seconds'])
    for seconds in cfg['seconds']:
        for seed in cfg['seeds']:
            for c in ordered:
                for arm in cfg['source_arms']:
                    identity = s.key(c['id'], arm)
                    if not (root/'models'/(identity+'.json')).exists():
                        continue
                    for backend in cfg['backends']:
                        path = root/'scores'/identity/f'{backend}-{seconds}-{seed}.json'
                        if not path.exists():
                            yield str(path), score_one, (root, c, identity, backend, seconds, seed, cfg['chains'])


def run(root, source, follow=False):
    from .report import report
    with s.worker_lock(root/'launch.lock'):
        manifest = prepare(root, source); cfg = manifest['config']
        cases = s.load(root/'cases.CPU_ONLY.json'); active = {}; start = time.monotonic(); launched_at = time.time(); last = 0
        try:
            # Dependency integration smoke is mandatory before the large queue.
            from .smoke import check
            check()
            with BackendPools(tuple(dict.fromkeys(['cp_sat', *cfg['backends']])), cfg['cpu_workers']) as pools:
                while not STOP:
                    for name, (backend, f) in list(active.items()):
                        if f.done():
                            f.result(); del active[name]
                    busy = Counter(backend for backend, _ in active.values())
                    imported = import_models(root, manifest, cases)
                    for name, fun, args in tasks(root, cases, cfg):
                        if len(active) >= cfg['cpu_workers']:
                            break
                        backend = 'cp_sat' if fun is reference_one else args[3]
                        if name not in active and busy[backend] < pools.capacity[backend]:
                            active[name] = (backend, pools.submit(backend, fun, *args))
                            busy[backend] += 1
                    if time.monotonic()-last >= cfg['report_interval_seconds'] or not active:
                        summary = report(root)
                        state = {'status': summary['status'], 'imported_models': imported,
                            'expected_models': len(cases)*len(cfg['source_arms']), 'active_cpu': len(active),
                            'active_by_backend': dict(busy), 'worker_capacity_by_backend': pools.capacity,
                            'completed_scores': summary['completed_scores'], 'expected_scores': summary['expected_scores'],
                            'seconds_this_launch': time.monotonic()-start, 'time': time.time()}
                        s.save(root/'status.json', state); print(json.dumps(state), flush=True); last = time.monotonic()
                        if summary['status'] == 'complete':
                            return 0
                    if not active and not follow:
                        break
                    # In follow mode wait for source output. Never occupy GPUs artificially.
                    if not active and (source/'status.json').exists():
                        source_state = s.load(source/'status.json'); state = source_state.get('status')
                        if state == 'complete' or (source_state.get('time', 0) >= launched_at and state in ('failed', 'stopped_checkpointed')):
                            raise RuntimeError('Source ended before all E002 models arrived; checkpoints retained: '+str(state))
                    time.sleep(2)
        except BaseException as exc:
            s.save(root/'status.json', {'status': 'failed', 'error': repr(exc), 'time': time.time()})
            raise
        summary = report(root)
        s.save(root/'status.json', {'status': 'stopped_checkpointed' if STOP else 'partial_waiting_for_source',
               'completed_scores': summary['completed_scores'], 'expected_scores': summary['expected_scores'], 'time': time.time()})
        return 130 if STOP else 0


def main():
    base = Path(os.environ.get('QIHC_BASE', '/inspire/hdd/global_user/czxs25210022/why'))
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['prepare', 'run', 'report'])
    p.add_argument('--run', type=Path, default=Path(os.environ.get('QIHC_E002_RUN', str(base/'results/E002-1.1_solver_comparison'))))
    p.add_argument('--source', type=Path, default=Path(os.environ.get('QIHC_PROMPT_V3_RUN', str(base/'results/E001-1.3_prompt_fixed'))))
    p.add_argument('--follow', action='store_true')
    a = p.parse_args(); root, source = a.run.resolve(), a.source.resolve()
    signal.signal(signal.SIGTERM, stop); signal.signal(signal.SIGINT, stop)
    if a.action == 'prepare':
        with s.worker_lock(root/'launch.lock'):
            prepare(root, source)
    elif a.action == 'report':
        from .report import report
        report(root)
    else:
        return run(root, source, a.follow)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
