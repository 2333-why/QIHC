#!/usr/bin/env python3
"""No-training comparison queue: CPU evaluates, GPU only generates, all work resumable."""
from __future__ import annotations
import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.benchmarks.download_public import LOCK, audit, atomic_write, json_bytes
from tools.benchmarks.public_simulation import model_identity
from qihc.generic.bounded import VERSION, objective, violations
from qihc.generic.public_modeling import prompt, parse, answer_agreement
from qihc.generic.factor_search import solve as sample
from qihc.generic.bounded_reference import solve as exact
from qihc.generic.benchmark import make_case, parse_object

STOP = False


def stop(signum, frame):
    global STOP
    STOP = True
    print(f'SIGNAL {signum}: finish current atomic checkpoint then stop', flush=True)


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def save(path, value):
    atomic_write(path, json_bytes(value))


def code_hash():
    h = hashlib.sha256()
    files = list((ROOT/'qihc/generic').glob('*.py')) + list(Path(__file__).parent.glob('*.py'))
    files += list(Path(__file__).parent.glob('*.json')) + [ROOT/'qihc/problems/cvrp/direct_llm.py']
    for p in sorted(files):
        h.update(p.relative_to(ROOT).as_posix().encode())
        h.update(p.read_bytes().replace(b'\r\n', b'\n'))
    return h.hexdigest()


def key(cid, phase):
    return hashlib.sha256((cid+'\0'+phase).encode()).hexdigest()


@contextmanager
def worker_lock(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a+b') as handle:
        if os.name == 'nt':
            import msvcrt
            handle.write(b'0'); handle.flush(); handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        # OS releases lock on normal exit, SIGTERM, or SIGKILL.
        yield


def enqueue(run, case, phase, text):
    identity = key(case['id'], phase)
    row = {'id': identity, 'case_id': case['id'], 'phase': phase, 'prompt': text}
    path = run/'requests'/f'{identity}.json'
    if path.exists():
        if load(path) != row:
            raise ValueError('Existing request differs')
    else:
        save(path, row)


def modeling_prompt(case):
    p = prompt(case['question'])
    if case.get('gold'):
        p += '\nOriginal decision variable names, in the problem index order: '+json.dumps(
            [v['name'] for v in case['gold']['variables']])+'. Preserve these names; do not add decision variables.'
    return p


def direct_prompt(case):
    if case.get('gold'):
        instruction = 'Return JSON only {"x":[0,1,...]} in the original variable index order.'
    else:
        instruction = 'Return JSON only {"objective":number,"solution":string}. Give the optimal objective in the original units and describe the complete decisions.'
    return 'Solve the constrained optimization problem directly, without external tools. '+instruction+'\n'+case['question']


def prepare(args):
    import math
    if (not 0 <= args.per_dataset <= 100000 or not 1 <= args.synthetic_seeds <= 100 or
            not 1 <= args.steps <= 1000000 or not 1 <= args.chains <= 256 or not 1 <= args.seeds <= 20 or
            not math.isfinite(args.solver_seconds) or args.solver_seconds <= 0):
        raise ValueError('Invalid preparation budget')
    settings = {'per_dataset': args.per_dataset, 'synthetic_seeds': args.synthetic_seeds,
                'steps': args.steps, 'chains': args.chains, 'seeds': args.seeds,
                'solver_seconds': args.solver_seconds, 'source': code_hash(),
                'model': str(args.model.resolve()), 'model_files': model_identity(args.model),
                'public_revision': json.loads(LOCK.read_text())['revision']}
    if getattr(args, 'cases_file', None):
        settings['external_cases_sha256'] = hashlib.sha256(args.cases_file.read_bytes()).hexdigest()
    if (args.run/'manifest.json').exists():
        old = load(args.run/'manifest.json')
        if old['settings'] != settings:
            raise ValueError('Preparation changed: choose a new comparison run directory')
        validate_run(args.run)
        print('Preparation already complete', flush=True); return
    if any(v < 1 for v in (args.synthetic_seeds, args.steps, args.chains, args.seeds)) or args.per_dataset < 0:
        raise ValueError('Invalid preparation budgets')
    external = getattr(args, 'cases_file', None)
    if external:
        public, answers = [], []
    else:
        _, public, answers = audit(json.loads(LOCK.read_text()), args.data/'raw')
    labels = {r['id']: r['reference_answer'] for r in answers}
    cases = load(external) if external else []
    for dataset in sorted({r['dataset'] for r in public}):
        rows = [r for r in public if r['dataset'] == dataset]
        count = min(args.per_dataset or len(rows), len(rows))
        indices = [0] if count == 1 else [i*(len(rows)-1)//(count-1) for i in range(count)]
        cases.extend({'id': rows[i]['id'], 'dataset': dataset, 'question': rows[i]['question'],
                      'reference_answer': labels[rows[i]['id']], 'gold': None} for i in indices)
    for family in ('knapsack', 'vertex_cover', 'assignment', 'balanced_cut'):
        sizes = (3, 5, 7) if family == 'assignment' else (16, 24, 40) if family == 'vertex_cover' else (16, 32, 64)
        for size in sizes:
            for seed in range(args.synthetic_seeds):
                c = make_case(family, size, 20261022+seed, 'test')
                g = c['gold']
                gold = {'schema': VERSION, 'variables': [{'name': f'x{i}', 'lower': 0, 'upper': 1} for i in range(g['n'])],
                        'objective': g['objective'], 'constraints': g['constraints'], 'logic': []}
                cases.append({'id': c['id'], 'dataset': 'synthetic_'+family, 'question': c['description'],
                              'gold': gold, 'reference_answer': None})
    args.run.mkdir(parents=True, exist_ok=True)
    save(args.run/'cases.CPU_ONLY.json', cases)
    for c in cases:
        enqueue(args.run, c, 'model', modeling_prompt(c))
        enqueue(args.run, c, 'direct', direct_prompt(c))
    save(args.run/'manifest.json', {'settings': settings, 'cases': len(cases),
         'cases_sha256': hashlib.sha256((args.run/'cases.CPU_ONLY.json').read_bytes()).hexdigest(),
         'training': False, 'ablations': False,
         'arms': ['direct_llm', 'llm_cp_sat', 'solver_feedback_cp_sat', 'qihc',
                  'gold_cp_sat', 'gold_simulated_annealing', 'gold_random']})
    print(f'PREPARED {len(cases)} cases; CPU_ONLY gold/answers never read by GPU worker', flush=True)


def validate_run(run):
    manifest = load(run/'manifest.json')
    if manifest['settings']['source'] != code_hash():
        raise ValueError('Source differs from prepared run; do not mix experiments')
    if hashlib.sha256((run/'cases.CPU_ONLY.json').read_bytes()).hexdigest() != manifest['cases_sha256']:
        raise ValueError('Case manifest changed')
    return manifest


def evaluate(case, spec, result):
    out = dict(result)
    out['predicted_model_feasible'] = bool(result.get('feasible', False))
    if case['gold']:
        try:
            if result.get('x') is None:
                raise ValueError('No candidate')
            if len(result['x']) != len(spec['variables']):
                raise ValueError('Candidate dimension differs from model')
            values = {v['name']: x for v, x in zip(spec['variables'], result['x'])}
            if set(values) != {v['name'] for v in case['gold']['variables']}:
                raise ValueError('Original variable mapping differs')
            x = [values[v['name']] for v in case['gold']['variables']]
            out['verified_feasible'] = bool(violations(case['gold'], x).sum() == 0)
            out['verified_objective'] = float(objective(case['gold'], x))
        except (ValueError, TypeError, KeyError) as exc:
            out.update(verified_feasible=False, verification_error=str(exc))
    else:
        out['verified_feasible'] = None
        out['reference_answer_agreement'] = answer_agreement(result, case['reference_answer']) if result.get('objective') is not None else False
    return out


def solve_model(case, raw, backend, cfg):
    try:
        spec, reason = parse(raw, case['question'])
        if spec is None:
            return {'status': 'unsupported', 'diagnostic': reason, 'runs': []}
    except (ValueError, TypeError, KeyError) as exc:
        return {'status': 'invalid_model', 'diagnostic': str(exc), 'runs': []}
    results = [exact(spec, cfg['solver_seconds'])] if backend == 'cp_sat' else [
        sample(spec, seed=s, steps=cfg['steps'], chains=cfg['chains']) for s in range(cfg['seeds'])]
    first = results[0]
    # Never include gold/answer-agreement in the feedback diagnostic.
    diagnostic = {k: first.get(k) for k in ('status', 'x', 'feasible', 'objective', 'violation')}
    if first.get('x') is not None:
        diagnostic['constraint_violations'] = [
            {'id': c['id'], 'violation': float(v)} for c, v in zip(
                spec['constraints']+spec['logic'], violations(spec, first['x'])[1:])]
    return {'status': 'searched', 'diagnostic': diagnostic,
            'runs': [evaluate(case, spec, r) for r in results]}


def review_text(case, raw, diagnostic):
    return (modeling_prompt(case)+'\nAudit and correct the previous model against the original text. '
            'Preserve every requirement; do not drop rules to make search easier. '
            'Search failure is not proof of infeasibility. Return the full model and grounding envelope. '
            '\nPrevious response:\n'+raw+'\nSolver diagnostics (not ground truth):\n'+json.dumps(diagnostic))


def process_case(run, case, cfg):
    folder = run/'progress'/key(case['id'], 'case')
    def response(phase):
        p = run/'responses'/f"{key(case['id'], phase)}.json"
        if not p.exists():
            return None
        value = load(p)
        req = run/'requests'/p.name
        if value['request_sha256'] != hashlib.sha256(req.read_bytes()).hexdigest():
            raise ValueError('Response/request mismatch')
        return value
    # Write/checkpoint each expensive arm independently.
    initial = response('model')
    if initial is not None:
        for backend, phase in [('cp_sat', 'solver_feedback'), ('pbit', 'qihc')]:
            path = folder/(backend+'_initial.json')
            if not path.exists():
                save(path, solve_model(case, initial['response'], backend, cfg))
            enqueue(run, case, phase, review_text(case, initial['response'], load(path)['diagnostic']))
            revised = response(phase)
            path = folder/(phase+'.json')
            if revised is not None and not path.exists():
                save(path, solve_model(case, revised['response'], backend, cfg))
    direct = response('direct')
    if direct is not None and not (folder/'direct.json').exists():
        try:
            value = parse_object(direct['response'])
            if case['gold']:
                x = value['x']
                if set(value) != {'x'} or not isinstance(x, list) or len(x) != len(case['gold']['variables']) or any(type(v) is not int or v not in (0, 1) for v in x):
                    raise ValueError('Invalid direct binary solution')
                result = evaluate(case, case['gold'], {'x': x, 'objective': float(objective(case['gold'], x)), 'feasible': bool(violations(case['gold'], x).sum() == 0)})
            else:
                import math
                if set(value) != {'objective', 'solution'} or type(value['objective']) not in (int, float) or not math.isfinite(value['objective']) or not isinstance(value['solution'], str):
                    raise ValueError('Invalid direct answer')
                result = {'verified_feasible': None, 'objective': value['objective'],
                          'reference_answer_agreement': answer_agreement({'feasible': True, 'objective': value['objective']}, case['reference_answer']),
                          'solution': value['solution'], 'scope': 'unverified_objective_answer_only'}
            record = {'status': 'answered', 'runs': [result]}
        except (ValueError, TypeError, KeyError) as exc:
            record = {'status': 'invalid_answer', 'error': str(exc), 'runs': []}
        save(folder/'direct.json', record)
    needed = ['direct.json', 'cp_sat_initial.json', 'pbit_initial.json', 'solver_feedback.json', 'qihc.json']
    if not all((folder/name).exists() for name in needed):
        return False
    final = run/'final'/f"{key(case['id'], 'case')}.json"
    if not final.exists():
        arms = {'direct_llm': load(folder/'direct.json'), 'llm_cp_sat': load(folder/'cp_sat_initial.json'),
                'solver_feedback_cp_sat': load(folder/'solver_feedback.json'), 'qihc': load(folder/'qihc.json')}
        if case['gold']:
            for arm, method in [('gold_cp_sat', None), ('gold_simulated_annealing', 'metropolis'), ('gold_random', 'uniform')]:
                path = folder/(arm+'.json')
                if not path.exists():
                    results = [exact(case['gold'], cfg['solver_seconds'])] if method is None else [
                        sample(case['gold'], method=method, feedback=False, seed=s,
                               steps=cfg['steps'], chains=cfg['chains']) for s in range(cfg['seeds'])]
                    save(path, {'status': 'gold_model_solver_only', 'runs': [evaluate(case, case['gold'], r) for r in results]})
                arms[arm] = load(path)
        usage = {phase: {k: v for k, v in response(phase).items() if k not in ('response', 'request_sha256')}
                 for phase in ('model', 'direct', 'solver_feedback', 'qihc')}
        save(final, {'id': case['id'], 'dataset': case['dataset'], 'arms': arms, 'generation': usage,
                     'initial_search': {'cp_sat': load(folder/'cp_sat_initial.json'), 'pbit': load(folder/'pbit_initial.json')}})
    return True


def summarize(run, cases):
    rows = [load(p) for p in sorted((run/'final').glob('*.json'))]
    groups = {}
    for dataset in sorted({c['dataset'] for c in cases}):
        selected = [r for r in rows if r['dataset'] == dataset]
        groups[dataset] = {'expected': sum(c['dataset'] == dataset for c in cases), 'completed': len(selected), 'arms': {}}
        verified = all(c.get('gold') is not None for c in cases if c['dataset'] == dataset)
        arm_names = ['direct_llm', 'llm_cp_sat', 'solver_feedback_cp_sat', 'qihc']
        if verified: arm_names += ['gold_cp_sat', 'gold_simulated_annealing', 'gold_random']
        for arm in arm_names:
            total = len(selected)
            groups[dataset]['arms'][arm] = {metric: sum(
                sum(v.get(metric) is True for v in r['arms'][arm]['runs'])/len(r['arms'][arm]['runs'])
                if r['arms'][arm]['runs'] else 0 for r in selected)/total if total else None
                for metric in (['verified_feasible'] if verified else ['reference_answer_agreement'])}
            solved = [v for r in selected for v in r['arms'][arm]['runs'] if v.get('verified_feasible') is True]
            stats = groups[dataset]['arms'][arm]
            stats['mean_verified_objective_feasible_only'] = sum(v['verified_objective'] for v in solved)/len(solved) if solved else None
            gaps = []
            for r in selected:
                c = next(c for c in cases if c['id'] == r['id'])
                if not c.get('reference_status'): continue
                ref = c['reference_answer']
                direction = 1 if c['gold']['objective']['sense'] == 'min' else -1
                seed_gaps = [direction*(v['verified_objective']-ref)/max(abs(ref), 1e-9)
                             for v in r['arms'][arm]['runs'] if v.get('verified_feasible') is True]
                if seed_gaps: gaps.append(sum(seed_gaps)/len(seed_gaps))
            stats['mean_signed_gap_to_published_reference_feasible_instances'] = sum(gaps)/len(gaps) if gaps else None
            phases = {'direct_llm': ['direct'], 'llm_cp_sat': ['model'],
                      'solver_feedback_cp_sat': ['model', 'solver_feedback'], 'qihc': ['model', 'qihc']}.get(arm, [])
            stats['llm_calls'] = len(phases)*len(selected)
            stats['llm_seconds'] = sum(r['generation'][p].get('seconds', 0) for r in selected for p in phases)
            stats['output_tokens'] = sum(r['generation'][p].get('output_tokens', 0) for r in selected for p in phases)
            stats['search_seconds_all_seeds'] = sum(v.get('seconds', 0) for r in selected for v in r['arms'][arm]['runs'])
            if arm in ('solver_feedback_cp_sat', 'qihc'):
                first = 'cp_sat' if arm == 'solver_feedback_cp_sat' else 'pbit'
                stats['search_seconds_all_seeds'] += sum(v.get('seconds', 0) for r in selected for v in r['initial_search'][first]['runs'])
    save(run/'summary.json', {'status': 'complete' if len(rows) == len(cases) else 'partial',
                             'completed': len(rows), 'expected': len(cases), 'datasets': groups,
                             'warning': 'Public objective agreement is not independently verified feasibility; full per-stage costs in final/.'})


def cpu(args):
    manifest = validate_run(args.run)
    cases = load(args.run/'cases.CPU_ONLY.json')
    cfg = manifest['settings']
    with worker_lock(args.run/'cpu.lock'):
        while not STOP:
            complete = 0
            for case in cases:
                if STOP: break
                final = args.run/'final'/f"{key(case['id'], 'case')}.json"
                if final.exists() or process_case(args.run, case, cfg):
                    complete += 1
            summarize(args.run, cases)
            print(f'CPU completed={complete}/{len(cases)}', flush=True)
            if complete == len(cases) or args.once: break
            time.sleep(2)


def gpu(args):
    if not 1 <= args.idle_exit <= 600:
        raise ValueError('idle-exit must be 1..600 seconds')
    manifest = load(args.run/'manifest.json')  # Do NOT read cases.CPU_ONLY.json.
    cfg = manifest['settings']
    if cfg['source'] != code_hash() or model_identity(Path(cfg['model'])) != cfg['model_files']:
        raise ValueError('GPU source/model differs from CPU preparation')
    from qihc.problems.cvrp.direct_llm import LocalDirectLLMGenerator
    with worker_lock(args.run/'gpu.lock'):
        generator = None
        idle = time.monotonic()
        while not STOP:
            pending = [p for p in sorted((args.run/'requests').glob('*.json')) if not (args.run/'responses'/p.name).exists()]
            if not pending:
                if (args.run/'summary.json').exists() and load(args.run/'summary.json')['status'] == 'complete':
                    print('All comparison inference complete', flush=True); break
                if time.monotonic()-idle > args.idle_exit:
                    save(args.run/'gpu_status.json', {'status': 'waiting_for_cpu_resubmit_same_command', 'safe_checkpoint': True})
                    print('No ready inference work: exiting GPU allocation; resume same command after CPU advances.', flush=True)
                    break
                time.sleep(1); continue
            if generator is None:
                generator = LocalDirectLLMGenerator(cfg['model'], device='auto', temperature=0,
                                                     max_input_tokens=16384, max_new_tokens=8192)
            for path in pending:
                if STOP: break
                request = load(path)
                print(f"INFER {request['case_id']} {request['phase']}", flush=True)
                started = time.perf_counter()
                try:
                    response = generator.generate(request['prompt'])
                    result = {'response': response, 'status': 'generated', **generator.last_usage}
                except ValueError as exc:
                    result = {'response': '', 'status': 'generation_error', 'error': str(exc)}
                result.update(seconds=time.perf_counter()-started, request_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
                save(args.run/'responses'/path.name, result)
                save(args.run/'gpu_status.json', {'status': 'running', 'last_case': request['case_id'], 'phase': request['phase']})
                idle = time.monotonic()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='stage', required=True)
    p = sub.add_parser('prepare')
    for name in ('run', 'data', 'model'): p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--per-dataset', type=int, default=40)
    p.add_argument('--cases-file', type=Path, help='Audited CPU-only cases; replaces generic public datasets')
    p.add_argument('--synthetic-seeds', type=int, default=5)
    p.add_argument('--steps', type=int, default=4096)
    p.add_argument('--chains', type=int, default=16)
    p.add_argument('--seeds', type=int, default=3)
    p.add_argument('--solver-seconds', type=float, default=5)
    p = sub.add_parser('cpu'); p.add_argument('--run', type=Path, required=True); p.add_argument('--once', action='store_true')
    p = sub.add_parser('gpu'); p.add_argument('--run', type=Path, required=True); p.add_argument('--idle-exit', type=float, default=45)
    args = parser.parse_args()
    signal.signal(signal.SIGTERM, stop); signal.signal(signal.SIGINT, stop)
    {'prepare': prepare, 'cpu': cpu, 'gpu': gpu}[args.stage](args)


if __name__ == '__main__': main()
