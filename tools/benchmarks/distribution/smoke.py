"""Tiny plumbing check: REAL CPU solvers, MOCK model replies. Not scientific results."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from qihc.generic.bounded import VERSION
from tools.benchmarks import comparison as c
from tools.benchmarks.distribution import experiment as e


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--subprocess-cpu', action='store_true', help='Also test the spawn-based CPU pool')
    args = p.parse_args()
    with tempfile.TemporaryDirectory(prefix='qihc-distribution-smoke-') as directory:
        root = Path(directory)
        model = root/'model'; model.mkdir()
        c.save(model/'config.json', {'fixture': True})
        spec = {'schema': VERSION, 'variables': [{'name': 'x0', 'lower': 0, 'upper': 1}, {'name': 'x1', 'lower': 0, 'upper': 1}],
                'objective': {'sense': 'max', 'linear': [3, 1], 'quadratic': []},
                'constraints': [{'id': 'choose_one', 'a': [1, 1], 'op': 'eq', 'rhs': 1}], 'logic': []}
        case = {'id': 'fixture', 'dataset': 'engineering_smoke_only',
                'question': 'Choose exactly one binary variable x0 or x1. Maximize 3*x0 + x1.',
                'gold': spec, 'reference_answer': 3, 'reference_status': 'exhaustive_fixture'}
        targets = ['objective', 'variable:x0', 'variable:x1', 'constraint:choose_one']
        raw = json.dumps({'model': spec, 'grounding': [{'target': t, 'quote': case['question'], 'explanation': 'Test fixture.'} for t in targets]})
        source, run = root/'old_read_only', root/'new'
        c.save(source/'cases.CPU_ONLY.json', [case])
        c.save(source/'manifest.json', {'cases_sha256': e.digest(source/'cases.CPU_ONLY.json'),
               'settings': {'model': str(model), 'model_files': e.model_identity(model)}})
        options = argparse.Namespace(from_run=source, run=run, budget=256, chains=2, seeds=1,
                                     solver_seconds=.2, limit=0, arms=list(e.ARMS))
        e.prepare(options)
        def deliver():
            for path in (run/'requests').glob('*.json'):
                if not (run/'responses'/path.name).exists():
                    c.save(run/'responses'/path.name, {'response': raw, 'status': 'MOCK_fixture_only',
                           'seconds': 0., 'output_tokens': 0, 'request_sha256': e.digest(path)})
        deliver()
        cfg = e.validate(run)['settings']
        if args.subprocess_cpu:
            subprocess.run([sys.executable, str(Path(e.__file__)), 'cpu', '--run', str(run), '--workers', '2', '--once'], check=True)
            if not list((run/'progress').glob('*/*.json')):
                raise RuntimeError('CPU pool failed to checkpoint')
        for _ in range(80):
            result = e.process_case(run, case, cfg)
            deliver()
            if result == 'complete':
                break
        report = e.summarize(run, [case], cfg)
        if report['status'] != 'complete':
            raise RuntimeError('Mocked-inference pipeline did not complete')
        e.prepare(options)  # Resume guard.
        if e.process_case(run, case, cfg) != 'complete':
            raise RuntimeError('Resume failed')
        print(f'PASS: {len(e.ARMS)} arms; real CPU Gibbs/Metropolis/CP-SAT; mock LLM; summary and resume. NOT a full-model/GPU validation.', flush=True)


if __name__ == '__main__':
    main()
