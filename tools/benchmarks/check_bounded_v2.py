#!/usr/bin/env python3
"""Small correctness/large-block integration checks, NOT formal benchmark results."""
import argparse
import itertools
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from qihc.generic.bounded import objective, violations
from qihc.generic.block_search import solve


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=2048)
    parser.add_argument('--seeds', type=int, default=3)
    args = parser.parse_args()
    if not 1 <= args.seeds <= 100:
        parser.error('seeds must be 1..100')
    small = json.loads(Path(__file__).with_name('bounded_v2_example.json').read_text())
    large = {'schema': small['schema'],
             'variables': [{'name': f'x{i}', 'lower': 0, 'upper': 1} for i in range(140)],
             'objective': {'sense': 'max', 'linear': [1]*140, 'quadratic': []},
             'constraints': [{'id': 'global_balance', 'a': [1]*140, 'op': 'eq', 'rhs': 70}], 'logic': []}
    candidates = itertools.product(*(range(v['lower'], v['upper']+1) for v in small['variables']))
    gold = max(float(objective(small, x)) for x in candidates if violations(small, x).sum() == 0)
    records = []
    for name, spec, optimum in [('integer_logic', small, gold), ('cross_block_140', large, 70.)]:
        for method, feedback in [('pbit', True), ('pbit', False), ('metropolis', True), ('uniform', False)]:
            for seed in range(args.seeds):
                result = solve(spec, method=method, feedback=feedback, seed=seed,
                               steps=args.steps, chains=8, block_size=128, block_updates=32)
                assert result['feasible'] == (violations(spec, result['x']).sum() == 0)
                result.update(case=name, reference_optimum=optimum)
                records.append(result)
                print(json.dumps({k: result[k] for k in ('case', 'method', 'feedback_enabled', 'seed', 'bits', 'feasible', 'objective')}), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({'scope': 'integration_checks_not_formal_evaluation',
                                      'records': records}, indent=2)+'\n', encoding='utf-8')


if __name__ == '__main__':
    main()
