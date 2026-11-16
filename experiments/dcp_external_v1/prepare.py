"""Prepare and audit a frozen DCP-Bench Open subset on a CPU node."""
import argparse
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

from experiments.pbit_joint_v1.search import episode
from qihc.experiment_runtime.storage import save
from qihc.generic.bounded import validate
from qihc.generic.solver_baselines.registry import solve

RELEASE = 'https://github.com/DCP-Bench/DCP-Bench-Open/releases/download/v0.3.0/dcp-bench-open.jsonl'
SHA256 = 'bf4f50f435c03a5600194306937e37039cdfa2e002dfa1c6548cd815c8601385'
KNAPSACK = 'knapsack'
CLIQUE = 'csplib_074_maximum_clique'
BUS = 'csplib_022_bus_driver_scheduling'
FIELDS = {KNAPSACK: ('values', 'weights', 'capacity'), CLIQUE: ('n', 'adj'),
          BUS: ('num_work', 'num_shifts', 'shifts')}
SEARCH = dict(seed=17, chains=32, window_steps=128, decisions=4)


def formulation(family, data):
    constraints = []

    def add(coefficients, op, rhs):
        constraints.append(dict(id=f'c{len(constraints)}', a=coefficients, op=op, rhs=rhs))

    if family == KNAPSACK:
        linear = data['values']
        add(data['weights'], 'le', data['capacity'])
    elif family == CLIQUE:
        n = data['n']
        linear = [1] * n
        for i in range(n):
            for j in range(i):
                if not data['adj'][i][j]:
                    coefficients = [0] * n
                    coefficients[i] = coefficients[j] = 1
                    add(coefficients, 'le', 1)
    elif family == BUS:
        linear = [1] * data['num_shifts']
        for task in range(data['num_work']):
            add([int(task in shift) for shift in data['shifts']], 'eq', 1)
    else:
        raise ValueError(family)
    variables = [dict(name=f'x{i}', lower=0, upper=1) for i in range(len(linear))]
    return dict(schema='bounded-integer-v2', variables=variables, logic=[], constraints=constraints,
                objective=dict(sense='min' if family == BUS else 'max', linear=linear, quadratic=[]))


def original_score(family, data, x):
    """Independently score a binary witness from the published instance fields."""
    if family == KNAPSACK:
        n = len(data['values'])
    elif family == CLIQUE:
        n = data['n']
    else:
        n = data['num_shifts']
    if len(x) != n or any(bit not in (0, 1) for bit in x):
        raise ValueError('Invalid binary witness')
    chosen = [i for i in range(n) if x[i]]
    if family == KNAPSACK:
        return sum(data['weights'][i] for i in chosen) <= data['capacity'], sum(data['values'][i] for i in chosen)
    if family == CLIQUE:
        return all(data['adj'][i][j] for i in chosen for j in chosen if i < j), len(chosen)
    covered = sorted(task for i in chosen for task in data['shifts'][i])
    return covered == list(range(data['num_work'])), len(chosen)


def audit(raw, out):
    if hashlib.sha256(raw).hexdigest() != SHA256:
        raise ValueError('Dataset checksum mismatch')
    problems = [json.loads(line) for line in raw.splitlines()]
    public, references, audit_rows = [], [], []
    seen = set()
    for problem in problems:
        family = problem['id']
        if family not in FIELDS:
            continue
        for i, instance in enumerate(problem['instances']):
            case_id = f'dcp_{family}_{i:02}'
            data = {k: instance[k] for k in FIELDS[family]}
            signature = (family, json.dumps(data, sort_keys=True))
            if signature in seen:
                raise ValueError(f'Duplicate instance: {case_id}')
            seen.add(signature)
            spec = formulation(family, data)
            row = dict(id=case_id, family=family, example=i == 0, variables=len(spec['variables']),
                       constraints=len(spec['constraints']))
            audit_rows.append(row)
            try:
                validate(spec)
            except ValueError as exc:
                if str(exc) != 'At most 512 source constraints':
                    raise
                row['excluded'] = str(exc)
                continue
            result = solve(spec, 'scip', 30, seed=17)
            if not result['optimality_proven']:
                raise RuntimeError(f'{case_id}: {result["status"]}')
            feasible, optimum = original_score(family, data, result['x'])
            if not feasible or optimum != result['objective']:
                raise ValueError(f'Invalid reference: {case_id}')
            if i == 0:
                witness = problem['example_solution']['c' if family == CLIQUE else 'x']
                if original_score(family, data, witness) != (True, optimum):
                    raise ValueError(f'Published example disagrees: {case_id}')
            row['scip_optimum'] = optimum
            names = [v['name'] for v in spec['variables']]
            question = (problem['description'] + '\nInstance: ' + json.dumps(data, separators=(',', ':'))
                        + '\nUse x0, x1, ... in the order of the source binary decision vector.')
            public.append(dict(id=case_id, question=question, variable_names=names))
            references.append(dict(id=case_id, gold=spec, optimum=optimum, witness=result['x']))
            search = episode(spec, case_id, SEARCH, lambda *_: 'cool')
            ok, score = original_score(family, data, search['x'])
            row.update(pbit_feasible=bool(ok), pbit_objective=score if ok else None,
                       pbit_hit=bool(ok and score == optimum), attempts=search['heat_bath_attempts'])
    report = dict(experiment='E005-1.16', sha256=SHA256, release_problems=len(problems), rows=audit_rows)
    save(out/'public.json', public)
    save(out/'reference.json', references)
    save(out/'audit.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, help='Verified local copy of the release asset')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    cached = args.out/'dcp-bench-open-v0.3.0.jsonl'
    if args.source or cached.exists():
        raw = (args.source or cached).read_bytes()
    else:
        with urlopen(RELEASE, timeout=90) as response:
            raw = response.read()
    report = audit(raw, args.out)
    if not cached.exists():
        cached.write_bytes(raw)
    rows = report['rows']
    print(json.dumps(dict(selected=len(rows), verified=sum('scip_optimum' in r for r in rows),
                          gold_pbit_hits=sum(r.get('pbit_hit', False) for r in rows))))


if __name__ == '__main__':
    main()
