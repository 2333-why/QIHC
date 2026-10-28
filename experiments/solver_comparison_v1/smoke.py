"""Small real-backend integration check before accepting a prepared environment."""
import itertools
import json
import os
from pathlib import Path
import sys
from qihc.generic.bounded import objective, violations
from qihc.generic.solver_baselines.registry import BACKENDS, solve
from qihc.generic.solver_baselines.isolation import BackendPools, NATIVE_MODULES


def case():
    return {'schema': 'bounded-integer-v2', 'variables': [
        {'name': 'x0', 'lower': -1, 'upper': 2}, {'name': 'x1', 'lower': 0, 'upper': 1},
        {'name': 'x2', 'lower': 0, 'upper': 1}],
        'objective': {'sense': 'min', 'linear': [-2, 1, 2], 'quadratic': [[0, 1, -3], [0, 0, 1]]},
        'constraints': [{'id': 'resource', 'a': [1, 1, 1], 'op': 'le', 'rhs': 3}],
        'logic': [{'id': 'imp', 'expr': {'op': 'implies', 'args': [{'var': 1}, {'var': 2}]}}]}


def check_one(name):
    spec = case()
    optimum = min(float(objective(spec, x)) for x in itertools.product(range(-1, 3), range(2), range(2)) if violations(spec, x).sum() == 0)
    row = solve(spec, name, 2. if name in NATIVE_MODULES else .1, seed=0)
    if row.get('x') is None or bool(violations(spec, row['x']).sum() == 0) != row['feasible']:
        raise RuntimeError('Backend smoke failed: '+name)
    if name in NATIVE_MODULES and (not row['optimality_proven'] or abs(row['objective']-optimum) > 1e-7):
        raise RuntimeError('Backend failed exact brute-force check: '+name)
    loaded = {key: getattr(sys.modules[module], '__file__', None)
              for key, module in NATIVE_MODULES.items() if module in sys.modules}
    if set(loaded) != ({name} if name in NATIVE_MODULES else set()):
        raise RuntimeError('Unexpected solver native modules in worker: '+str(loaded))
    out = {k: row.get(k) for k in ('status', 'feasible', 'objective', 'seconds')}
    out.update(worker_pid=os.getpid(), loaded_solver_modules=loaded)
    if Path('/proc/self/maps').exists():
        out['native_library_paths'] = sorted({line.split()[-1] for line in
            Path('/proc/self/maps').read_text().splitlines()
            if any(term in line.lower() for term in ('libhighs', 'libscip', 'libortools'))})
    return out


def check():
    with BackendPools(BACKENDS, len(BACKENDS)) as pools:
        futures = {name: pools.submit(name, check_one, name) for name in BACKENDS}
        out = {name: future.result() for name, future in futures.items()}
    if len({r['worker_pid'] for r in out.values()}) != len(BACKENDS):
        raise RuntimeError('Smoke reused processes across solver backends')
    return out


if __name__ == '__main__':
    print(json.dumps(check(), indent=2))
