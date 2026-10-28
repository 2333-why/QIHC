"""Small real-backend integration check before accepting a prepared environment."""
import itertools
import json
from qihc.generic.bounded import objective, violations
from qihc.generic.solver_baselines.registry import BACKENDS, solve


def case():
    return {'schema': 'bounded-integer-v2', 'variables': [
        {'name': 'x0', 'lower': -1, 'upper': 2}, {'name': 'x1', 'lower': 0, 'upper': 1},
        {'name': 'x2', 'lower': 0, 'upper': 1}],
        'objective': {'sense': 'min', 'linear': [-2, 1, 2], 'quadratic': [[0, 1, -3], [0, 0, 1]]},
        'constraints': [{'id': 'resource', 'a': [1, 1, 1], 'op': 'le', 'rhs': 3}],
        'logic': [{'id': 'imp', 'expr': {'op': 'implies', 'args': [{'var': 1}, {'var': 2}]}}]}


def check():
    spec = case()
    optimum = min(float(objective(spec, x)) for x in itertools.product(range(-1, 3), range(2), range(2)) if violations(spec, x).sum() == 0)
    out = {}
    for name in BACKENDS:
        row = solve(spec, name, 2. if name in ('cp_sat', 'scip', 'highs') else .1, seed=0)
        if row.get('x') is None or bool(violations(spec, row['x']).sum() == 0) != row['feasible']:
            raise RuntimeError('Backend smoke failed: '+name)
        if name in ('cp_sat', 'scip', 'highs') and (not row['optimality_proven'] or abs(row['objective']-optimum) > 1e-7):
            raise RuntimeError('Backend failed exact brute-force check: '+name)
        out[name] = {k: row.get(k) for k in ('status', 'feasible', 'objective', 'seconds')}
    return out


if __name__ == '__main__':
    print(json.dumps(check(), indent=2))
