"""E002 CP-SAT adapter with total-budget accounting; historical adapter stays frozen."""
from __future__ import annotations
import math
import time
from ..bounded import validate, objective, violations


def solve(spec, seconds=5., *, seed=0, include_build=False):
    validate(spec)
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError('Positive finite reference timeout required')
    from ortools.sat.python import cp_model
    started = time.perf_counter()
    model = cp_model.CpModel()
    x = [model.NewBoolVar(v['name']) if (v['lower'], v['upper']) == (0, 1)
         else model.NewIntVar(v['lower'], v['upper'], v['name']) for v in spec['variables']]
    for c in spec['constraints']:
        lhs = sum(v*a for v, a in zip(x, c['a']))
        model.Add(lhs <= c['rhs'] if c['op'] == 'le' else lhs >= c['rhs'] if c['op'] == 'ge' else lhs == c['rhs'])
    def expression(expr):
        if 'var' in expr:
            return x[expr['var']]
        args = [expression(a) for a in expr['args']]
        op = expr['op']
        if op == 'not':
            return args[0].Not()
        if op == 'implies':
            args, op = [args[0].Not(), args[1]], 'or'
        z = model.NewBoolVar('gate')
        if op == 'and':
            model.AddBoolAnd(args).OnlyEnforceIf(z)
            model.AddBoolOr([a.Not() for a in args]).OnlyEnforceIf(z.Not())
        elif op == 'or':
            model.AddBoolOr(args).OnlyEnforceIf(z)
            model.AddBoolAnd([a.Not() for a in args]).OnlyEnforceIf(z.Not())
        else:
            model.Add(args[0] == args[1]).OnlyEnforceIf(z)
            model.Add(args[0] != args[1]).OnlyEnforceIf(z.Not())
        return z
    for c in spec['logic']:
        model.AddBoolOr([expression(c['expr'])])
    obj = sum(c*v for c, v in zip(spec['objective']['linear'], x))
    for i, j, c in spec['objective']['quadratic']:
        bounds = [a*b for a in (spec['variables'][i]['lower'], spec['variables'][i]['upper'])
                  for b in (spec['variables'][j]['lower'], spec['variables'][j]['upper'])]
        product = model.NewIntVar(min(bounds), max(bounds), 'product')
        model.AddMultiplicationEquality(product, [x[i], x[j]])
        obj += c*product
    (model.Minimize if spec['objective']['sense'] == 'min' else model.Maximize)(obj)
    solver = cp_model.CpSolver()
    remaining = seconds-(time.perf_counter()-started) if include_build else seconds
    if remaining <= 0:
        return {'status': 'BUILD_TIME_LIMIT', 'x': None, 'feasible': False,
                'optimality_proven': False, 'seconds': time.perf_counter()-started}
    solver.parameters.max_time_in_seconds = remaining
    solver.parameters.num_search_workers = 1
    solver.parameters.random_seed = seed
    status = solver.Solve(model)
    candidate = [solver.Value(v) for v in x] if status in (cp_model.OPTIMAL, cp_model.FEASIBLE) else None
    return {'status': solver.StatusName(status), 'x': candidate,
            'feasible': candidate is not None and bool(violations(spec, candidate).sum() == 0),
            'objective': float(objective(spec, candidate)) if candidate is not None else None,
            'optimality_proven': status == cp_model.OPTIMAL,
            'bound': float(solver.BestObjectiveBound()) if candidate is not None else None,
            'seconds': time.perf_counter()-started, 'scope': 'predicted_model_only'}
