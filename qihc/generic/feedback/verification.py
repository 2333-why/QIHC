"""Offline complete small-model equivalence, with explicit unknown outcomes."""
from __future__ import annotations

import time
import json
import math
import numpy as np

from ..bounded import objective, violations


def enumerate_states(variables, maximum=65536):
    widths = [v['upper'] - v['lower'] + 1 for v in variables]
    total = 1
    for w in widths: total *= w
    if total > maximum: return None
    values = np.arange(total, dtype=np.int64); cols = []
    for v, w in zip(variables, widths):
        cols.append(values % w + v['lower']); values //= w
    return np.column_stack(cols)


def ordering_equal(left, right):
    """O(n log n), preserving all ties, not just neighboring unsorted states."""
    if not len(left): return True
    order = np.argsort(left, kind='stable'); a, b = np.asarray(left)[order], np.asarray(right)[order]
    return bool(np.all((np.diff(a) == 0) == (np.diff(b) == 0)) and np.all(np.diff(b) >= 0))


def normalized(model):
    """Sufficient algebraic identity, never a heuristic correctness verdict."""
    rows = set()
    for c in model['constraints']:
        a = list(c['a']) + [c['rhs']]; op = c['op']
        if op == 'ge': a = [-v for v in a]; op = 'le'
        if op == 'eq' and next((v for v in a if v), 0) < 0: a = [-v for v in a]
        divisor = math.gcd(*a) or 1
        rows.add((op, tuple(v//divisor for v in a)))
    obj = model['objective']; linear = list(obj['linear']); quadratic = []
    for i,j,c in obj['quadratic']:
        if i == j and model['variables'][i]['lower']==0 and model['variables'][i]['upper']==1: linear[i] += c
        elif c: quadratic.append((i,j,c))
    sign = 1 if obj['sense']=='min' else -1
    divisor = math.gcd(*(linear+[c for _,_,c in quadratic])) or 1
    return (model['variables'], sorted(rows),
            sorted(json.dumps(c['expr'],sort_keys=True) for c in model['logic']),
            tuple(sign*c//divisor for c in linear), tuple((i,j,sign*c//divisor) for i,j,c in sorted(quadratic)))


def equivalent(model, gold, maximum=65536):
    if model is None: return dict(status='invalid', compiled=False)
    if [v['name'] for v in model['variables']] != [v['name'] for v in gold['variables']]:
        return dict(status='unknown', compiled=True, reason='variable_mapping_differs')
    if normalized(model) == normalized(gold):
        return dict(status='verified_correct', compiled=True, proof='algebraic_identity', enumeration_completed=False)
    variables = [dict(name=g['name'], lower=min(g['lower'], m['lower']), upper=max(g['upper'], m['upper']))
                 for g, m in zip(gold['variables'], model['variables'])]
    xs = enumerate_states(variables, maximum)
    exhaustive = xs is not None
    if not exhaustive:
        rng = np.random.default_rng(20261108)
        xs = np.column_stack([rng.integers(v['lower'],v['upper']+1,2048) for v in variables])
        xs[0] = [v['lower'] for v in variables]; xs[1] = [v['upper'] for v in variables]
    gf = violations(gold, xs).sum(axis=1) == 0; mf = violations(model, xs).sum(axis=1) == 0
    bad = np.flatnonzero(gf != mf)
    if len(bad): return dict(status='disproved', compiled=True, proof='feasible_set_counterexample', x=xs[int(bad[0])].tolist())
    left = objective(gold, xs[gf]) * (1 if gold['objective']['sense'] == 'min' else -1)
    right = objective(model, xs[gf]) * (1 if model['objective']['sense'] == 'min' else -1)
    okay = ordering_equal(left, right)
    if okay and not exhaustive:
        return dict(status='unknown', compiled=True, reason='enumeration_limit_no_counterexample',
                    enumeration_completed=False, states_checked=len(xs))
    return dict(status='verified_correct' if okay else 'disproved', compiled=True,
                proof='exhaustive_feasible_set_and_objective_order' if exhaustive else 'sampled_objective_order_counterexample', enumeration_completed=exhaustive,
                states_checked=len(xs), feasible_states=int(gf.sum()),
                objective_values_equal=bool(np.array_equal(objective(gold, xs[gf]), objective(model, xs[gf]))))


def reference(gold):
    xs = enumerate_states(gold['variables'])
    if xs is None: return dict(status='needs_solver_reference')
    feasible = xs[violations(gold, xs).sum(axis=1) == 0]
    if not len(feasible): return dict(status='infeasible', optimality_proven=True, x=None)
    values = objective(gold, feasible); i = int(np.argmin(values) if gold['objective']['sense'] == 'min' else np.argmax(values))
    return dict(status='optimal', optimality_proven=True, x=feasible[i].tolist(), objective=float(values[i]))


def score(gold, row, ref):
    x = row.get('x')
    try: feasible = x is not None and bool(violations(gold, x).sum() == 0)
    except (ValueError, TypeError): feasible = False
    value = float(objective(gold, x)) if feasible else None
    proven = bool(ref.get('optimality_proven')) and ref.get('objective') is not None
    gap = None
    if feasible and ref.get('objective') is not None:
        delta = value - ref['objective']
        gap = (delta if gold['objective']['sense'] == 'min' else -delta) / max(1., abs(ref['objective']))
    return dict(original_feasible=feasible, original_objective=value,
                optimal_hit=bool(feasible and proven and abs(value - ref['objective']) < 1e-8),
                gap=gap, reference_proven=proven)
