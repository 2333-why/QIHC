"""Exact binary MILP reformulation of the bounded-integer IR.

No penalty approximation: products and Boolean expressions are linearized with
auxiliary binary variables. Used identically by HiGHS and SCIP in E002.
"""
from dataclasses import dataclass
import math
import time
import numpy as np
from ..bounded import validate, objective, violations
from ..factor_search import encoding


class UnsupportedModel(ValueError):
    """Explicit capability limit; record it instead of silently approximating."""


@dataclass
class BinaryMILP:
    sites: list
    lower: np.ndarray
    costs: list
    offset: int
    rows: list
    sign: int

    def decode(self, bits):
        x = self.lower.copy()
        for bit, (i, weight) in zip(bits, self.sites):
            x[i] += weight*round(float(bit))
        return x.astype(int).tolist()


def compile_model(spec, max_variables=20000):
    validate(spec)
    sites, lower = encoding(spec)
    costs, rows = [0]*len(sites), []
    expansions = [[] for _ in lower]
    for k, (i, w) in enumerate(sites):
        expansions[i].append((k, w))

    def new():
        if len(costs) >= max_variables:
            raise UnsupportedModel('Exact MILP expansion exceeds configured variable limit')
        costs.append(0)
        return len(costs)-1

    def row(terms, op, rhs):
        a = {}
        for k, v in terms:
            a[k] = a.get(k, 0)+int(v)
        rows.append(({k: v for k, v in a.items() if v}, op, int(rhs)))

    products = {}
    def product(i, j):
        if i == j:
            return i
        key = tuple(sorted((i, j)))
        if key not in products:
            z = new(); products[key] = z
            row([(z, 1), (i, -1)], 'le', 0)
            row([(z, 1), (j, -1)], 'le', 0)
            row([(z, 1), (i, -1), (j, -1)], 'ge', -1)
        return products[key]

    def gate(op, args):
        z = new()
        if op == 'not':
            row([(z, 1), (args[0], 1)], 'eq', 1)
        elif op == 'and':
            for a in args:
                row([(z, 1), (a, -1)], 'le', 0)
            row([(z, 1)]+[(a, -1) for a in args], 'ge', 1-len(args))
        elif op == 'or':
            for a in args:
                row([(z, 1), (a, -1)], 'ge', 0)
            row([(z, 1)]+[(a, -1) for a in args], 'le', 0)
        return z

    def expr(e):
        if 'var' in e:
            return expansions[e['var']][0][0]
        args = [expr(a) for a in e['args']]
        if e['op'] == 'implies':
            return gate('or', [gate('not', [args[0]]), args[1]])
        if e['op'] == 'iff':
            return gate('and', [gate('or', [gate('not', [args[0]]), args[1]]),
                                gate('or', [gate('not', [args[1]]), args[0]])])
        return gate(e['op'], args)

    for c in spec['constraints']:
        row([(k, a*w) for i, a in enumerate(c['a']) for k, w in expansions[i]],
            c['op'], c['rhs']-sum(int(v)*a for v, a in zip(lower, c['a'])))
    for c in spec['logic']:
        row([(expr(c['expr']), 1)], 'eq', 1)
    offset = sum(int(v)*a for v, a in zip(lower, spec['objective']['linear']))
    for i, a in enumerate(spec['objective']['linear']):
        for k, w in expansions[i]:
            costs[k] += a*w
    for i, j, a in spec['objective']['quadratic']:
        offset += a*int(lower[i])*int(lower[j])
        for k, w in expansions[i]:
            costs[k] += a*w*int(lower[j])
        for k, w in expansions[j]:
            costs[k] += a*w*int(lower[i])
        for k, w in expansions[i]:
            for l, v in expansions[j]:
                z = product(k, l)
                costs[z] += a*w*v
    sign = 1 if spec['objective']['sense'] == 'min' else -1
    # Reject numerically unsafe instances, never silently round a large MILP.
    if sum(abs(v) for v in costs)+abs(offset) >= 2**52 or any(
            sum(abs(v) for v in a.values())+abs(rhs) >= 2**52 for a, _, rhs in rows):
        raise UnsupportedModel('Exact MILP coefficients exceed float64 integer safety budget')
    return BinaryMILP(sites, lower, [sign*v for v in costs], sign*offset, rows, sign)


def solve(spec, backend, seconds, seed=0):
    started = time.perf_counter()
    ir = compile_model(spec)
    compile_seconds = time.perf_counter()-started
    if not ir.costs:
        x = ir.lower.astype(int).tolist()
        feasible = bool(violations(spec, x).sum() == 0)
        return {'x': x, 'feasible': feasible, 'objective': float(objective(spec, x)),
                'status': 'OPTIMAL' if feasible else 'INFEASIBLE', 'optimality_proven': feasible,
                'seconds': time.perf_counter()-started, 'compile_seconds': compile_seconds}
    if backend == 'highs':
        import highspy
        m = highspy.Highs()
        def checked(status):
            if status != highspy.HighsStatus.kOk:
                raise RuntimeError('HiGHS rejected model construction: '+str(status))
        for k, v in {'output_flag': False, 'threads': 1, 'parallel': 'off',
                     'random_seed': seed, 'mip_rel_gap': 0., 'mip_abs_gap': 0.}.items():
            if m.setOptionValue(k, v) != highspy.HighsStatus.kOk:
                raise RuntimeError('HiGHS rejected option '+k)
        checked(m.addVars(len(ir.costs), np.zeros(len(ir.costs)), np.ones(len(ir.costs))))
        indices = np.arange(len(ir.costs), dtype=np.int32)
        checked(m.changeColsCost(len(indices), indices, np.array(ir.costs, dtype=float)))
        checked(m.changeColsIntegrality(len(indices), indices, np.ones(len(indices), dtype=np.uint8)))
        for a, op, rhs in ir.rows:
            checked(m.addRow(-highspy.kHighsInf if op == 'le' else rhs,
                     highspy.kHighsInf if op == 'ge' else rhs,
                     len(a), np.array(list(a), dtype=np.int32), np.array(list(a.values()), dtype=float)))
        remaining = seconds-(time.perf_counter()-started)
        if remaining <= 0:
            return {'x': None, 'status': 'BUILD_TIME_LIMIT', 'feasible': False,
                    'seconds': time.perf_counter()-started, 'compile_seconds': compile_seconds}
        checked(m.setOptionValue('time_limit', remaining))
        if m.run() == highspy.HighsStatus.kError:
            raise RuntimeError('HiGHS solver returned an error')
        sol, info, status = m.getSolution(), m.getInfo(), m.getModelStatus()
        bits = sol.col_value if sol.value_valid else None
        optimal = status == highspy.HighsModelStatus.kOptimal
        status = m.modelStatusToString(status)
        bound = info.mip_dual_bound
    elif backend == 'scip':
        from pyscipopt import Model, quicksum
        m = Model(); m.hideOutput()
        m.setIntParam('parallel/maxnthreads', 1)
        m.setIntParam('randomization/randomseedshift', seed)
        m.setRealParam('limits/gap', 0.); m.setRealParam('limits/absgap', 0.)
        x = [m.addVar(vtype='B', name='b'+str(k)) for k in range(len(ir.costs))]
        for a, op, rhs in ir.rows:
            lhs = quicksum(v*x[k] for k, v in a.items())
            m.addCons(lhs <= rhs if op == 'le' else lhs >= rhs if op == 'ge' else lhs == rhs)
        m.setObjective(quicksum(c*v for c, v in zip(ir.costs, x)), 'minimize')
        remaining = seconds-(time.perf_counter()-started)
        if remaining <= 0:
            return {'x': None, 'status': 'BUILD_TIME_LIMIT', 'feasible': False,
                    'seconds': time.perf_counter()-started, 'compile_seconds': compile_seconds}
        m.setRealParam('limits/time', remaining); m.optimize()
        sol = m.getBestSol(); bits = [m.getSolVal(sol, v) for v in x] if sol is not None else None
        status = str(m.getStatus()); optimal = status == 'optimal'; bound = m.getDualbound()
    else:
        raise ValueError('Unknown MILP backend')
    x = ir.decode(bits) if bits is not None else None
    feasible = x is not None and bool(violations(spec, x).sum() == 0)
    return {'x': x, 'feasible': feasible, 'objective': float(objective(spec, x)) if x is not None else None,
            'optimality_proven': optimal and feasible, 'status': status,
            'bound': ir.sign*(bound+ir.offset) if math.isfinite(bound) and abs(bound) < 1e19 else None,
            'seconds': time.perf_counter()-started, 'compile_seconds': compile_seconds,
            'binary_variables': len(ir.costs), 'linear_rows': len(ir.rows), 'scope': 'predicted_model_only'}
