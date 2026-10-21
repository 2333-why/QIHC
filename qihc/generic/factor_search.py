"""Family-independent p-bit simulation on original residual energies.

No FPGA capacity, QUBO matrix, slack bits, generated code or family-specific moves.
The state is a binary encoding of bounded integers; energies may be higher-order.
"""
from __future__ import annotations
import time
import numpy as np
from .bounded import validate, objective, violations
from .energy import bounded_weights


def encoding(spec):
    validate(spec)
    sites = [(i, w) for i, v in enumerate(spec['variables'])
             for w in bounded_weights(v['upper']-v['lower'])]
    return sites, np.array([v['lower'] for v in spec['variables']], dtype=float)


def objective_scale(spec):
    # Conservative absolute bound on the original objective over the complete box.
    radius = np.array([max(abs(v['lower']), abs(v['upper'])) for v in spec['variables']])
    return max(1., sum(abs(c)*r for c, r in zip(spec['objective']['linear'], radius)) +
               sum(abs(c)*radius[i]*radius[j] for i, j, c in spec['objective']['quadratic']))


def energy(spec, x, weights):
    sign = 1 if spec['objective']['sense'] == 'min' else -1
    residuals = violations(spec, x)[..., 1:]
    return sign*objective(spec, x)/objective_scale(spec) + (residuals**2) @ weights


def solve(spec, *, seed=0, chains=16, steps=4096, feedback=True, method='pbit'):
    if method not in ('pbit', 'metropolis', 'uniform') or type(feedback) is not bool:
        raise ValueError('Invalid method/feedback')
    if any(type(v) is not int or not 1 <= v <= m for v, m in ((chains, 256), (steps, 1000000))):
        raise ValueError('Invalid budget')
    sites, lower = encoding(spec)
    start = time.perf_counter()
    init, schedule, draws = [np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(3)]
    bits = init.integers(0, 2, (chains, len(sites)))
    x = np.tile(lower, (chains, 1))
    for k, (i, w) in enumerate(sites):
        x[:, i] += w*bits[:, k]
    weights = np.full(len(spec['constraints'])+len(spec['logic']), .1)
    archive, best, history = None, None, []
    sign = 1 if spec['objective']['sense'] == 'min' else -1
    def inspect():
        nonlocal archive, best
        r = violations(spec, x)
        v, f = r.sum(axis=1), objective(spec, x)
        j = min(range(chains), key=lambda k: (v[k] > 0, v[k], sign*f[k]))
        key = (bool(v[j] > 0), float(v[j]), float(sign*f[j]))
        if best is None or key < best:
            best, archive = key, x[j].astype(int).tolist()
        return r[:, 1:]
    inspect()
    # Constant models need no dummy bit updates.
    actual_steps = steps if sites else 0
    interval = max(1, steps//8)
    for step in range(actual_steps):
        if step == int(.75*steps) and method != 'uniform':
            weights = np.maximum(weights, 2.)
        site = int(schedule.integers(len(sites)))
        i, w = sites[site]
        x0, x1 = x.copy(), x.copy()
        x0[:, i] -= bits[:, site]*w
        x1[:, i] = x0[:, i]+w
        delta = energy(spec, x1, weights)-energy(spec, x0, weights)
        temperature = .02 ** (step/max(1, steps-1))
        u = draws.random(chains)
        if method == 'pbit':
            new = (u < 1/(1+np.exp(np.clip(delta/temperature, -60, 60)))).astype(int)
        elif method == 'metropolis':
            change = (1-2*bits[:, site])*delta
            new = np.where(u < np.exp(np.minimum(0, -change/temperature)), 1-bits[:, site], bits[:, site])
        else:
            new = (u < .5).astype(int)
        x[:, i] = x0[:, i]+new*w
        bits[:, site] = new
        r = inspect()
        if (step+1) % interval == 0 or step+1 == steps:
            rates = (r > 0).mean(axis=0)
            history.append({'step': step+1, 'violation_rates': rates.tolist(), 'weights': weights.tolist()})
            if feedback and method != 'uniform' and step+1 < steps:
                weights = np.minimum(8, weights*(1+.75*rates))
    return {'x': archive, 'objective': float(objective(spec, archive)), 'feasible': not best[0],
            'violation': best[1], 'bits': len(sites), 'auxiliary_bits': 0,
            'steps': actual_steps, 'requested_steps': steps, 'chains': chains,
            'bit_updates': actual_steps*chains, 'energy_evaluations': 2*actual_steps*chains,
            'method': method, 'feedback_enabled': feedback, 'seed': seed,
            'history': history, 'final_weights': weights.tolist(),
            'seconds': time.perf_counter()-start, 'backend': 'original_factor_pbit_float64',
            'optimality_proven': False}
