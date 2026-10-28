"""Common residual energy, initialization and accounting for sampling controls.

PT kernels are explicit reference implementations, NOT a reproduction of the
2026 ILP-PT paper's locally balanced multi-step proposal or GPU implementation.
"""
import time
import numpy as np
from ..bounded import objective, violations
from ..factor_search import encoding, objective_scale


def swap_log_ratio(fi, fj, pi, pj, ti, tj, li, lj):
    """Log joint target(new)/target(old), works for temperature/penalty swaps."""
    return -(fj+li*pj)/ti-(fi+lj*pi)/tj+(fi+li*pi)/ti+(fj+lj*pj)/tj


def solve(spec, method, seconds=5., seed=0, chains=16, max_evaluations=None):
    if method not in ('pbit', 'sa', 'pt', 'penalty_pt'):
        raise ValueError('Unknown sampling method')
    if not np.isfinite(seconds) or seconds <= 0 or not 1 <= chains <= 256:
        raise ValueError('Invalid sampler budget')
    if max_evaluations is not None and (type(max_evaluations) is not int or max_evaluations < chains):
        raise ValueError('Evaluation budget must include all initial chains')
    started = time.perf_counter()
    sites, lower = encoding(spec)
    rng = np.random.default_rng(seed)
    bits = rng.integers(0, 2, (chains, len(sites)))
    x = np.tile(lower, (chains, 1))
    for k, (i, w) in enumerate(sites):
        x[:, i] += bits[:, k]*w
    sign = 1 if spec['objective']['sense'] == 'min' else -1
    scale = objective_scale(spec)
    def score(states):
        residual = violations(spec, states)
        return sign*objective(spec, states)/scale, (residual**2).sum(axis=-1), residual.sum(axis=-1)
    f, p, v = score(x)
    evaluations = chains; updates = swaps = accepts = 0
    best, candidate, trace = None, None, []
    def archive():
        nonlocal best, candidate
        j = min(range(chains), key=lambda j: (v[j] > 0, v[j], f[j]))
        key = (bool(v[j] > 0), float(v[j]), float(f[j]))
        if best is None or key < best:
            best, candidate = key, x[j].astype(int).tolist()
            if len(trace) < 256:
                trace.append({'seconds': time.perf_counter()-started, 'evaluations': evaluations,
                              'feasible': not key[0], 'objective': sign*key[2]*scale})
    archive()
    # Initialize identically across methods for a given seed. No reference solution.
    while sites and time.perf_counter()-started < seconds:
        if max_evaluations is not None and evaluations+chains > max_evaluations:
            break
        fraction = min(1., (time.perf_counter()-started)/seconds) if max_evaluations is None else evaluations/max_evaluations
        anneal = .02**fraction
        temps = np.geomspace(.02, 1., chains) if method == 'pt' else np.full(chains, anneal)
        lambdas = np.geomspace(.05, 3., chains) if method == 'penalty_pt' else np.full(chains, 3.)
        k = int(rng.integers(len(sites))); i, w = sites[k]
        proposal = x.copy(); proposal[:, i] += (1-2*bits[:, k])*w
        nf, np_, nv = score(proposal); evaluations += chains
        delta = nf+lambdas*np_-f-lambdas*p
        u = rng.random(chains)
        if method == 'pbit':
            take = u < 1/(1+np.exp(np.clip(delta/temps, -700, 700)))
        else:
            take = np.log(np.maximum(u, 1e-300)) < np.minimum(0., -delta/temps)
        x[take] = proposal[take]; bits[take, k] = 1-bits[take, k]
        f[take], p[take], v[take] = nf[take], np_[take], nv[take]
        updates += chains; accepts += int(take.sum())
        archive()
        if method in ('pt', 'penalty_pt') and updates//chains % 16 == 0:
            parity = (updates//chains//16) % 2
            for a in range(parity, chains-1, 2):
                b = a+1
                ratio = swap_log_ratio(f[a], f[b], p[a], p[b], temps[a], temps[b], lambdas[a], lambdas[b])
                if np.log(max(rng.random(), 1e-300)) < min(0., ratio):
                    for values in (x, bits, f, p, v):
                        values[[a, b]] = values[[b, a]]
                    swaps += 1
    return {'x': candidate, 'feasible': not best[0], 'objective': float(objective(spec, candidate)),
            'optimality_proven': False, 'status': 'BUDGET_END', 'seconds': time.perf_counter()-started,
            'candidate_evaluations': evaluations, 'updates': updates, 'accepted_updates': accepts,
            'accepted_swaps': swaps, 'trace': trace, 'chains': chains, 'seed': seed,
            'energy': 'signed_objective/box_bound + lambda*sum(original_integer_residual^2)',
            'backend': method, 'scope': 'predicted_model_only'}
