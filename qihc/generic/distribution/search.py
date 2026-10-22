"""Budgeted factor sampling, controlled interventions and generic policy selection.

These are software Gibbs/Metropolis kernels, NOT a claim of a hardware advantage.
No family labels, reference solutions, executable model code or semantic oracle.
All probes count against the same candidate-state scoring budget as exploitation.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import reduce
from math import gcd
import time

import numpy as np

from ..bounded import objective, violations
from ..factor_search import encoding, objective_scale


REPRESENTATIONS = ('raw_square', 'primitive_square', 'primitive_linear')
UPDATES = ('single', 'factor_pair')


@dataclass(frozen=True)
class Policy:
    representation: str = 'raw_square'
    update: str = 'single'

    def __post_init__(self):
        if self.representation not in REPRESENTATIONS or self.update not in UPDATES:
            raise ValueError('Unsupported search policy')

    def as_dict(self):
        return {'representation': self.representation, 'update': self.update}


def primitive_divisors(spec):
    # Including rhs preserves integrality: every nonzero normalized residual >= 1.
    return np.array([max(1, reduce(gcd, [abs(v) for v in c['a']]+[abs(c['rhs'])]))
                     for c in spec['constraints']] + [1]*len(spec['logic']), dtype=float)


def factor_sites(spec, sites):
    """Constraint supports, not hand-written moves for named problem families."""
    def leaves(expr):
        if 'var' in expr:
            return {expr['var']}
        return set().union(*(leaves(a) for a in expr['args']))
    supports = [{i for i, a in enumerate(c['a']) if a} for c in spec['constraints']]
    supports += [leaves(c['expr']) for c in spec['logic']]
    supports += [{i, j} for i, j, coefficient in spec['objective']['quadratic'] if coefficient]
    supports += [{i} for i in range(len(spec['variables']))]
    return sorted({tuple(k for k, (i, _) in enumerate(sites) if i in support)
                   for support in supports} - {()})


def penalty(residual, divisors, representation):
    r = residual if representation == 'raw_square' else residual/divisors
    return r if representation == 'primitive_linear' else r*r


def energy_values(spec, x, weights, representation):
    """Public pure helper, including original domain validation for tests."""
    r = violations(spec, x)[..., 1:]
    sign = 1 if spec['objective']['sense'] == 'min' else -1
    return sign*objective(spec, x)/objective_scale(spec) + penalty(
        r, primitive_divisors(spec), representation) @ weights


def state_key(spec, x):
    r = float(violations(spec, x).sum())
    sign = 1 if spec['objective']['sense'] == 'min' else -1
    return (r > 0, r, sign*float(objective(spec, x)))


def distribution_summary(spec, xs, rs, fs, changed, attempted):
    """Descriptive finite, correlated annealing samples; not equilibrium inference."""
    x, r, f = np.asarray(xs), np.asarray(rs), np.asarray(fs)
    ids = [c['id'] for c in spec['constraints']+spec['logic']]
    rates = (r > 0).mean(axis=0) if len(ids) else np.array([])
    focus = sorted(range(len(ids)), key=lambda j: (-rates[j], j))[:8]
    pairs = [{'ids': [ids[i], ids[j]], 'joint_violation_rate': float(((r[:, i] > 0) & (r[:, j] > 0)).mean()),
              'excess_over_product': float(((r[:, i] > 0) & (r[:, j] > 0)).mean()-rates[i]*rates[j])}
             for a, i in enumerate(focus) for j in focus[a+1:]]
    pairs.sort(key=lambda v: (-v['joint_violation_rate'], v['ids']))
    variables = []
    for i, var in enumerate(spec['variables']):
        _, counts = np.unique(x[:, i], return_counts=True)
        p = counts/counts.sum()
        variables.append({'name': var['name'], 'empirical_entropy_nats': float(-(p*np.log(p)).sum()),
                          'observed_values': len(counts), 'fixed_domain': var['lower'] == var['upper']})
    variables.sort(key=lambda v: (v['empirical_entropy_nats'], v['name']))
    return {'scope': 'predicted_model_only_not_semantic_proof',
            'sampling_warning': 'Correlated annealing snapshots, not independent or equilibrium samples. No infeasibility proof.',
            'snapshot_states': len(x), 'observed_feasible_rate': float((r.sum(axis=1) == 0).mean()),
            'objective_quantiles': [float(v) for v in np.quantile(f, [.1, .5, .9])],
            'transition_change_rate': changed/max(1, attempted),
            'constraints': [{'id': ids[i], 'violation_rate': float(rates[i]), 'mean_residual': float(r[:, i].mean())}
                            for i in focus],
            'constraints_total': len(ids), 'constraints_reported': len(focus),
            'co_violations': pairs[:8], 'low_entropy_variables': variables[:8]}


def window(spec, *, policy, method, seed, chains, budget, initial_bits=None,
           weights=None, temperature=(1., .05)):
    """Exact block heat-bath (<=2 bits) or symmetric Metropolis block proposals.

    Cache current energy. Heat-bath scores 1/3 alternatives per chain; Metropolis
    scores only its one proposed state. Never waste baseline work to match a cap.
    Block choices depend on static factor supports, never the current state.
    """
    if method not in ('pbit', 'gibbs', 'metropolis'):
        raise ValueError('Invalid sampling method')
    if type(chains) is not int or not 1 <= chains <= 256 or type(budget) is not int or budget < 1:
        raise ValueError('Positive bounded chains and score budget required')
    if len(temperature) != 2 or any(not np.isfinite(t) or t <= 0 for t in temperature):
        raise ValueError('Positive finite temperature endpoints required')
    sites, lower = encoding(spec)
    rng = np.random.default_rng(seed)
    bits = rng.integers(0, 2, (chains, len(sites))) if initial_bits is None else np.array(initial_bits, copy=True)
    if bits.shape != (chains, len(sites)) or not np.isin(bits, [0, 1]).all():
        raise ValueError('Invalid initial bits')
    x = np.tile(lower, (chains, 1))
    for k, (i, w) in enumerate(sites):
        x[:, i] += bits[:, k]*w
    divisors = primitive_divisors(spec)
    weights = np.full(len(divisors), 3.) if weights is None else np.asarray(weights, dtype=float)
    if weights.shape != divisors.shape or not np.isfinite(weights).all() or (weights <= 0).any():
        raise ValueError('Invalid positive penalty weights')
    supports = [v for v in factor_sites(spec, sites) if len(v) >= 2]
    sign = 1 if spec['objective']['sense'] == 'min' else -1
    scale = objective_scale(spec)
    def score(states):
        r = violations(spec, states)[..., 1:]
        f = objective(spec, states)
        e = sign*f/scale + penalty(r, divisors, policy.representation) @ weights
        return r, f, e
    r, f, current_energy = score(x)
    used, transitions, changed = 1, 0, 0
    archive, best = None, None
    first_feasible = None
    def inspect():
        nonlocal archive, best, first_feasible
        v = r.sum(axis=1)
        j = min(range(chains), key=lambda k: (v[k] > 0, v[k], sign*f[k]))
        key = (bool(v[j] > 0), float(v[j]), float(sign*f[j]))
        if best is None or key < best:
            best, archive = key, x[j].astype(int).tolist()
        if first_feasible is None and not key[0]:
            first_feasible = used*chains
    inspect()
    # Bounded memory: <=17 snapshots, independent of run length.
    xs, rs, fs = [x.copy()], [r.copy()], [f.copy()]
    next_snapshot = max(1, budget//16)
    while sites and used+1 <= budget:
        pair_cost = 1 if method == 'metropolis' else 3
        pair = policy.update == 'factor_pair' and supports and rng.random() < .75 and used+pair_cost <= budget
        block = sorted(rng.choice(supports[int(rng.integers(len(supports)))], 2, replace=False).tolist()) if pair else [int(rng.integers(len(sites)))]
        states = 2**len(block)
        codes = np.array([[(code >> j) & 1 for j in range(len(block))] for code in range(states)])
        candidates = np.repeat(x[:, None, :], states, axis=1)
        for j, k in enumerate(block):
            i, w = sites[k]
            candidates[:, :, i] += (codes[None, :, j]-bits[:, None, k])*w
        t = temperature[0]*(temperature[1]/temperature[0])**(used/max(1, budget-1))
        u = rng.random(chains)
        current = sum(bits[:, k]*(2**j) for j, k in enumerate(block)).astype(int)
        if method in ('pbit', 'gibbs'):
            alternatives = (current[:, None]+np.arange(1, states)[None, :]) % states
            ar, af, ae = score(candidates[np.arange(chains)[:, None], alternatives])
            cr = np.empty((chains, states, len(divisors)))
            cf, ce = np.empty((chains, states)), np.empty((chains, states))
            cr[np.arange(chains), current], cf[np.arange(chains), current], ce[np.arange(chains), current] = r, f, current_energy
            cr[np.arange(chains)[:, None], alternatives] = ar
            cf[np.arange(chains)[:, None], alternatives], ce[np.arange(chains)[:, None], alternatives] = af, ae
            mass = np.exp(np.maximum(-700, -(ce-ce.min(axis=1, keepdims=True))/t))
            cdf = np.cumsum(mass/mass.sum(axis=1, keepdims=True), axis=1)
            chosen = np.minimum((u[:, None] > cdf).sum(axis=1), states-1)
            r, f, current_energy = cr[np.arange(chains), chosen], cf[np.arange(chains), chosen], ce[np.arange(chains), chosen]
            used += states-1
        else:
            # Uniform different assignment: symmetric even for a 2-bit block.
            proposed = (current+rng.integers(1, states, chains)) % states
            pr, pf, pe = score(candidates[np.arange(chains), proposed])
            accept = u < np.exp(np.minimum(0, -(pe-current_energy)/t))
            chosen = np.where(accept, proposed, current)
            r, f, current_energy = np.where(accept[:, None], pr, r), np.where(accept, pf, f), np.where(accept, pe, current_energy)
            used += 1
        changed += int((chosen != current).sum())
        transitions += chains
        bits[:, block] = codes[chosen]
        x = candidates[np.arange(chains), chosen]
        inspect()
        if used >= next_snapshot and len(xs) < 17:
            xs.append(x.copy()); rs.append(r.copy()); fs.append(f.copy())
            next_snapshot += max(1, budget//16)
    summary = distribution_summary(spec, np.concatenate(xs), np.concatenate(rs), np.concatenate(fs), changed, transitions)
    return {'x': archive, 'key': best, 'bits': bits, 'violation_rates': (np.concatenate(rs) > 0).mean(axis=0),
            'diagnostic': summary, 'candidate_evaluations': used*chains, 'budget': budget*chains,
            'first_feasible_evaluations': first_feasible, 'transition_attempts': transitions,
            'policy': policy.as_dict()}


def solve(spec, *, seed=0, chains=16, budget=8192, method='pbit', interventions=False,
          selection='fixed'):
    """Four equal budget stages: baseline, paired probes, policy pilots, exploit.

    Controls spend the SAME probe slots on sham interventions / baseline pilots.
    Every visited candidate may enter the output archive in every arm. The final
    penalty=3 gives exact global minima on feasible integer models, but finite
    sampling is never claimed to find one. Interventions are temporary diagnostics.
    """
    if type(budget) is not int or not 256 <= budget <= 1000000:
        raise ValueError('Score budget per chain must be 256..1000000')
    if selection not in ('fixed', 'energy', 'update', 'joint', 'random') or type(interventions) is not bool:
        raise ValueError('Invalid selection/intervention setting')
    started = time.perf_counter()
    # Validation precedes all numerical work, even on fixed-variable models.
    sites, _ = encoding(spec)
    seeds = np.random.SeedSequence(seed).generate_state(32).tolist()
    base_policy = Policy()
    base = window(spec, policy=base_policy, method=method, seed=seeds[0], chains=chains, budget=budget//4)
    windows = [base]
    ids = [c['id'] for c in spec['constraints']+spec['logic']]
    targets = sorted(range(len(ids)), key=lambda j: (-base['violation_rates'][j], j))[:2]
    reports = []
    for slot in range(2):
        target = targets[slot] if slot < len(targets) else None
        paired = []
        for branch in range(2):
            weights = np.full(len(ids), 3.)
            if interventions and target is not None:
                weights[target] *= (.25, 4.)[branch]
            w = window(spec, policy=base_policy, method=method, seed=seeds[1+slot], chains=chains,
                       budget=budget//16, initial_bits=base['bits'], weights=weights, temperature=(.5, .5))
            windows.append(w); paired.append(w)
        if target is not None:
            delta = paired[1]['violation_rates']-paired[0]['violation_rates']
            related = sorted(range(len(ids)), key=lambda j: (-abs(delta[j]), j))[:8]
            reports.append({'constraint': ids[target], 'intervention': interventions,
                'multipliers': [.25, 4.] if interventions else [1., 1.],
                'paired_initial_states_and_random_seed': True,
                'violation_rate_delta_high_minus_low': [{'id': ids[j], 'delta': float(delta[j])} for j in related],
                'objective_median_delta': paired[1]['diagnostic']['objective_quantiles'][1]-paired[0]['diagnostic']['objective_quantiles'][1],
                'scope': 'finite_budget_sensitivity_not_semantic_or_causal_proof'})
    policies = [Policy(r, u) for r in REPRESENTATIONS for u in UPDATES]
    eligible = [p for p in policies if (selection in ('joint', 'energy', 'random') or p.representation == 'raw_square')
                and (selection in ('joint', 'update', 'random') or p.update == 'single')]
    pilots = []
    for i, _ in enumerate(policies):
        p = eligible[i % len(eligible)]
        w = window(spec, policy=p, method=method, seed=seeds[3+i], chains=chains,
                   budget=(budget//4)//6, initial_bits=base['bits'])
        windows.append(w); pilots.append(w)
    # Aggregate repeated pilot evidence equally per eligible policy; do not choose
    # a favorable seed. Rank by feasible sample mass then best visited state.
    ranks = []
    for p in eligible:
        rows = [w for w in pilots if w['policy'] == p.as_dict()]
        key = (-float(np.mean([w['diagnostic']['observed_feasible_rate'] for w in rows])),
               float(np.mean([w['key'][1] for w in rows])), float(np.mean([w['key'][2] for w in rows])))
        ranks.append((key, p))
    chosen = min(ranks, key=lambda row: row[0])[1]
    if selection == 'random':
        chosen = eligible[int(np.random.default_rng(seeds[21]).integers(len(eligible)))]
    # Use reserved, not consumed, budgets so structural no-op models cannot spend
    # a huge unused allowance. Rounding leaves at most a few unspent scores.
    reserved = budget//4 + 4*(budget//16) + 6*((budget//4)//6)
    final = window(spec, policy=chosen, method=method, seed=seeds[20], chains=chains,
                   budget=budget-reserved, initial_bits=base['bits'])
    windows.append(final)
    winner = min(windows, key=lambda w: w['key'])
    candidate = winner['x']
    spent = sum(w['candidate_evaluations'] for w in windows)
    diagnostic = {'best': {'x': candidate, 'predicted_objective': float(objective(spec, candidate)),
                          'predicted_model_feasible': not winner['key'][0], 'violation': winner['key'][1]},
                  'distribution': base['diagnostic'], 'interventions': reports,
                  'selected_policy': chosen.as_dict(),
                  'selection_uses': 'seeded_random_no_quality_ranking' if selection == 'random' else 'predicted_model_only_no_gold',
                  'semantic_correctness': 'unverified'}
    offset, first = 0, None
    for w in windows:
        if first is None and w['first_feasible_evaluations'] is not None:
            first = offset+w['first_feasible_evaluations']
        offset += w['candidate_evaluations']
    return {'x': candidate, 'objective': float(objective(spec, candidate)), 'feasible': not winner['key'][0],
            'violation': winner['key'][1], 'seed': seed, 'method': method, 'selection': selection,
            'interventions_enabled': interventions, 'diagnostic': diagnostic,
            'selected_policy': chosen.as_dict(), 'candidate_evaluations': spent,
            'candidate_budget': budget*chains, 'unused_budget': budget*chains-spent,
            'transition_attempts': sum(w['transition_attempts'] for w in windows),
            'first_predicted_feasible_evaluations': first,
            'probe_ledger': [{'stage': 'baseline' if i == 0 else 'intervention' if i < 5 else 'pilot' if i < 11 else 'exploit',
                              'candidate_evaluations': w['candidate_evaluations'], 'budget': w['budget'],
                              'policy': w['policy']} for i, w in enumerate(windows)],
            'bits': len(sites), 'chains': chains, 'seconds': time.perf_counter()-started,
            'backend': 'distribution_factor_simulation_v1', 'optimality_proven': False}
