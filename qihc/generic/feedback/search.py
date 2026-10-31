"""E004 source-blind conditional probes. No task labels or reference answers.

Finite-time software heat-bath/Metropolis, not hardware or equilibrium claims.
All temporary controls are applied to copies, and final candidates use the
original mathematical model. Historical search kernels remain untouched.
"""
import copy
import math
import time
import numpy as np
from ..bounded import validate, objective, violations
from ..factor_search import encoding, objective_scale
from ..distribution.search import primitive_divisors


class Energy:
    def __init__(self, spec, weights=None, scale=None):
        self.spec = validate(spec)
        self.scale = float(scale or objective_scale(spec))
        self.sign = 1 if spec['objective']['sense'] == 'min' else -1
        self.div = primitive_divisors(spec)
        self.weights = np.full(len(self.div), 3.) if weights is None else np.asarray(weights)
        self.a = np.array([c['a'] for c in spec['constraints']], dtype=float).reshape(-1, len(spec['variables']))
        self.rhs = np.array([c['rhs'] for c in spec['constraints']])
        self.ops = [c['op'] for c in spec['constraints']]
        self.lin = np.array(spec['objective']['linear'], dtype=float)
        self.quad = np.array(spec['objective']['quadratic'], dtype=float).reshape(-1, 3)

    def values(self, x):
        x = np.asarray(x)
        f = x @ self.lin
        if len(self.quad):
            i, j = self.quad[:, 0].astype(int), self.quad[:, 1].astype(int)
            f = f+(x[..., i]*x[..., j]) @ self.quad[:, 2]
        r = x @ self.a.T-self.rhs
        for k, op in enumerate(self.ops):
            r[..., k] = abs(r[..., k]) if op == 'eq' else np.maximum(0, r[..., k] if op == 'le' else -r[..., k])
        if self.spec['logic']:
            from ..bounded import logic_value
            logic = np.stack([(~logic_value(c['expr'], x)).astype(float) for c in self.spec['logic']], axis=-1)
            r = np.concatenate([r, logic], axis=-1)
        e = self.sign*f/self.scale + ((r/self.div)**2) @ self.weights
        return f, r, e


def conditional_probability(e0, e1, temperature):
    if temperature <= 0:
        raise ValueError('Positive temperature required')
    z = np.clip((np.asarray(e1)-np.asarray(e0))/temperature, -700, 700)
    return 1/(1+np.exp(z))


def encode_states(spec, xs):
    sites, lower = encoding(spec)
    remain = np.asarray(xs, dtype=int)-lower
    bits = np.zeros((len(remain), len(sites)), dtype=int)
    for k in reversed(range(len(sites))):
        i, w = sites[k]
        bits[:, k] = remain[:, i] >= w
        remain[:, i] -= bits[:, k]*w
    if np.any(remain):
        raise ValueError('State is outside integer encoding')
    return bits


def sample(spec, seconds, seed, chains=32, method='pbit', temperature=.5,
           initial=None, clamps=None, weights=None, scale=None, read_q=True, max_steps=None):
    start = time.monotonic(); deadline = start+max(0., seconds)
    energy = Energy(spec, weights, scale)
    sites, lower = encoding(spec)
    rng = np.random.default_rng(seed)
    bits = (rng.integers(0, 2, (chains, len(sites))) if initial is None
            else encode_states(spec, np.asarray(initial)[:chains]))
    chains = len(bits)
    clamps = {int(i): int(v) for i, v in (clamps or {}).items()}
    for i, value in clamps.items():
        var = spec['variables'][i]
        if not var['lower'] <= value <= var['upper']:
            raise ValueError('Clamp outside declared domain')
        remaining = value-var['lower']
        for k in reversed(range(len(sites))):
            j, w = sites[k]
            if i == j:
                bits[:, k] = remaining >= w
                remaining -= int(remaining >= w)*w
        if remaining:
            raise ValueError('Clamp encoding failed')
    xs = np.tile(lower, (chains, 1))
    for k, (i, w) in enumerate(sites):
        xs[:, i] += bits[:, k]*w
    free = [k for k, (i, _) in enumerate(sites) if i not in clamps]
    f, r, e = energy.values(xs)
    evaluations = chains; changes = steps = 0
    samples = [xs.copy()]; best = None; best_x = None; qsum = {}; qcount = {}
    def inspect():
        nonlocal best, best_x
        residual = r.sum(axis=1)
        k = min(range(chains), key=lambda j: (residual[j] > 0, residual[j], energy.sign*f[j]))
        key = (bool(residual[k] > 0), float(residual[k]), float(energy.sign*f[k]))
        if best is None or key < best:
            best, best_x = key, xs[k].astype(int).tolist()
    inspect()
    next_snapshot = start+max(seconds/16, .001)
    while free and time.monotonic() < deadline and (max_steps is None or steps < max_steps):
        k = int(rng.choice(free)); i, w = sites[k]
        other = xs.copy(); other[:, i] += (1-2*bits[:, k])*w
        nf, nr, ne = energy.values(other); evaluations += chains
        if method == 'metropolis':
            accept = rng.random(chains) < np.exp(np.minimum(0, -(ne-e)/temperature))
        elif method in ('pbit', 'gibbs'):
            q = conditional_probability(np.where(bits[:, k] == 0, e, ne), np.where(bits[:, k] == 1, e, ne), temperature)
            chosen = rng.random(chains) < q
            accept = chosen != bits[:, k]
        else:
            raise ValueError('Unknown sampling kernel')
        if read_q:
            q = conditional_probability(np.where(bits[:, k] == 0, e, ne), np.where(bits[:, k] == 1, e, ne), temperature)
            qsum[k] = qsum.get(k, 0.)+float(q.sum()); qcount[k] = qcount.get(k, 0)+chains
        bits[accept, k] = 1-bits[accept, k]
        xs[accept] = other[accept]; f[accept] = nf[accept]; r[accept] = nr[accept]; e[accept] = ne[accept]
        changes += int(accept.sum()); steps += 1; inspect()
        if time.monotonic() >= next_snapshot and len(samples) < 16:
            samples.append(xs.copy()); next_snapshot += max(seconds/16, .001)
    samples.append(xs.copy())
    return summarize(spec, np.concatenate(samples), xs, best_x, evaluations, changes, steps*chains,
                     time.monotonic()-start, [{'bit': k, 'variable': sites[k][0], 'weight': sites[k][1],
                     'conditional_q_mean': qsum[k]/qcount[k]} for k in sorted(qsum)])


def summarize(spec, samples, final, best_x, evaluations=0, changes=0, attempts=0, seconds=0., qs=None):
    e = Energy(spec); f, r, _ = e.values(samples)
    half = max(1, len(final)//2)
    stability = 1-float(np.mean(abs(final[:half].mean(axis=0)-final[-half:].mean(axis=0))/
                               np.maximum(1, [v['upper']-v['lower'] for v in spec['variables']])))
    unique = np.unique(samples, axis=0)
    # Deterministic spread, not a hidden quality-selected or gold-selected pool.
    representatives = unique[np.linspace(0, len(unique)-1, min(16, len(unique))).astype(int)]
    return {'x': best_x, 'states': representatives.astype(int).tolist(), 'final': final.astype(int).tolist(),
        'best_objective': float(objective(spec,best_x)) if best_x is not None else None,
        'best_residuals': violations(spec,best_x).tolist() if best_x is not None else None,
        'best_key': [bool(violations(spec,best_x).sum()>0),float(violations(spec,best_x).sum()),
                     e.sign*float(objective(spec,best_x))] if best_x is not None else None,
        'mean': samples.mean(axis=0).tolist(), 'variance': samples.var(axis=0).tolist(),
        'residual_mean': (r/e.div).mean(axis=0).tolist(), 'violation_rates': (r > 0).mean(axis=0).tolist(),
        'objective_quantiles': np.quantile(f, [.1, .5, .9]).tolist(),
        'feasible_mass': float((r.sum(axis=1) == 0).mean()), 'stability_proxy': max(0., stability),
        'unique_states': len(unique), 'candidate_evaluations': evaluations, 'seconds': seconds,
        'transition_attempts': attempts, 'transition_changes': changes, 'q': qs or [],
        'sampling_scope': 'finite correlated snapshots; no equilibrium or semantic certification'}


def scip_pool(spec, seconds, seed, clamps=None, diversity=0., read_q=True):
    """Native conditional SCIP pool with no-good constraints; no fake Gibbs mass."""
    from ..solver_baselines.isolation import ensure_backend
    ensure_backend('scip')
    from pyscipopt import Model, quicksum
    from ..solver_baselines.milp import compile_model
    start = time.monotonic(); deadline = start+seconds
    ir = compile_model(spec)
    m = Model(); m.hideOutput(); m.setIntParam('parallel/maxnthreads', 1)
    m.setIntParam('randomization/randomseedshift', int(seed % (2**31-1)))
    vs = [m.addVar(vtype='B') for _ in ir.costs]
    for a, op, rhs in ir.rows:
        lhs = quicksum(v*vs[k] for k, v in a.items())
        m.addCons(lhs <= rhs if op == 'le' else lhs >= rhs if op == 'ge' else lhs == rhs)
    for i, value in (clamps or {}).items():
        m.addCons(quicksum(w*vs[k] for k, (j, w) in enumerate(ir.sites) if j == int(i)) == int(value)-int(ir.lower[int(i)]))
    rng = np.random.default_rng(seed)
    costs = np.array(ir.costs, dtype=float)
    if diversity:
        costs += rng.uniform(-1, 1, len(costs))*diversity*max(1, np.max(abs(costs), initial=1))
    m.setObjective(quicksum(float(c)*v for c, v in zip(costs, vs)), 'minimize')
    states = []; statuses = []; bounds = []; decision_count = len(ir.sites)
    while time.monotonic() < deadline and len(states) < 16:
        m.setRealParam('limits/time', max(.001, deadline-time.monotonic()))
        m.optimize(); statuses.append(str(m.getStatus()))
        b = m.getDualbound(); bounds.append(float(b) if math.isfinite(b) and abs(b) < 1e19 else None)
        sol = m.getBestSol()
        if sol is None:
            break
        bits = [int(m.getSolVal(sol, v) > .5) for v in vs]
        states.append(ir.decode(bits))
        if not decision_count:
            break
        m.freeTransform()
        m.addCons(quicksum(1-vs[k] if bits[k] else vs[k] for k in range(decision_count)) >= 1)
    if not states:
        return {'x': None, 'states': [], 'final': [], 'mean': [], 'variance': [], 'q': [],
            'residual_mean': [], 'violation_rates': [], 'objective_quantiles': [], 'feasible_mass': 0.,
            'stability_proxy': 0., 'unique_states': 0, 'candidate_evaluations': 0,
            'seconds': time.monotonic()-start, 'solver_statuses': statuses, 'solver_bounds': bounds}
    sign = 1 if spec['objective']['sense'] == 'min' else -1
    best = min(states, key=lambda x: (float(violations(spec, x).sum()), sign*float(objective(spec, x))))
    result = summarize(spec, np.array(states), np.array(states), best, seconds=time.monotonic()-start)
    if read_q:
        energy=Energy(spec); xs=np.array(states); qs=[]
        bits=encode_states(spec,xs)
        for k,(i,w) in enumerate(ir.sites[:16]):
            if time.monotonic()>=deadline:
                break
            zero,one=xs.copy(),xs.copy()
            zero[:,i]-=bits[:,k]*w;one[:,i]+=(1-bits[:,k])*w
            q=conditional_probability(energy.values(zero)[2],energy.values(one)[2],.5)
            qs.append({'bit':k,'variable':i,'weight':w,'conditional_q_mean':float(q.mean())})
        result['q']=qs
    result['seconds']=time.monotonic()-start
    result.update(sampling_scope='conditional optimization pool; NOT Boltzmann frequencies',
                  solver_statuses=statuses, solver_bounds=bounds, bound_scope='probe objective incl perturbation; not original bound')
    return result


def run_window(spec, seconds, seed, backend, chains, **kwargs):
    if backend == 'scip':
        return scip_pool(spec, seconds, seed, clamps=kwargs.get('clamps'),
                         diversity=kwargs.get('diversity', 0.), read_q=kwargs.get('read_q', True))
    kwargs.pop('diversity', None)
    return sample(spec, seconds, seed, chains, method=backend, **kwargs)


def compact(row, read_q=True):
    return {k: v for k, v in row.items() if k not in ('final',) and (read_q or k != 'q')}


def hypothesis_disagreement(spec, alternatives, states, read_q=True):
    if not alternatives or not states:
        return 0.
    x = np.array(states); baseline = Energy(spec); f, r, _ = baseline.values(x)
    scores = []
    for alt in alternatives:
        if alt['variables'] != spec['variables']:
            continue
        af, ar, _ = Energy(alt, scale=baseline.scale).values(x)
        feasible = np.mean((r.sum(axis=1) == 0) != (ar.sum(axis=1) == 0))
        pairs = np.arange(max(0, len(x)-1))
        ranks = np.mean(np.sign(baseline.sign*(f[pairs+1]-f[pairs])) !=
                        np.sign((1 if alt['objective']['sense']=='min' else -1)*(af[pairs+1]-af[pairs]))) if len(pairs) else 0.
        scores.append(max(float(feasible), float(ranks)))
    return max(scores, default=0.)


def probe(spec, mechanism, seconds, seed, backend, chains, base, alternatives, cfg, read_q=True):
    start = time.monotonic(); deadline = start+seconds
    rng = np.random.default_rng(seed)
    initial = base.get('final') or None
    variance = base.get('variance') or [0]*len(spec['variables'])
    targets = sorted(range(len(variance)), key=lambda i: (-variance[i], i))
    target = next((i for i in targets if spec['variables'][i]['lower'] < spec['variables'][i]['upper']), None)
    windows = []; metadata = {}; scale = objective_scale(spec)
    plans = []
    if mechanism == 'M1':
        plans = [(spec, {'temperature': t, 'diversity': t}, str(t)) for t in cfg['temperatures']]
    elif mechanism == 'M2' and target is not None:
        plans = [(spec, {'clamps': {target: v}}, str(v)) for v in
                 (spec['variables'][target]['lower'], spec['variables'][target]['upper'])]
        metadata['clamped_variable'] = target
    elif mechanism == 'M3':
        rates = base.get('violation_rates', [])
        if backend == 'scip':
            return {'mechanism': mechanism, 'status': 'unsupported_native_hard_constraint_penalty',
                    'windows': [], 'seconds': time.monotonic()-start, 'score_features': [0., 0., 0., 0.]}
        if rates:
            k = max(range(len(rates)), key=lambda i: (rates[i], -i))
            metadata['constraint_id'] = (spec['constraints']+spec['logic'])[k]['id']
            for mult in cfg['penalty_multipliers']:
                w = np.full(len(rates), 3.); w[k] *= mult
                plans.append((spec, {'weights': w}, str(mult)))
    elif mechanism == 'M4' and alternatives:
        # Find variable whose conditional energy response differs most across
        # hypotheses. Target search is bounded and charged to the same deadline.
        states = np.array(base.get('states') or [])
        if len(states):
            best = (-1., None)
            for i in targets[:16]:
                if time.monotonic() >= deadline:
                    break
                lo, hi = spec['variables'][i]['lower'], spec['variables'][i]['upper']
                if lo == hi:
                    continue
                zero, one = states.copy(), states.copy(); zero[:, i] = lo; one[:, i] = hi
                predictions = []
                for model in [spec]+alternatives[:2]:
                    ev = Energy(model, scale=scale)
                    de = ev.values(one)[2]-ev.values(zero)[2]
                    predictions.append(conditional_probability(0., de, .5) if read_q else np.sign(de))
                d = float(np.mean(np.max(predictions, axis=0)-np.min(predictions, axis=0)))
                if d > best[0]:
                    best = (d, i)
            target = best[1]; metadata['conditional_disagreement_proxy'] = best[0]
        if target is not None:
            metadata['clamped_variable'] = target
            for i, model in enumerate([spec]+alternatives[:2]):
                for v in (spec['variables'][target]['lower'], spec['variables'][target]['upper']):
                    plans.append((model, {'clamps': {target: v}, 'scale': scale}, f'hypothesis-{i}-value-{v}'))
    if not plans:
        plans = [(spec, {}, 'no_target_baseline')]
        metadata['fallback'] = 'no_valid_target_or_hypothesis'
    for j, (model, controls, label) in enumerate(plans):
        remaining = deadline-time.monotonic()
        if remaining <= 0:
            break
        controls = dict(controls, initial=initial, read_q=read_q)
        row = run_window(model, remaining/(len(plans)-j), seed, backend, chains, **controls)
        windows.append(dict(label=label, model_index=0 if model is spec else 1+next(i for i,v in enumerate(alternatives) if v is model),
                            **compact(row, read_q)))
    states = [x for w in windows for x in w['states']]
    stability = float(np.mean([w['stability_proxy'] for w in windows])) if windows else 0.
    old = {tuple(x) for x in base.get('states', [])}
    novelty = sum(tuple(x) not in old for x in states)/max(1, len(states))
    disagreement = hypothesis_disagreement(spec, alternatives, states, read_q)
    elapsed = time.monotonic()-start
    response = None
    if len(windows) >= 2 and len(windows[0].get('mean', [])) == len(windows[1].get('mean', [])):
        response = (np.array(windows[1]['mean'])-np.array(windows[0]['mean'])).tolist()
    metadata['variable_response_second_minus_first'] = response
    if mechanism == 'M3' and len(windows) == 2:
        metadata['normalized_residual_response'] = ((np.array(windows[1]['residual_mean'])-
            np.array(windows[0]['residual_mean']))/math.log(16)).tolist()
    return {'mechanism': mechanism, 'status': 'complete', 'windows': windows,
            'seconds': elapsed, 'metadata': metadata,
            'score_features': [disagreement, stability, novelty, min(1., elapsed/max(seconds, .001))]}


def evidence(spec, arm, seconds, seed, backend, chains, base, alternatives, cfg):
    read_q = arm != 'no_q'; start = time.monotonic()
    if arm in ('C0', 'C1', 'C2', 'C3'):
        row = run_window(spec, seconds, seed, backend, chains, initial=base.get('final') or None, read_q=read_q)
        return {'probes': [], 'passive': compact(row), 'seconds': time.monotonic()-start}
    kinds = ['M1', 'M2', 'M3', 'M4'] if arm not in ('M1', 'M2', 'M3', 'M4') else [arm]
    adaptive = arm in ('U2', 'B1', 'B2', 'no_memory', 'no_q', 'event')
    reports = []; routing = []
    for j, kind in enumerate(kinds):
        remaining = seconds-(time.monotonic()-start)
        if remaining <= 0:
            break
        allocation = min(seconds/10, remaining) if adaptive else remaining/(len(kinds)-j)
        reports.append(probe(spec, kind, allocation, seed+j, backend, chains, base, alternatives, cfg, read_q))
    if adaptive:
        valid = [r for r in reports if r['status'] == 'complete']
        coeff = cfg['router_coefficients']
        scores = [sum(a*b for a, b in zip(coeff[:3], r['score_features'][:3]))-coeff[3]*r['score_features'][3] for r in valid]
        if valid:
            weights = (np.ones(len(scores)) if not alternatives or max(scores)-min(scores)<.05
                       else np.exp(np.array(scores)-max(scores)))
            weights /= weights.sum()
            reserved = max(0., seconds-(time.monotonic()-start))
            for j, (row, weight, score) in enumerate(zip(valid, weights, scores)):
                allocation = min(reserved*float(weight), max(0., seconds-(time.monotonic()-start)))
                routing.append({'mechanism': row['mechanism'], 'score': score, 'seconds_reserved': allocation})
                if allocation > .001:
                    reports.append(probe(spec, row['mechanism'], allocation, seed+10+j, backend, chains, base, alternatives, cfg, read_q))
    return {'probes': reports, 'routing': routing, 'seconds': time.monotonic()-start,
            'scope': 'source-model evidence only; no claim of semantic validity'}
