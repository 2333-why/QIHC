"""Budgeted source-local diagnostics. No gold, family labels or answer access."""
from __future__ import annotations

import copy
import math
import time

import numpy as np

from .search import sample, scip_pool
from ..bounded import objective, violations


def check_residual(check, states):
    r = np.asarray(states) @ np.asarray(check['a']) - check['rhs']
    return abs(r) if check['op'] == 'eq' else np.maximum(0, r if check['op'] == 'le' else -r)


def diagnostics(spec, checks):
    """Deterministic, auditable source hypotheses, not a whole-model oracle."""
    cards = []
    for c in checks:
        if c['kind'] == 'constraint':
            found = any(all(row[k] == c[k] for k in ('a', 'op', 'rhs')) for row in spec['constraints'])
            if not found:
                cards.append(dict(kind='missing_literal_form', check_id=c['id'], quote=c['quote'], confidence=c['confidence'],
                                  claim='No identical row; an equivalent encoding may exist. Inspect source.', severity='hypothesis'))
        else:
            for key in ('sense', 'linear', 'quadratic'):
                if spec['objective'][key] != c[key]:
                    cards.append(dict(kind='objective_alignment', check_id=c['id'], quote=c['quote'], field=key,
                                      confidence=c['confidence'], claim='Candidate differs from a fallible source interpretation', severity='hypothesis'))
    return cards


def _actions(spec, checks, maximum, rng):
    score = np.zeros(len(spec['variables']))
    for c in checks:
        if c['kind'] == 'constraint': score += (np.array(c['a']) != 0) * c['confidence']
    order = sorted(range(len(score)), key=lambda i: (-score[i], i))
    actions = [{}]
    for i in order:
        v = spec['variables'][i]
        for value in sorted({v['lower'], v['upper']}): actions.append({i: value})
        if len(actions) >= maximum: break
    # Include paired conditions without spending most of a short budget on them.
    for i, j in zip(order[::2], order[1::2]):
        actions.append({i: spec['variables'][i]['upper'], j: spec['variables'][j]['lower']})
    if len(actions) > maximum:
        singles = actions[1:]; indices = rng.choice(len(singles), maximum - 1, replace=False)
        actions = [{}] + [singles[int(i)] for i in sorted(indices)]
    return actions


def _witnesses(spec, checks, states):
    if len(states) == 0: return [], 0
    xs = np.asarray(states, dtype=int); feasible = violations(spec, xs).sum(axis=1) == 0
    cards = []
    for c in checks:
        if c['kind'] != 'constraint': continue
        conflict = feasible & (check_residual(c, xs) > 0)
        ids = np.flatnonzero(conflict)
        if len(ids):
            i = int(ids[0])
            cards.append(dict(kind='candidate_feasible_source_conflict', check_id=c['id'], quote=c['quote'],
                x=xs[i].tolist(), observed_rate=float(conflict.mean()), independent_final_chains=len(xs),
                confidence=c['confidence'], severity='source_hypothesis',
                claim='Candidate-feasible witness violates this extracted check; confirm the check from source'))
    return cards, len({c['check_id'] for c in cards})


def _scip_violation(spec, check, seconds, seed):
    """Maximize violation of a source requirement while satisfying candidate."""
    from ..solver_baselines.registry import solve
    probe = copy.deepcopy(spec)
    a = check['a']; sign = -1 if check['op'] == 'ge' else 1
    probe['objective'] = dict(sense='max', linear=[sign * v for v in a], quadratic=[])
    row = solve(probe, 'scip', seconds/2 if check['op']=='eq' else seconds, seed)
    states = [row['x']] if row.get('x') is not None else []
    if check['op']=='eq':
        probe['objective']['linear']=[-v for v in a]
        opposite=solve(probe,'scip',seconds/2,seed+1)
        if opposite.get('x') is not None: states.append(opposite['x'])
        row=dict(row,opposite_status=opposite.get('status'))
    return states, row


def _reloop(spec, checks, deadline, seed):
    """CPT/OPT-inspired response diagnostics, explicitly an IR adaptation."""
    from ..solver_baselines.registry import solve
    cards = []; probes = []; remaining = max(0.001, deadline - time.monotonic())
    base = solve(spec, 'scip', min(remaining / 4, 1.5), seed)
    for k, c in enumerate(checks):
        remaining = deadline - time.monotonic()
        if remaining <= 0.02 or len(probes) >= 4: break
        model = copy.deepcopy(spec)
        if c['kind'] == 'constraint':
            candidates = [r for r in model['constraints'] if r['a'] == c['a']]
            if not candidates:
                cards.append(dict(kind='unmapped_source_rule', check_id=c['id'], quote=c['quote'], severity='hypothesis'))
                continue
            row = candidates[0]; original = row['rhs']; step = max(1, abs(original) // 2)
            row['rhs'] += -step if row['op'] == 'le' else step
        else:
            nonzero = [i for i, a in enumerate(model['objective']['linear']) if a]
            if not nonzero: continue
            model['objective']['linear'][nonzero[0]] *= 2
        result = solve(model, 'scip', min(remaining, max(.02, remaining / (4 - len(probes)))), seed + k + 1)
        changed = result.get('x') != base.get('x') or result.get('feasible') != base.get('feasible')
        probes.append(dict(check_id=c['id'], base_status=base.get('status'), status=result.get('status'), response_changed=changed))
        cards.append(dict(kind='parameter_response', check_id=c['id'], quote=c['quote'], response_changed=changed,
                          claim='No change can mean an inactive constraint or an uninformative perturbation; not proof of omission', severity='hypothesis'))
    return cards, probes


def probe(spec, checks, arm, cfg, seed):
    start = time.monotonic(); deadline = start + cfg['probe_seconds']; rng = np.random.default_rng(seed)
    if spec is None: return dict(cards=[], status='invalid_model', seconds=0, abstained=True)
    checks = checks.get('checks', []) if isinstance(checks, dict) else checks
    if arm == 'S': return dict(cards=[], status='self_review', seconds=0, abstained=False)
    if arm in ('A', 'D', 'B'):
        cards = diagnostics(spec, checks)
        # D adds execution feasibility diagnosis; B is source alignment only.
        if arm == 'D' and deadline - time.monotonic() > .01:
            from ..solver_baselines.registry import solve
            result = solve(spec, 'scip', max(.01, deadline - time.monotonic()), seed)
            cards.insert(0, dict(kind='execution_diagnostic', status=result.get('status'),
                                claim='Candidate solver status; source feasibility is not certified'))
        return dict(cards=cards[:cfg['max_cards']], status='complete', seconds=time.monotonic() - start,
                    abstained=not cards, method_scope='deterministic_IR_adaptation')
    if arm == 'R':
        cards, trace = _reloop(spec, checks, deadline, seed)
        return dict(cards=cards[:cfg['max_cards']], trace=trace, status='complete', seconds=time.monotonic() - start,
                    abstained=not cards, method_scope='ReLoop_CPT_OPT_inspired_IR_adaptation_not_native')
    cards, trace = [], []; seen = set()
    observations = {}; baseline = None
    if arm == 'X':
        linear = [c for c in checks if c['kind'] == 'constraint']
        for k, c in enumerate(linear[:4]):
            remaining = deadline - time.monotonic()
            if remaining <= .01: break
            states, row = _scip_violation(spec, c, remaining / max(1, 5 - k), seed + k)
            found, _ = _witnesses(spec, [c], states); cards += found
            trace.append(dict(check_id=c['id'], status=row.get('status'), bound=row.get('bound')))
        remaining = deadline - time.monotonic()
        if remaining > .01:
            pool = scip_pool(spec, remaining, seed + 100, read_q=False)
            found, _ = _witnesses(spec, checks, pool['states']); cards += found
        unique = {c['check_id']: c for c in cards}
        return dict(cards=list(unique.values())[:cfg['max_cards']], trace=trace, status='complete',
                    seconds=time.monotonic() - start, abstained=not cards, method_scope='active_violation_and_diverse_feasible_pool')
    if arm not in ('M', 'T', 'P0', 'P1', 'P2', 'PR'): raise ValueError('Unknown feedback arm: ' + arm)
    actions = [{}] if arm == 'P0' else _actions(spec, checks, cfg['candidate_actions'], rng)
    rewards = []; exploration = cfg['probe_seconds'] * .25

    def observe(action, seconds, temperature, suffix, confirm):
        nonlocal baseline
        phase_deadline = deadline if confirm else start + exploration
        remaining = phase_deadline - time.monotonic()
        if remaining <= .002: return 0
        kernel = sample
        if arm == 'T':
            from .tempering import sample_pt
            kernel = sample_pt
        result = kernel(spec, min(seconds, remaining), int((seed + suffix) % (2**31 - 1)), chains=cfg['chains'],
                        method='metropolis' if arm in ('M', 'T') else 'pbit', temperature=temperature, clamps=action, read_q=False)
        found, reward = _witnesses(spec, checks, result['final'])
        if confirm:
            final = np.asarray(result['final']); empirical = final.mean(axis=0)
            if not action and baseline is None: baseline = empirical
            shifts = []
            if baseline is not None:
                changes = abs(empirical - baseline)
                free = [i for i in np.argsort(-changes) if int(i) not in action][:3]
                shifts = [dict(variable=int(i), mean_shift=float(empirical[i]-baseline[i])) for i in free]
            for c in found:
                c.update(clamps={str(k): v for k, v in action.items()}, temperature=temperature,
                         conditional_mean_shifts=shifts,
                         sampling_scope='independent_final_chains_finite_time_not_equilibrium')
                entry = observations.setdefault(c['check_id'], dict(temperatures=set(), windows=0, rates=[]))
                entry['temperatures'].add(float(temperature)); entry['windows'] += 1; entry['rates'].append(c['observed_rate'])
                if arm=='T': c['sampling_scope']='replica_exchange_correlated_final_states_finite_time'
                if c['check_id'] not in seen: cards.append(c); seen.add(c['check_id'])
                elif action:
                    # Preserve an informative conditional response instead of
                    # permanently retaining the first un-clamped zero shift.
                    idx=next(i for i,old in enumerate(cards) if old['check_id']==c['check_id'])
                    old_strength=sum(abs(v['mean_shift']) for v in cards[idx].get('conditional_mean_shifts',[]))
                    if sum(abs(v['mean_shift']) for v in shifts)>old_strength: cards[idx]=c
        trace.append(dict(action={str(k): v for k, v in action.items()}, temperature=temperature,
                          phase='confirmation' if confirm else 'selection', seconds=result['seconds'],
                          evaluations=result['candidate_evaluations'], unique_states=result['unique_states'], reward=reward,
                          empirical_mean=result['mean'], feasible_mass=result['feasible_mass']))
        return reward

    for i, action in enumerate(actions):
        rewards.append(observe(action, exploration / len(actions), .5, i, False))
    # Select using exploratory rewards plus uncertainty and update between
    # independent confirmation windows; no selection state is itself a card.
    visits = np.ones(len(actions)); totals = np.asarray(rewards, dtype=float)
    temperatures = cfg['temperatures'] if arm in ('P2', 'M', 'PR') else [.5]
    deep_count = min(cfg['deep_actions'], len(actions)); total_windows = deep_count * len(temperatures)
    for step in range(deep_count):
        if step == 0: i = 0  # independent un-clamped confirmation baseline
        elif arm == 'PR': i = int(rng.integers(len(actions)))
        else:
            scores = totals / visits + np.sqrt(2 * np.log(2 + visits.sum()) / visits)
            i = int(np.argmax(scores))
        for ti, t in enumerate(temperatures):
            k = step * len(temperatures) + ti; before = len(seen)
            observe(actions[i], max(0, deadline - time.monotonic()) / max(1, total_windows - k), t, 1000 + k, True)
            totals[i] += len(seen) - before; visits[i] += 1
    usable = []
    for c in cards:
        obs = observations[c['check_id']]
        c.update(temperatures_confirmed=sorted(obs['temperatures']), confirming_windows=obs['windows'], observed_rates=obs['rates'])
        # This is a repeatability gate, not a calibrated semantic confidence.
        stable = len(obs['temperatures']) >= 2 if arm in ('P2','M','PR') else True
        c['repeatability_gate'] = 'pass' if stable else 'insufficient_temperature_confirmation'
        if stable: usable.append(c)
    return dict(cards=usable[:cfg['max_cards']], tentative_cards=cards, trace=trace, status='complete', seconds=time.monotonic() - start,
                abstained=not usable, method_scope='empirical_conditional_response',
                limitation='source interpretations remain fallible; no stationary-distribution claim')
