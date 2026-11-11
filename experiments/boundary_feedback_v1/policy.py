"""Candidate selection by source-only witness judgments, never by gold labels."""
import copy
import itertools
import json
import time
import numpy as np
from qihc.generic.bounded import objective, violations
from qihc.generic.feedback.search import Energy, conditional_probability, sample, scip_pool
from qihc.modeling.active_protocol_v2 import model_prompt, parse_model, compact
from qihc.modeling.protocol import load_object
from experiments.feedback_active_v3.runtime import seed_for

def shared(item, ask):
    base = item['baseline']
    if base is None:
        return dict(model=None, candidate=None, action='baseline_invalid')
    prompt = model_prompt(item['case']) + '\nAudit this draft against the ORIGINAL problem. Independently recompute entity-index mappings, every constraint and the objective. Return the complete corrected model, or {"keep":true} if no supported correction.\nDRAFT:\n' + json.dumps(compact(base))
    text = ask(prompt, seed_for(item['case']['id'], item['seed'], 'candidate'))
    try:
        if load_object(text) == {'keep': True}:
            return dict(model=base, candidate=base, action='candidate_keep', text=text)
        candidate = parse_model(text, item['case'])
        if candidate['variables'] != base['variables']:
            return dict(model=base, candidate=None, action='candidate_domain_change', text=text)
    except (ValueError, KeyError, TypeError, IndexError) as exc:
        return dict(model=base, candidate=None, action='candidate_parse_error', text=text, error=str(exc))
    return dict(model=base, candidate=candidate, action='candidate_ready', text=text)

def signatures(model, xs):
    x = np.asarray(xs)
    feasible = violations(model, x).sum(axis=1) == 0
    cost = objective(model, x) * (1 if model['objective']['sense'] == 'min' else -1)
    return feasible, cost

def collect(models, arm, seed, seconds):
    """Same total search budget. Q adds conditional response and soft penalties."""
    started = time.monotonic()
    end = started + seconds
    states, windows = [], []
    count = 4 if arm in ('S', 'P') else 8
    target = None
    for j in range(count):
        remaining = end-time.monotonic()
        if remaining <= .001:
            break
        model = models[j % 2]
        controls = {}
        if arm == 'Q' and j >= 2:
            controls['temperature'] = (1., .15, .5)[min(2, (j-2)//2)]
            if j in (2, 3):
                controls['weights'] = np.full(len(model['constraints'])+len(model['logic']), .3)
            if j >= 4:
                if target is None and states:
                    # Conditional heat-bath responses select the variable that
                    # most distinguishes the two proposed energy landscapes.
                    xs = np.asarray(sorted(set(map(tuple, states)))[:64])
                    energies = [Energy(m) for m in models]
                    best = (-1., None)
                    for i, var in enumerate(model['variables']):
                        if time.monotonic() >= end or var['lower'] == var['upper']:
                            continue
                        lo, hi = xs.copy(), xs.copy()
                        lo[:, i], hi[:, i] = var['lower'], var['upper']
                        response = [conditional_probability(e.values(lo)[2], e.values(hi)[2], .5) for e in energies]
                        gain = float(np.mean(abs(response[0]-response[1])))
                        if gain > best[0]:
                            best = (gain, i)
                    target = best[1]
                if target is not None:
                    bound = 'lower' if j < 6 else 'upper'
                    controls['clamps'] = {target: model['variables'][target][bound]}
        allocation = max(0., end-time.monotonic())/(count-j)
        if allocation <= .001:
            break
        if arm == 'S':
            row = scip_pool(model, allocation, seed+j, diversity=0. if j < 2 else .2, read_q=False)
        else:
            row = sample(model, allocation, seed+j, chains=32, read_q=False, **controls)
        states.extend(row.get('states', []))
        states.extend(row.get('final', []))
        if row.get('x') is not None:
            states.append(row['x'])
        windows.append(dict(index=j, seconds=row['seconds'], unique_states=row['unique_states'],
                            candidate_evaluations=row.get('candidate_evaluations', 0),
                            target=target, temperature=controls.get('temperature', .5),
                            softened='weights' in controls, clamps=controls.get('clamps')))
    unique = sorted(set(map(tuple, states)))
    return unique, dict(seconds=time.monotonic()-started, windows=windows, pooled_states=len(unique))

def witnesses(models, states, excluded=(), limit=8):
    """Prioritize feasibility conflicts, order conflicts, then diverse controls."""
    xs = [list(x) for x in sorted(set(map(tuple, states))-set(map(tuple, excluded)))]
    if not xs:
        return [], []
    (a, av), (b, bv) = [signatures(m, xs) for m in models]
    selected = list(np.flatnonzero(a != b)[:limit])
    for i, j in itertools.combinations(range(len(xs)), 2):
        if len(selected) >= limit:
            break
        if a[i] and a[j] and b[i] and b[j] and np.sign(av[i]-av[j]) != np.sign(bv[i]-bv[j]):
            missing = [k for k in (i, j) if k not in selected]
            if len(selected)+len(missing) <= limit:
                selected.extend(missing)
    # Include each model's incumbent when there is space, then farthest states.
    for feasible, cost in ((a, av), (b, bv)):
        valid = np.flatnonzero(feasible)
        if len(valid):
            k = int(valid[np.argmin(cost[valid])])
            if k not in selected and len(selected) < limit:
                selected.append(k)
    while len(selected) < min(limit, len(xs)):
        available = [i for i in range(len(xs)) if i not in selected]
        k = max(available, key=lambda i: min(sum(u != v for u, v in zip(xs[i], xs[j])) for j in selected)) if selected else available[0]
        selected.append(k)
    chosen = [xs[i] for i in selected]
    pairs = list(itertools.combinations(range(len(chosen)), 2))
    return chosen, pairs

def judge_prompt(case, xs, pairs):
    # Do not expose candidate models, their predictions, or the probe identity.
    return ('Read ORIGINAL only. Independently evaluate each assignment against ALL stated requirements. '
            'Use true/false for feasible; null if uncertain. For each requested pair return -1 if the first '
            'is better for the ORIGINAL objective, 1 if worse, 0 if tied; null if either infeasible or uncertain. '
            'Return JSON only: {"feasible":[true,false,null,...],"orders":[-1,0,1,null,...],'
            '"quote":"a nonempty verbatim relevant substring from ORIGINAL"}. '
            'Do not infer correctness from how a state was generated.\nVariable order: '+json.dumps(case['variable_names'])+
            '\nORIGINAL:\n'+case['question']+'\nAssignments: '+json.dumps(xs)+'\nPairs (assignment indices): '+json.dumps(pairs))

def assess(text, case, models, xs, pairs):
    try:
        obj = load_object(text)
        if set(obj) != {'feasible', 'orders', 'quote'}:
            raise ValueError('judge envelope')
        if not isinstance(obj['quote'], str) or not obj['quote'].strip() or obj['quote'] not in case['question']:
            raise ValueError('judge quote')
        labels, orders = obj['feasible'], obj['orders']
        if not isinstance(labels, list) or not isinstance(orders, list) or len(labels) != len(xs) or len(orders) != len(pairs):
            raise ValueError('judge dimensions')
        if any(x is not None and type(x) is not bool for x in labels):
            raise ValueError('feasibility labels')
        if any(x is not None and (type(x) is not int or x not in (-1, 0, 1)) for x in orders):
            raise ValueError('order labels')
        if any(o is not None and (labels[i] is not True or labels[j] is not True) for (i, j), o in zip(pairs, orders)):
            raise ValueError('ordering of infeasible/unknown assignments')
    except (ValueError, TypeError, KeyError) as exc:
        return dict(status='judge_parse_error', error=str(exc), text=text)
    errors, covered = [], sum(v is not None for v in labels)+sum(v is not None for v in orders)
    for model in models:
        feasible, cost = signatures(model, xs)
        error = sum(bool(feasible[i]) != v for i, v in enumerate(labels) if v is not None)
        for (i, j), label in zip(pairs, orders):
            if label is not None:
                error += int(not (feasible[i] and feasible[j]) or int(np.sign(cost[i]-cost[j])) != label)
        errors.append(int(error))
    return dict(status='judged', errors=errors, covered=covered, labels=obj, text=text)

def decide(item, arm, base, ask, seconds=20, cache=None):
    old = base['model']
    if arm == 'A':
        return dict(model=item['model'], action='original')
    if arm == 'B':
        return dict(model=old, action='compile_baseline')
    candidate = base['candidate']
    if candidate is None:
        return dict(model=old, action=base['action'])
    if candidate == old:
        return dict(model=old, action='candidate_keep' if base['action'] == 'candidate_keep' else 'candidate_unchanged')
    cid, seed = item['case']['id'], item['seed']
    models = [old, candidate]
    if arm == 'C':
        reviews = []
        for phase in ('discover', 'validate'):
            # Counterbalance presentation order; accept only two explicit approvals.
            reverse = seed_for(cid, seed, phase) % 2
            options = models[::-1] if reverse else models
            text = ask(model_prompt(item['case'])+'\nCompare these two models with the original. Return ONLY {"choice":0} or {"choice":1}, or {"choice":null} if unsure or both wrong.\n'+json.dumps([compact(m) for m in options]), seed_for(cid, seed, 'review', phase))
            try:
                obj = load_object(text)
                if set(obj) != {'choice'} or type(obj['choice']) is not int or obj['choice'] not in (0, 1):
                    raise ValueError('Expected explicit choice')
                accepted = obj['choice'] == (0 if reverse else 1)
            except (ValueError, TypeError, KeyError):
                return dict(model=old, action='source_review_abstain', reviews=reviews+[text])
            reviews.append(text)
            if not accepted:
                return dict(model=old, action='source_review_keep', reviews=reviews)
        return dict(model=candidate, action='accepted', reviews=reviews)
    evidence, excluded = [], []
    for phase in ('discover', 'validate'):
        states, telemetry = collect(models, arm, seed_for(cid, seed, phase), seconds/2)
        xs, pairs = witnesses(models, states, excluded)
        if not xs:
            return dict(model=old, action='no_fresh_witness', evidence=evidence, probe=telemetry)
        text = ask(judge_prompt(item['case'], xs, pairs), seed_for(cid, seed, 'judge', phase))
        verdict = assess(text, item['case'], models, xs, pairs)
        evidence.append(dict(phase=phase, assignments=xs, pairs=pairs, probe=telemetry, verdict=verdict))
        if verdict['status'] != 'judged':
            return dict(model=old, action=verdict['status'], evidence=evidence)
        if verdict['covered'] < 2:
            return dict(model=old, action='judge_unknown', evidence=evidence)
        if verdict['errors'][1] > 0:
            return dict(model=old, action='candidate_contradicted', evidence=evidence)
        if phase == 'discover' and verdict['errors'][0] == 0:
            return dict(model=old, action='no_judged_advantage', evidence=evidence)
        excluded.extend(xs)
    return dict(model=candidate, action='accepted', evidence=evidence)
