"""Resumable bounded-IR heat-bath search; policies never edit the input model.

All candidate states are ranked by original feasibility/objective, never energy.
Branch trials copy the identical state and share site/draw streams across actions.
"""
import copy
import hashlib
import json
import numpy as np
from qihc.generic.factor_search import encoding
from qihc.generic.feedback.search import Energy, conditional_probability

ACTIONS = ('hold', 'cool', 'heat', 'harden', 'balance')


def seed(*parts):
    return int(hashlib.sha256(json.dumps(parts).encode()).hexdigest()[:8], 16)


def key(best, spec):
    return (best['violation'] > 0, best['violation'],
            best['objective'] * (1 if spec['objective']['sense'] == 'min' else -1))


def inspect(spec, x, energy):
    f, r, e = energy.values(x)
    v = r.sum(axis=1)
    i = np.lexsort((energy.sign*f, v, v > 0))[0]
    best = dict(x=x[i].astype(int).tolist(), violation=float(v[i]), objective=float(f[i]))
    return best, f, r, e


def initial(spec, random_seed, chains=32):
    energy = Energy(spec)
    sites, lower = encoding(spec)
    bits = np.random.default_rng(random_seed).integers(2, size=(chains, len(sites)))
    x = np.tile(lower, (chains, 1))
    for k, (j, w) in enumerate(sites):
        x[:, j] += bits[:, k]*w
    best, f, r, _ = inspect(spec, x, energy)
    chain_best = [dict(x=x[i].tolist(),violation=float(r[i].sum()),objective=float(f[i])) for i in range(chains)]
    return dict(bits=bits.tolist(), x=x.tolist(), weights=energy.weights.tolist(),
                temperature=.5, best=best, chain_best=chain_best, steps=0, history=[])


def advance(spec, state, action, random_seed, steps=128):
    if action not in ACTIONS or steps < 1:
        raise ValueError('Invalid action or budget')
    state = copy.deepcopy(state)
    x = np.asarray(state['x'], dtype=int)
    bits = np.asarray(state['bits'], dtype=int)
    w = np.asarray(state['weights'], dtype=float)
    temp = state['temperature']
    if action == 'cool': temp = max(.05, temp/2)
    if action == 'heat': temp = min(2., temp*2)
    if action == 'harden': w = np.minimum(8., w*1.5)
    if action == 'balance' and len(w) > 1 and state['history']:
        rates = state['history'][-1]['violation_rates']
        lo, hi = int(np.argmin(rates)), int(np.argmax(rates))
        if rates[hi] > rates[lo]:
            delta = min(.5, w[lo]-2.125, 8.-w[hi])
            w[lo] -= delta; w[hi] += delta
    energy = Energy(spec, weights=w)
    sites, _ = encoding(spec)
    if not sites: raise ValueError('No mutable encoded bits')
    best, f, r, e = inspect(spec, x, energy)
    archive = state['best']
    chain_best = state['chain_best']
    if key(best, spec) < key(archive, spec): archive = best
    rng = np.random.default_rng(random_seed)
    mass = []; rates = np.zeros(len(w)); changes = 0; improvements = 0
    for _ in range(steps):
        k = int(rng.integers(len(sites))); j, weight = sites[k]
        trial = x.copy(); trial[:, j] += (1-2*bits[:, k])*weight
        nf, nr, ne = energy.values(trial)
        zero = bits[:, k] == 0
        q = conditional_probability(np.where(zero, e, ne), np.where(zero, ne, e), temp)
        change = (rng.random(len(x)) < q) != bits[:, k]
        bits[change, k] = 1-bits[change, k]
        x[change] = trial[change]; f[change] = nf[change]; r[change] = nr[change]; e[change] = ne[change]
        v = r.sum(axis=1)
        i = np.lexsort((energy.sign*f, v, v > 0))[0]
        candidate = dict(x=x[i].tolist(), violation=float(v[i]), objective=float(f[i]))
        for j in range(len(x)):
            chain_candidate = dict(x=x[j].tolist(),violation=float(v[j]),objective=float(f[j]))
            if key(chain_candidate,spec) < key(chain_best[j],spec): chain_best[j]=chain_candidate
        if key(candidate, spec) < key(archive, spec):
            archive = candidate; improvements += 1
        changes += int(change.sum()); rates += (r > 0).mean(axis=0)
        mass.append(float((v == 0).mean()))
    history = state['history'] + [dict(step=state['steps']+steps, action=action,
        temperature=temp, weights=w.tolist(), feasible_mass=float(np.mean(mass)),
        violation_rates=(rates/steps).tolist(), diversity=len(np.unique(x, axis=0))/len(x),
        change_rate=changes/(steps*len(x)), improvements=improvements,
        best=archive, chain_feasible=float(np.mean([b['violation']==0 for b in chain_best])),
        chain_quality=float(np.mean([quality(spec,b) for b in chain_best])),energy_mean=float(e.mean()))]
    return dict(bits=bits.tolist(), x=x.tolist(), weights=w.tolist(), temperature=temp,
                best=archive,chain_best=chain_best, steps=state['steps']+steps, history=history)


def quality(spec, b):
    """Bounded feasibility-first score; no optimum or raw energy needed."""
    from qihc.generic.factor_search import objective_scale
    if b['violation'] > 0: return -b['violation']/(1+b['violation'])
    sign=1 if spec['objective']['sense']=='min' else -1
    return 1.-sign*b['objective']/(4*objective_scale(spec))


def utility(spec, state):
    # Population progress only breaks ties in best-solution quality.
    return (quality(spec,state['best']),float(np.mean([quality(spec,b) for b in state['chain_best']])))


def compare(a,b):
    for x,y in zip(a,b):
        if abs(x-y)>1e-6:return 1 if x>y else -1
    return 0


def prompt(spec, state, trajectory=True):
    # No case ID, reference optimum, labels, or future rollout information.
    from qihc.modeling.active_protocol_v2 import compact
    context = dict(model=compact(spec), steps=state['steps'], temperature=state['temperature'],
                   weights=state['weights'], best=state['best'])
    if trajectory: context['history'] = state['history'][-4:]
    return ('Control a p-bit solver on this FIXED mathematical model. Do not edit the model. '
            'Choose one action: A hold; B halve temperature (min .05); C double temperature (max 2); '
            'D increase all penalties by 1.5 (max 8); E transfer up to .5 penalty from the least '
            'violated row to the most violated row (bounds 2.125..8). '
            'Prefer feasible solutions, then improve the original objective. Lower raw energy '
            'across different penalties does not imply improvement.\n'
            +json.dumps(context, separators=(',', ':'))+'\nAction:')


def rule(state):
    h = state['history'][-1]
    if h['feasible_mass'] < .1: return 'harden'
    if h['improvements'] == 0 and h['diversity'] < .25: return 'heat'
    return 'cool'


def preference(spec, state, random_seed, steps=128, select=2, confirm=4):
    def scores(action, phase, count):
        return [utility(spec, advance(spec, state, action, seed(random_seed, phase, i), steps))
                for i in range(count)]
    screen = {a:scores(a, 'selection', select) for a in ACTIONS}
    ranked = sorted(ACTIONS, key=lambda a:tuple(-np.mean(screen[a],axis=0))+(ACTIONS.index(a),))
    chosen, rejected = ranked[0], ranked[-1]
    a = scores(chosen, 'confirmation', confirm); b = scores(rejected, 'confirmation', confirm)
    directions=[compare(x,y) for x,y in zip(a,b)]
    # Reject ties and unstable labels; finite empirical confirmation, not proof.
    accepted = compare(np.mean(a,axis=0),np.mean(b,axis=0))>0 and directions.count(1)>=(confirm+1)//2 and -1 not in directions
    return dict(chosen=chosen, rejected=rejected, accepted=bool(accepted),
                selection=screen, confirmation=dict(chosen=a, rejected=b))
