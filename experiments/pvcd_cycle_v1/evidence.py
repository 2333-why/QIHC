"""Training labels only. Sample both sides; certify every card independently."""
import time
import numpy as np
from qihc.generic.bounded import violations, objective
from qihc.generic.feedback.search import sample

def compatible(model, gold):
    return model is not None and [v['name'] for v in model['variables']] == [v['name'] for v in gold['variables']]

def cards_from_states(model, gold, states, limit=6):
    if not compatible(model, gold) or not len(states): return []
    xs = np.unique(np.asarray(states, dtype=int), axis=0)
    gf = violations(gold, xs).sum(axis=1) == 0
    mf = violations(model, xs).sum(axis=1) == 0
    buckets = [[], [], []]
    for i in np.flatnonzero(gf != mf):
        buckets[int(gf[i])].append(dict(kind='feasibility', x=xs[i].tolist(),
            correct_feasible=bool(gf[i]), candidate_feasible=bool(mf[i])))
    both = xs[gf & mf]
    if len(both)>1:
        def values(m): return objective(m,both)*(1 if m['objective']['sense']=='min' else -1)
        gv,mv = values(gold),values(model)
        for i in range(len(both)):
            for j in np.flatnonzero(np.sign(gv[i]-gv[i+1:]) != np.sign(mv[i]-mv[i+1:]))[:limit]:
                j = int(j)+i+1
                buckets[2].append(dict(kind='objective',x=both[i].tolist(),y=both[j].tolist(),
                    gold_order=int(np.sign(gv[i]-gv[j])),model_order=int(np.sign(mv[i]-mv[j]))))
            if len(buckets[2])>=limit: break
    # Round robin prevents numerous loose-constraint witnesses hiding tight ones or objective ties.
    result=[]
    for i in range(limit):
        for b in buckets:
            if i<len(b): result.append(b[i])
            if len(result)==limit: return result
    return result

def collect(model, gold, seed, cfg, uniform=False):
    if not compatible(model,gold): return [], []
    rng=np.random.default_rng(seed); uniform_rng=np.random.default_rng(seed+1000003)
    states=[]; trace=[]
    deadline=time.monotonic()+cfg['probe_seconds_per_round']
    for k in range(cfg['probe_calls']):
        spec=model if k%2==0 else gold
        # Same alternating domains and endpoint interventions for both sources.
        clamps={}
        if k>=2:
            i=int(rng.integers(len(spec['variables'])))
            clamps[i]=spec['variables'][i]['lower' if k%2==0 else 'upper']
        started=time.monotonic()
        if uniform:
            xs=np.column_stack([uniform_rng.integers(v['lower'],v['upper']+1,cfg['chains']) for v in spec['variables']])
            for i,v in clamps.items(): xs[:,i]=v
            updates=0
        else:
            left=max(0.,deadline-time.monotonic())
            row=sample(spec,left/(cfg['probe_calls']-k),seed+k,chains=cfg['chains'],
                       temperature=(.25,.5,1.)[k%3],clamps=clamps,read_q=False)
            xs=row['final']; updates=row.get('transition_attempts',0)
        states.extend(np.asarray(xs,dtype=int).tolist())
        trace.append(dict(side='candidate' if k%2==0 else 'gold',clamps=clamps,
                          states=len(xs),updates=updates,seconds=time.monotonic()-started))
    return states,trace
