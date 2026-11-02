"""Matched-context, factor-resolved evidence. No evaluator or task labels."""
import itertools
import time
import numpy as np
from qihc.generic.feedback import Energy, run_window
from qihc.generic.bounded import objective, violations


def quartets(context,i,j):
    xs=np.tile(context,(4,1))
    xs[:,i]=[0,1,0,1];xs[:,j]=[0,0,1,1]
    return xs


def readout(model,xs,scale):
    e=Energy(model,scale=scale);f,r,total=e.values(xs)
    objective_energy=e.sign*f/scale
    penalties=(r/e.div)**2*e.weights
    contrast=lambda a: (a[3]-a[1]-a[2]+a[0]).tolist()
    pc=np.asarray(contrast(penalties));nonzero=np.flatnonzero(np.abs(pc)>1e-12)
    selected=sorted(map(int,nonzero),key=lambda k:(-abs(pc[k]),k))[:8]
    ids=[c['id'] for c in model['constraints']+model['logic']]
    return {'objective_values':f.tolist(),'feasible':(r.sum(axis=1)==0).tolist(),
            'objective_energy':objective_energy.tolist(),'objective_contrast':contrast(objective_energy),
            'penalty_contrasts':[float(pc[k]) for k in selected],
            'factor_ids':[ids[k] for k in selected],
            'omitted_nonzero_factors':int(len(nonzero)-len(selected)),
            'remaining_penalty_contrast_sum':float(pc.sum()-sum(pc[k] for k in selected)),
            'total_energy':total.tolist(),'total_contrast':contrast(total),
            'conditional_log_odds_T_0_5':[-float(total[1]-total[0])/.5,-float(total[3]-total[2])/.5]}


def probe(model,contract,arm,base,seconds,seed,chains,quotes):
    start=time.monotonic();deadline=start+seconds
    packet={'scope':'source-only hypothesis vs generated model; not semantic truth','cards':[], 'provider':arm}
    if model is None:return dict(packet,status='probe_unavailable_invalid_model')
    if not contract or contract['variables']!=model['variables']:
        return dict(packet,status='no_compatible_source_hypothesis')
    if any((v['lower'],v['upper'])!=(0,1) for v in model['variables']):
        return dict(packet,status='nonbinary_probe_unsupported')
    n=len(model['variables']);rng=np.random.default_rng(seed)
    if arm=='S':
        run_window(model,max(.001,deadline-time.monotonic()),seed,'pbit',chains)
        return dict(packet,status='source_only_no_numeric_evidence')
    pairs=list(itertools.combinations(range(n),2))
    if not pairs:return dict(packet,status='no_pair')
    # A sees every quadratic pair analytically; the sampling groups get 32 candidates.
    pair_order=rng.permutation(len(pairs))
    candidates=[pairs[int(k)] for k in pair_order[:len(pairs) if arm=='A' else 32]]
    scale=Energy(model).scale
    zero=np.zeros(n,dtype=int)
    def priority(pair):
        xs=quartets(zero,*pair)
        a=readout(model,xs,scale);b=readout(contract,xs,scale)
        return (sum(x!=y for x,y in zip(a['feasible'],b['feasible']))
                +sum(np.sign(np.diff(a['objective_energy']))!=np.sign(np.diff(b['objective_energy'])))
                +min(1.,abs(a['objective_contrast']-b['objective_contrast'])))
    ranked=[]
    for pair in candidates:
        if time.monotonic()>=deadline:break
        ranked.append((float(priority(pair)),pair))
    if arm=='R':targets=[p for _,p in ranked[:2]]
    else:targets=[p for _,p in sorted(ranked,key=lambda x:(-x[0],x[1]))[:2]]
    packet['candidate_pairs_scored']=len(ranked)
    provider='scip' if arm=='X' else 'metropolis' if arm=='M' else 'pbit'
    rows=[]
    contexts=[zero.tolist()]+list(base.get('states',[])) if arm in ('P','R') else [zero.tolist()]
    if arm=='A':
        contexts=[zero.tolist()]+rng.integers(0,2,(16,n)).tolist()
    else:
        plans=[(p,v) for p in targets for v in (0,1)]
        for pos,(pair,value) in enumerate(plans):
            left=deadline-time.monotonic()
            if left<=.02:break
            # Reserve the last third for deterministic evidence checking.
            row=run_window(model,max(.001,left*.65/(len(plans)-pos)),seed+pos,provider,chains,
                           clamps={pair[0]:value},read_q=True)
            rows.append({'target':pair,'clamp_value':value,'status':row.get('solver_statuses'),
                         'bounds':row.get('solver_bounds'),'candidate_evaluations':row.get('candidate_evaluations'),
                         'seconds':row.get('seconds')})
            contexts.extend(row.get('states',[]))
    packet['windows']=rows
    seen=set()
    # Keep complete cards and select the strongest concrete discrepancy.
    found=[]
    for pair in targets:
        for context in contexts:
            if time.monotonic()>=deadline:break
            xs=quartets(context,*pair)
            key=tuple(map(tuple,xs))
            if key in seen:continue
            seen.add(key)
            a=readout(model,xs,scale);b=readout(contract,xs,scale)
            mismatch=sum(x!=y for x,y in zip(a['feasible'],b['feasible']))
            mismatch+=int(np.count_nonzero(np.sign(np.diff(a['objective_energy']))!=np.sign(np.diff(b['objective_energy']))))
            if not mismatch:continue
            # Greedy bounded simplification; never claim globally minimal.
            checks=0
            for k in range(n):
                if k in pair or checks>=32 or time.monotonic()>=deadline:continue
                if not xs[0,k]:continue
                trial=xs.copy();trial[:,k]=0;checks+=1
                ta=readout(model,trial,scale);tb=readout(contract,trial,scale)
                tm=sum(x!=y for x,y in zip(ta['feasible'],tb['feasible']))+int(np.count_nonzero(np.sign(np.diff(ta['objective_energy']))!=np.sign(np.diff(tb['objective_energy']))))
                if tm:xs,a,b,mismatch=trial,ta,tb,tm
            found.append({'pair':list(pair),'states':xs.tolist(),'source_quotes':quotes,
                          'model':a,'hypothesis':b,'common_objective_scale':scale,
                          'mismatch_count':mismatch,'simplification_checks':checks})
    found.sort(key=lambda x:(-x['mismatch_count'],x['pair'],x['states']))
    chosen=[]
    for card in found:
        if card['pair'] not in [c['pair'] for c in chosen]:chosen.append(card)
        if len(chosen)==2:break
    packet.update(cards=chosen,status='complete',contexts_checked=len(seen),seconds=time.monotonic()-start)
    return packet
