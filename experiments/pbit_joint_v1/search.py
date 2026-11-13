"""Finite-budget conditional heat-bath control; no gold or reference in decisions."""
import copy
import json
import time
import numpy as np
from qihc.generic.feedback.search import sample, Energy,encode_states,conditional_probability
from qihc.generic.factor_search import encoding,objective_scale
from qihc.experiment_runtime.pbit_control import initial as old_initial, key, quality, seed

ACTIONS=('cool','heat','clamp')


def initial(spec, random_seed, chains):
    old=old_initial(spec,random_seed,chains)
    return dict(x=old['x'],best=old['best'],temperature=.5,steps=0,history=[],q=[],clamp_writes=0)


def intervention(spec,state):
    """Clamp one uncertain binary variable to the population's minority value."""
    mutable=[i for i,v in enumerate(spec['variables']) if v['lower']!=v['upper']]
    if len(mutable)<2:return {}
    q={r['variable']:r['conditional_q_mean'] for r in state['q']}
    eligible=[i for i in mutable if spec['variables'][i]['lower']==0 and spec['variables'][i]['upper']==1 and i in q]
    if not eligible:return {}
    i=min(eligible,key=lambda i:(abs(q[i]-.5),i))
    return {i:int(np.mean(np.asarray(state['x'])[:,i])<.5)}


def collapse_training_state(spec,state):
    """A legal, deliberately low-diversity training state; no optimum is consulted."""
    out=copy.deepcopy(state);out['x']=[list(state['best']['x']) for _ in state['x']]
    xs=np.asarray(out['x']);bits=encode_states(spec,xs);energy=Energy(spec)
    _,residual,e=energy.values(xs);qs=[]
    for k,(i,w) in enumerate(encoding(spec)[0]):
        other=xs.copy();other[:,i]+=(1-2*bits[:,k])*w
        _,_,ne=energy.values(other)
        q=conditional_probability(np.where(bits[:,k]==0,e,ne),np.where(bits[:,k]==0,ne,e),out['temperature'])
        qs.append(dict(bit=k,variable=i,weight=w,conditional_q_mean=float(q.mean())))
    out['q']=qs
    if out['history']:
        out['history'][-1]=dict(out['history'][-1],diversity=1/len(xs),change_rate=0.,
                               feasible_mass=float(np.mean(residual.sum(axis=1)==0)))
    return out


def advance(spec,state,action,random_seed,steps):
    if action not in ACTIONS:raise ValueError(action)
    out=copy.deepcopy(state);xs=np.asarray(state['x']);half=len(xs)//2
    if len(xs)<2 or len(xs)%2:raise ValueError('Use an even chain count')
    temperature=max(.05,state['temperature']/2)
    exploratory=min(2.,state['temperature']*2) if action=='heat' else temperature
    clamp=intervention(spec,state) if action=='clamp' else {}
    if not encoding(spec)[0]:
        out['steps']+=steps
        out['history'].append(dict(action=action,clamps={},mutable=False))
        return out
    runs=[]
    for part in range(2):
        row=sample(spec,3600,seed(random_seed,part),chains=half,initial=xs[part*half:(part+1)*half],
                   clamps=clamp if part else {},temperature=exploratory if part else temperature,
                   read_q=True,max_steps=steps)
        if row['transition_attempts']!=half*steps:raise RuntimeError('Search budget incomplete')
        runs.append(row)
        e=Energy(spec);f,r,_=e.values(np.asarray([row['x']]))
        best=dict(x=row['x'],objective=float(f[0]),violation=float(r.sum()))
        if key(best,spec)<key(out['best'],spec):out['best']=best
    out['x']=runs[0]['final']+runs[1]['final'];out['q']=runs[0]['q']
    out['temperature']=temperature;out['steps']+=steps
    # Extra intervention writes are disclosed separately from heat-bath attempts.
    writes=sum(int(xs[j,i]!=v) for i,v in clamp.items() for j in range(half,len(xs)))
    out['clamp_writes']+=writes
    out['history'].append(dict(action=action,clamps=clamp,explorer_temperature=exploratory,
        feasible_mass=float(np.mean([r['feasible_mass'] for r in runs])),
        change_rate=float(np.mean([r['transition_changes']/r['transition_attempts'] for r in runs])),
        diversity=len(np.unique(out['x'],axis=0))/len(xs),best=out['best']))
    return out


def rule(spec,state):
    last=state['history'][-1] if state['history'] else {}
    if last.get('diversity',1)<.25:
        return 'clamp' if intervention(spec,state) else 'heat'
    return 'cool'


def prompt(spec,state,remaining):
    return ('Control a fixed p-bit model. A: cool both chain halves. '
            'B: cool the protected half and heat the exploratory half for one window. '
            'C: cool both halves and temporarily clamp the most uncertain binary variable '
            'to its minority value in the exploratory half. The other half is always unclamped. '
            'A clamp lasts one window; all best solutions remain archived. '
            'Prefer original feasibility then objective. Return A, B or C.\n'+
            json.dumps(dict(model_summary=dict(variables=len(spec['variables']),
                constraints=len(spec['constraints'])+len(spec['logic']),sense=spec['objective']['sense'],
                objective_scale=objective_scale(spec)),remaining=remaining,temperature=state['temperature'],
                best=state['best'],history=state['history'][-4:],conditional_q=state['q'],
                proposed_clamp=intervention(spec,state)),separators=(',',':'))+'\nAction:')


def outcome(spec,state,action,random_seed,remaining,steps):
    out=state
    for k in range(remaining):
        out=advance(spec,out,action if k==0 else 'cool',seed(random_seed,k),steps)
    return quality(spec,out['best'])


def label(spec,state,random_seed,remaining,cfg):
    def scores(action,phase,n):
        return [outcome(spec,state,action,seed(random_seed,phase,i),remaining,cfg['window_steps']) for i in range(n)]
    screen={a:scores(a,'select',cfg['selection_repeats']) for a in ACTIONS}
    alternative=max(ACTIONS[1:],key=lambda a:np.mean(screen[a]))
    baseline=scores('cool','confirm',cfg['confirmation_repeats'])
    candidate=scores(alternative,'confirm',cfg['confirmation_repeats'])
    gains=np.asarray(candidate)-baseline
    confirmed=bool(np.all(gains>=-1e-8) and np.count_nonzero(gains>1e-8)>=(len(gains)+1)//2)
    return dict(action=alternative if confirmed else 'cool',confirmed_gain=confirmed,
                selection=screen,confirmation=dict(alternative=alternative,baseline=baseline,candidate=candidate))


def episode(spec,cid,cfg,choose):
    began=time.monotonic();state=initial(spec,seed(cid,cfg['seed'],'init'),cfg['chains'])
    state=advance(spec,state,'cool',seed(cid,'warm'),cfg['window_steps'])
    decisions=[];decision_seconds=0
    for i in range(cfg['decisions']):
        tick=time.monotonic();action=choose(spec,state,cfg['decisions']-i);decision_seconds+=time.monotonic()-tick
        decisions.append(action);state=advance(spec,state,action,seed(cid,cfg['seed'],i,'eval'),cfg['window_steps'])
    seconds=time.monotonic()-began
    return dict(x=state['best']['x'],actions=decisions,history=state['history'],
                heat_bath_attempts=state['steps']*cfg['chains'] if encoding(spec)[0] else 0,
                intervention_writes=state['clamp_writes'],episode_seconds=seconds,
                decision_seconds=decision_seconds,search_seconds=seconds-decision_seconds)
