"""Gold targets are training-only; conditional samples select supervised token weights."""
import numpy as np
from qihc.generic.feedback.search import sample
from qihc.experiment_runtime.pbit_control import seed
from experiments.pvcd_cycle_v1.evidence import compatible,cards_from_states
from experiments.posttraining_v1.learn import repair_weights


def collect(model,gold,cid,cfg):
    if not compatible(model,gold):return dict(cards=[],trace=[],reason='incompatible_or_invalid_candidate')
    states=[];trace=[]
    for side,spec in [('candidate',model),('gold',gold)]:
        mutable=[i for i,v in enumerate(spec['variables']) if v['lower']!=v['upper']]
        rng=np.random.default_rng(seed(cid,side,'variables'))
        chosen=rng.permutation(mutable)[:2]
        actions=[{}]+[{int(i):spec['variables'][i][end]} for i in chosen for end in ('lower','upper')]
        for k,clamp in enumerate(actions):
            row=sample(spec,3600,seed(cid,side,k),chains=cfg['chains'],clamps=clamp,
                       temperature=(.25,.5,1.)[k%3],read_q=True,max_steps=cfg['probe_steps'])
            states.extend(row['final'])
            trace.append(dict(side=side,clamps=clamp,q=row['q'],attempts=row['transition_attempts']))
    cards=cards_from_states(model,gold,states,limit=64)
    return dict(cards=cards,trace=trace,reason='sampled')


def token_weights(tok,answer,model,gold,cards):
    if model is None:return [1.]*len(tok.encode(answer,add_special_tokens=False))
    return repair_weights(tok,answer,model,gold,cards)
