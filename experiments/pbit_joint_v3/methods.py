from collections import Counter, defaultdict
import numpy as np
from experiments.pbit_joint_v1 import modeling, run as common, search
from qihc.modeling.active_protocol_v2 import model_prompt


def confirmed_pairs(sample):
    """Pair each intervention with cool only when repeated trajectory gains agree."""
    returns = np.asarray(sample['returns']).mean(axis=-1)
    pairs = []
    for action in (1, 2):
        delta = returns[:,action] - returns[:,0]
        if (delta >= -1e-8).all() and (delta > 1e-8).sum() >= 2:
            pairs.append((action,0))
        elif (delta <= 1e-8).all() and (delta < -1e-8).sum() >= 2:
            pairs.append((0,action))
    return pairs


def control_data(evidence, arm):
    rows = []
    for cid,record in evidence.items():
        for s in record['samples']:
            if arm == 'balanced':
                rows.append((cid,s['prompt'],search.ACTIONS.index(s['labels']['trajectory']),-1))
            else:
                rows.extend((cid,s['prompt'],win,lose) for win,lose in confirmed_pairs(s))
    # Equal total loss per observed winning action, equal mass per case within action.
    case_counts = Counter((win,cid) for cid,_,win,_ in rows)
    cases_per_action = Counter(win for win,cid in case_counts)
    weights = [1/(case_counts[win,cid]*cases_per_action[win]) for cid,_,win,_ in rows]
    return rows, np.asarray(weights)/np.mean(weights) if weights else []


def balanced_replay(samples, budget, seed):
    """Equal exposure to observed winners. Sample a case before sampling its state."""
    groups=defaultdict(lambda:defaultdict(list))
    for i,(cid,_,win,_) in enumerate(samples):
        groups[win][cid].append(i)
    rng=np.random.default_rng(seed)
    order=[]
    while len(order)<budget:
        for win in rng.permutation(sorted(groups)):
            buckets=groups[win]
            cid=rng.choice(sorted(buckets))
            order.append(int(rng.choice(buckets[cid])))
    return order[:budget]


def train(engine, arm, folder, manifest, ready, rows, evidence, cfg):
    import torch
    from experiments.posttraining_v1.data import target
    parent = arm in ('slow','anchor')
    if parent:
        reference = torch.load(manifest['adapter'],map_location='cpu',weights_only=True)
        engine.restore(reference)
        def backward(i):
            row = rows[i]
            cid = row['case']['id']
            e = evidence[cid]
            text = target(row['gold'])
            weights = modeling.token_weights(engine.tok,text,e['candidate'],row['gold'],e['cards'])
            ids = engine.encode(text)+[engine.tok.eos_token_id]
            loss = engine.backward(model_prompt(row['case']),ids,'nll',weights=weights+[1.],
                                   reference=reference if arm=='anchor' else None)
            return dict(case_id=cid, loss=loss, weighted_tokens=sum(w>1 for w in weights))
        count = len(rows)
        lr = cfg['model_lr']*(.5 if arm=='slow' else 1)
    else:
        samples,weights = control_data(evidence,arm)
        tokens = common.action_ids(engine)
        def backward(i):
            cid,prompt,win,lose = samples[i]
            scores = common.logits(engine,prompt,tokens)
            if lose < 0:
                loss = -scores.log_softmax(0)[win]
            else:
                loss = torch.nn.functional.softplus(scores[lose]-scores[win])
            if arm=='balanced':
                loss = loss*float(weights[i])
            loss.backward()
            return dict(case_id=cid,loss=float(loss.detach()),winner=win,loser=lose)
        count = len(samples)
        lr = cfg['control_lr']
    order=(balanced_replay(samples,count*cfg['epochs'],cfg['seed']) if arm=='rank'
           else common.order_for(count,cfg['epochs']))
    common.optimize(engine,folder,order,lr,backward,
                    dict(arm=arm,ready=ready,config=cfg))
