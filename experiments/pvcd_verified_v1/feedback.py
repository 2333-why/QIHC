"""Training-only witness explanation; no full gold model or coefficient diff in prompt."""
import json
import numpy as np
from qihc.generic.bounded import violations,objective

def feasible(m,x):return bool(violations(m,np.asarray(x)).sum()==0)

def explain(model,gold,cards):
    result=[]
    for card in cards:
        c=dict(card)
        if c['kind']=='feasibility':
            x=list(c['x']); desired=bool(c['correct_feasible'])
            # Greedy simplification preserves the certified disagreement direction.
            for i,v in enumerate(gold['variables']):
                y=list(x);y[i]=v['lower']
                if feasible(gold,y)==desired and feasible(model,y)!=desired:x=y
            c.update(x=x,correct_feasible=feasible(gold,x),candidate_feasible=feasible(model,x))
            c['meaning']='This assignment MUST be feasible; your model excludes it.' if desired else 'This assignment MUST be infeasible; your model admits it. A reward/penalty term alone is not a hard constraint.'
        else:
            c['meaning']={-1:'x must be preferred to y',0:'x and y must tie',1:'y must be preferred to x'}[c['gold_order']]
            c['order_convention']='Orders compare minimization-normalized objectives: sign(sense_sign*(f(x)-f(y))), sense_sign=+1 for min and -1 for max.'
            c['raw_objective_values']={name:{v:float(objective(spec,np.asarray(c[v]))) for v in ('x','y')}
                                       for name,spec in (('reference',gold),('candidate',model))}
            c['reference_sense']=gold['objective']['sense'];c['candidate_sense']=model['objective']['sense']
        if c not in result:result.append(c)
    return result

def prompt(base,attempt,model,cards):
    mapping=[dict(index=i,name=v['name']) for i,v in enumerate(model['variables'])]
    return (base+'\nTRAINING REPAIR: inspect the certified disagreements below and re-derive the affected '
        'model components from the problem statement. Check every required relation, not just the shown '
        'assignments. Preserve already correct parts. Do not restrict the feasible set merely to fix an '
        'objective-order discrepancy. Respect ties. Counterexamples do not uniquely determine the model. '
        'Use the stated identifier mapping in the question. Sparse coefficient keys MUST be the zero-based '
        'indices below, never original entity labels. Return ONE complete valid JSON model; no arithmetic '
        'expressions, comments or ellipses. Close dictionaries with } and arrays with ].\nINDEX MAP: '+
        json.dumps(mapping)+'\nATTEMPT: '+attempt+'\nCERTIFIED FEEDBACK: '+json.dumps(cards,ensure_ascii=False))

def remaining(model,gold,cards):
    if model is None:return None
    count=0
    for c in cards:
        if c['kind']=='feasibility':count+=feasible(model,c['x'])!=feasible(gold,c['x'])
        else:
            if not all(feasible(model,c[v]) and feasible(gold,c[v]) for v in ('x','y')):
                count+=1;continue
            def order(spec):
                delta=float(objective(spec,np.asarray(c['x']))-objective(spec,np.asarray(c['y'])))
                return int(np.sign(delta*(1 if spec['objective']['sense']=='min' else -1)))
            count+=order(model)!=order(gold)
    return int(count)
