"""Offline consistency of semantic choices with original-problem test outcomes."""
import numpy as np
from qihc.generic.bounded import violations, objective

def audit(row,gold):
    stats=dict(known_labels=0,wrong_labels=0,unknown_labels=0,
               search_seconds=sum(r['seconds'] for r in row.get('search',[])),
               sampled_states=sum(r.get('pooled_states',0) for r in row.get('search',[])))
    for entry in row.get('evidence',[]):
        test,verdict=entry['test'],entry['verdict']
        if not test:
            continue  # The unprobed review has no assignment-outcome audit.
        if verdict['status']!='chosen':
            stats['unknown_labels']+=1
            continue
        xs=test['assignments']
        if test['kind']=='feasibility':
            truth=bool(violations(gold,xs[0]).sum()==0)
        else:
            if np.any(violations(gold,xs).sum(axis=1)>0):
                # Neither model's feasible-pair interpretation applies to original.
                truth=None
            else:
                values=objective(gold,xs)*(1 if gold['objective']['sense']=='min' else -1)
                truth=int(np.sign(values[0]-values[1]))
        stats['known_labels']+=1
        stats['wrong_labels']+=int(truth is None or test['predictions'][verdict['choice']]!=truth)
    return stats
