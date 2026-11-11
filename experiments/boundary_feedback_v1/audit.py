"""Offline diagnostics only. Never imported by the online feedback policy."""
import numpy as np
from qihc.generic.bounded import objective, violations

def audit(row, gold):
    result = dict(known_labels=0, wrong_labels=0, judged_windows=0, unique_states=0,
                  search_seconds=row.get('probe', {}).get('seconds', 0.), unknown_labels=0,
                  sampled_states=row.get('probe', {}).get('pooled_states', 0))
    for window in row.get('evidence', []):
        result['search_seconds'] += window['probe']['seconds']
        result['sampled_states'] += window['probe'].get('pooled_states', 0)
        result['unique_states'] += len(window['assignments'])
        verdict = window['verdict']
        if verdict['status'] != 'judged':
            continue
        result['judged_windows'] += 1
        xs = window['assignments']
        feasible = violations(gold, xs).sum(axis=1) == 0
        costs = objective(gold, xs)*(1 if gold['objective']['sense'] == 'min' else -1)
        labels = verdict['labels']
        for i, label in enumerate(labels['feasible']):
            if label is None:
                result['unknown_labels'] += 1
            else:
                result['known_labels'] += 1
                result['wrong_labels'] += int(bool(feasible[i]) != label)
        for (i, j), label in zip(window['pairs'], labels['orders']):
            if label is None:
                result['unknown_labels'] += 1
            else:
                result['known_labels'] += 1
                result['wrong_labels'] += int(not (feasible[i] and feasible[j]) or int(np.sign(costs[i]-costs[j])) != label)
    return result
