"""Synthetic references live only in the evaluator; no family dispatch in feedback."""
import json
import numpy as np
from qihc.generic.bounded import validate

FAMILIES = ('KP', 'MIS', 'MVC', 'MAXCUT', 'MCP', 'MDS')


def make_case(family, n, seed, split, index=0):
    rng = np.random.default_rng(seed)
    variables = [{'name': f'x{i}', 'lower': 0, 'upper': 1} for i in range(n)]
    spec = {'schema': 'bounded-integer-v2', 'variables': variables,
            'objective': {'sense': 'max', 'linear': [1]*n, 'quadratic': []},
            'constraints': [], 'logic': []}
    def constraint(indices, op, rhs):
        a = [0]*n
        for i in indices:
            a[i] = 1
        spec['constraints'].append({'id': 'c'+str(len(spec['constraints'])), 'a': a, 'op': op, 'rhs': rhs})
    lead = f'There are {n} objects indexed 0 through {n-1}. Return x0 through x{n-1}, each 0 or 1. '
    if family == 'KP':
        weights = rng.integers(1, 40, n).tolist()
        values = rng.integers(1, 80, n).tolist()
        capacity = int(sum(weights)*float(rng.uniform(.25, .65)))
        spec['objective']['linear'] = values
        spec['constraints'] = [{'id': 'capacity', 'a': weights, 'op': 'le', 'rhs': capacity}]
        question = lead+'Select objects to maximize total value; their total weight must not exceed capacity. '
        payload = {'weights': weights, 'values': values, 'capacity': capacity}
    else:
        # Sparse conflict rows bound both prompt length and the existing generic IR's
        # 512 constraints. MCP lists nonedges explicitly as forbidden pairs; no
        # silently dropped dense constraints or family-specific solver behavior.
        pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
        count = min(len(pairs), int(n*rng.uniform(.8, 1.8)), 460)
        chosen = rng.choice(len(pairs), count, replace=False)
        edges = sorted(pairs[int(i)] for i in chosen)
        payload = {'pairs': [list(e) for e in edges]}
        if family in ('MIS', 'MCP'):
            for i, j in edges:
                constraint([i, j], 'le', 1)
            meaning = ('Each listed pair is an edge; choose an independent set.' if family == 'MIS' else
                       'Each listed pair is a NONEDGE; all other distinct pairs are edges. Choose a clique.')
            question = lead+meaning+' Maximize the number selected; no listed pair may both be selected. '
        elif family == 'MVC':
            spec['objective']['sense'] = 'min'
            for i, j in edges:
                constraint([i, j], 'ge', 1)
            question = lead+'Each pair is an edge. Select as few vertices as possible, with at least one endpoint of every edge selected. '
        elif family == 'MAXCUT':
            spec['objective']['linear'] = [0]*n
            for i, j in edges:
                spec['objective']['linear'][i] += 1
                spec['objective']['linear'][j] += 1
                spec['objective']['quadratic'].append([i, j, -2])
            question = lead+'Use the bits to partition vertices into two groups. Maximize the number of listed edges whose endpoints belong to different groups. There is no balance requirement. '
        else:
            spec['objective']['sense'] = 'min'
            neighbors = [{i} for i in range(n)]
            for i, j in edges:
                neighbors[i].add(j); neighbors[j].add(i)
            for group in neighbors:
                constraint(sorted(group), 'ge', 1)
            question = lead+'Each pair is an undirected edge. Minimize the number of selected vertices so that every vertex is either selected itself or adjacent to a selected vertex. '
    question += '\nPublic input data:\n```json\n'+json.dumps(payload, separators=(',', ':'))+'\n```'
    validate(spec)
    return {'id': f'e004-{split}-{family}-{n}-{index}-{seed}', 'dataset': family,
            'size': n, 'split': split, 'question': question, 'gold': spec,
            'variable_names': [v['name'] for v in variables], 'generation_seed': seed}


def extension(cfg):
    cases = []
    for f, family in enumerate(FAMILIES):
        for j, n in enumerate(cfg['extension_sizes']):
            for k in range(cfg['extension_per_cell']):
                c = make_case(family, n, cfg['extension_seed']+f*100000+j*1000+k, 'extension', k)
                c['diagnostic_subset'] = k < cfg['diagnostic_per_cell']
                cases.append(c)
    return cases


def development(cfg):
    return [make_case(FAMILIES[k%6], (8, 16, 32, 64)[(k//6)%4], cfg['development_seed']+k,
                      'development_debug' if k < cfg['development_cases']//2 else 'development_acceptance', k)
            for k in range(cfg['development_cases'])]
