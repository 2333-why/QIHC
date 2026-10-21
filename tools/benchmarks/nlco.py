"""CPU-only benchmark adapter. Gold models never enter modeling prompts.

This is a binary-output, tool-allowed subset, NOT the official NLCO leaderboard.
Task-specific logic is restricted to data evaluation, not the proposed solver.
"""
import hashlib
import json
from pathlib import Path

from qihc.generic.bounded import VERSION, validate, violations, objective

TASKS = ('KP', 'MIS', 'MVC', 'MAXCUT')


def convert(row):
    task = row['task_id']
    a = row['instance_canonical_json']
    surface = row['instance_surface_json']
    if task == 'KP':
        n = a['n_items']
        items = surface['items']
        if [v['weight'] for v in items] != a['weights'] or [v['profit'] for v in items] != a['profits']:
            raise ValueError('Surface/canonical item permutation not supported')
        labels = [v['item_id'] for v in items]
        linear = a['profits']
        constraints = [{'id': 'capacity', 'a': a['weights'], 'op': 'le', 'rhs': a['capacity']}]
        quadratic = []
        witness = row['reference_solution_canonical_json']
        meaning = 'item selection in this identifier order'
    elif task in ('MIS', 'MVC', 'MAXCUT'):
        n = a['num_nodes']
        # Verify the release's edge-wise relabeling is a complete bijection.
        if len(surface['edges']) != len(a['edges']) or surface['num_nodes'] != n:
            raise ValueError('Surface/canonical graph size differs')
        mapping = {}
        for original, displayed in zip(a['edges'], surface['edges']):
            for endpoint in ('u', 'v'):
                k, v = original[endpoint], displayed[endpoint]
                if k in mapping and mapping[k] != v:
                    raise ValueError('Inconsistent graph relabeling')
                mapping[k] = v
        if set(mapping) != set(range(1, n+1)) or len(set(mapping.values())) != n:
            raise ValueError('Incomplete/non-bijective graph mapping')
        labels = [mapping[i] for i in range(1, n+1)]
        edges = [(e['u']-1, e['v']-1) for e in a['edges']]
        if any(type(u) is not int or type(v) is not int or not 0 <= u < n or not 0 <= v < n or u == v for u, v in edges):
            raise ValueError('Unsupported graph labels/self loop')
        if len({tuple(sorted(e)) for e in edges}) != len(edges):
            raise ValueError('Duplicate graph edges')
        linear = [0 if task == 'MAXCUT' else 1]*n
        constraints, quadratic = [], []
        for k, (u, v) in enumerate(edges):
            if task == 'MAXCUT':
                linear[u] += 1; linear[v] += 1
                quadratic.append([min(u,v), max(u,v), -2])
            else:
                coeff = [0]*n; coeff[u] = coeff[v] = 1
                constraints.append({'id': f'edge{k}', 'a': coeff, 'op': 'ge' if task == 'MVC' else 'le', 'rhs': 1})
        ref = row['reference_solution_canonical_json']
        selected = ref[0] if task == 'MAXCUT' else ref
        witness = [int(i in selected) for i in range(1, n+1)]
        meaning = ('partition membership (0/1)' if task == 'MAXCUT' else 'vertex selection')+' in this identifier order'
    else:
        raise ValueError('Task outside audited subset')
    gold = validate({'schema': VERSION, 'variables': [{'name': f'x{i}', 'lower': 0, 'upper': 1} for i in range(n)],
                     'objective': {'sense': 'min' if task == 'MVC' else 'max', 'linear': linear, 'quadratic': quadratic},
                     'constraints': constraints, 'logic': []})
    refobj = float(row['reference_objective_value'])
    if len(witness) != n or violations(gold, witness).sum() != 0 or abs(float(objective(gold, witness))-refobj) > 1e-6:
        raise ValueError('Reference witness inconsistent with independent gold model')
    contract = '\nEvaluation output convention: x0,...,x'+str(n-1)+' denote '+meaning+': '+json.dumps(labels)+'. Ignore the original response-format example; follow the requested output schema. '
    return {'id': 'nlco_'+row['id'], 'dataset': 'nlco_'+task+'_'+row['difficulty_tier'],
            'question': row['prompt']+contract, 'gold': gold, 'reference_answer': refobj,
            'reference_status': 'published_reference_not_claimed_proven_optimal',
            'surface_format': row['surface_format'], 'original_id': row['id']}


def cases_from_repo(repo, per_tier=20):
    if not 1 <= per_tier <= 10000:
        raise ValueError('per_tier must be positive')
    cases, excluded, files = [], [], {}
    for task in TASKS:
        for tier in ('S', 'M', 'L'):
            path = Path(repo)/'step2_contextualization/dataset'/task/f'{task}_{tier}_context.json'
            files[str(path.relative_to(repo))] = hashlib.sha256(path.read_bytes()).hexdigest()
            rows = json.loads(path.read_text(encoding='utf-8'))
            # Select BEFORE validation. Unsupported rows are reported, not replaced by easier cases.
            for row in rows[:per_tier]:
                try:
                    cases.append(convert(row))
                except (ValueError, TypeError, KeyError, IndexError) as exc:
                    excluded.append({'id': row.get('id'), 'reason': str(exc)})
    if not cases or len({c['id'] for c in cases}) != len(cases):
        raise ValueError('Empty or duplicate NLCO selection')
    return cases, {'files': files, 'excluded': excluded, 'included': len(cases),
                   'protocol': 'tool_allowed_binary_output_subset_not_official_leaderboard',
                   'supported_tasks': list(TASKS), 'requested_per_task_tier': per_tier}
