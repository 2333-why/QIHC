"""Explain changed scale hits using historical predictions, scores and cross-solving."""
from collections import Counter, defaultdict
from pathlib import Path
from qihc.experiment_runtime.storage import load, save
from qihc.experiment_runtime.pbit_control import seed
from qihc.generic.solver_baselines.registry import solve
from qihc.generic.feedback.verification import score
from qihc.generic.feedback.verification import equivalent


def examine(cid, old, base_pred, weighted_pred, gold, ref):
    cells = [old['cells'][x+'__schedule'] for x in ('base','weighted')]
    hits = [int(c['score']['optimal_hit']) for c in cells]
    feasible = [int(c['score']['original_feasible']) for c in cells]
    row = dict(id=cid,family=cid.split('_')[1],n=len(gold['variables']),hits=hits,
               feasible=feasible,semantic=[c['semantic'] for c in cells],models=[])
    if hits[0] == hits[1]:
        return row
    for pred,cell in zip((base_pred,weighted_pred),cells):
        spec = pred['model']
        item = dict(invalid=spec is None,error=pred.get('error'))
        if spec is not None:
            solution = solve(spec,'scip',120,seed(cid,'audit') & 0x7fffffff)
            item.update(candidate_proven=solution.get('optimality_proven',False),
                        status=solution['status'],original_score=score(gold,solution,ref),
                        verification=equivalent(spec,gold),
                        constraint_count=len(spec['constraints'])+len(spec['logic']))
        row['models'].append(item)
    return row


def summarize(rows, train_max=20):
    by_family = defaultdict(Counter)
    by_size = defaultdict(Counter)
    types = Counter()
    for r in rows:
        a,b = r['hits']
        size = 'above_train_max' if r['n']>train_max else 'within_train_max'
        for group in (by_family[r['family']],by_size[size]):
            group['cases'] += 1
            group['lost'] += int(a>b)
            group['gained'] += int(a<b)
            group['net'] += b-a
        if a>b:
            m = r['models'][1]
            if m['invalid']:
                kind = 'invalid_weighted'
            elif m['original_score']['optimal_hit']:
                kind = 'weighted_SCIP_recovers_original_optimum'
            elif not m['candidate_proven']:
                kind = 'candidate_optimum_unresolved'
            else:
                kind = 'candidate_optimal_solution_misses_original_optimum'
            types[kind] += 1
    return dict(cases=len(rows),families=dict(by_family),size_groups=dict(by_size),
                train_max=train_max,lost_case_diagnostic=types)
