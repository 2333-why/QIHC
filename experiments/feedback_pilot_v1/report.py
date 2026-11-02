"""Original-question paired screening; never select a winner from partial rows."""
import argparse
from collections import Counter, defaultdict
from pathlib import Path
import time
from qihc.experiment_runtime import resilient_storage as s
from experiments.feedback_comparison_v1.report import metrics


def report(root, pairs=False):
    m = s.load(root/'manifest.json')
    if m['experiment_id'] != 'E004-1.2':
        raise ValueError('Not an E004-1.2 pilot')
    cfg = m['config']
    casepath = root/'evaluation/cases.CPU_ONLY.json'
    baselinepath = root/'evaluation/e003_baselines.json'
    if s.digest(casepath) != m['evaluation_digest'] or s.digest(baselinepath) != m['baselines_digest']:
        raise ValueError('Pilot evaluator/baseline identity changed')
    cases = s.load(casepath)
    groups = defaultdict(dict)
    cost = defaultdict(Counter)
    initial_cost = Counter()
    for c in cases:
        for k, value in s.load(root/'initial'/(s.key(c['id'])+'.json')).get('generation_cost', {}).items():
            if isinstance(value, (int, float)): initial_cost[k] += value
    for b in s.load(baselinepath):
        groups[b['method']][b['case_id']] = b['metrics']
        cost[b['method']]['recorded_solver_wall_seconds'] += b.get('solver_wall_seconds') or 0
    for arm in cfg['public_arms']:
        method = 'E004/'+arm
        groups[method]
        for c in cases:
            key = s.key(c['id'], arm, '0')
            scorepath = root/'evaluation/scores'/(key+'.json')
            if not scorepath.exists(): continue
            row = s.load(scorepath)
            jobpath = root/'evaluation/jobs'/(key+'.json')
            job = s.load(jobpath)
            if row['job_sha256'] != s.digest(jobpath) or (job['case']['id'], job['arm'], job['seed']) != (c['id'], arm, 0):
                raise ValueError('Pilot score/job identity mismatch')
            if s.value_digest(job['case']['gold']) != s.value_digest(c['gold']):
                raise ValueError('Pilot scoring mathematics mismatch')
            ref = s.load(root/'evaluation/references'/(s.key(c['id'])+'.json'))
            if ref['gold_sha256'] != s.value_digest(c['gold']):
                raise ValueError('Pilot reference identity mismatch')
            groups[method][c['id']] = metrics(row.get('original_feasible'),row.get('objective'),c,ref)
            st = s.load(root/'states'/(key+'.json'))
            for field in ('generation_attempts','input_tokens','output_tokens','attributed_seconds','cpu_seconds','valid_patch_outputs','cost_incomplete'):
                cost[method][field] += st.get(field,0) or 0
            cost[method]['execution_errors'] += len(st.get('errors',[]))
            cost[method]['trajectories_with_errors'] += bool(st.get('errors'))
    fields = ('original_feasible','optimal_hit','quality1','quality5')
    rows = [{'method':name, 'completed':len(values),'expected':len(cases),
             'counts':{field:sum(v[field] for v in values.values()) for field in fields},
             'cost_completed_only':dict(cost[name])} for name, values in sorted(groups.items())]
    comparisons = []
    left = groups['E004/U2']
    for name, right in sorted(groups.items()):
        if name == 'E004/U2': continue
        ids = sorted(left.keys() & right.keys())
        counts = {}
        for field in fields:
            counts[field] = {'wins':sum(left[k][field] > right[k][field] for k in ids),
                             'ties':sum(left[k][field] == right[k][field] for k in ids),
                             'losses':sum(left[k][field] < right[k][field] for k in ids)}
        comparisons.append({'left':'E004/U2','right':name,'paired_instances':len(ids),
                            'complete':len(ids)==len(cases),'binary_metrics':counts,
                            'scope':'oracle-model reference' if '/canonical/' in name else
                                    'unequal total budget historical contrast' if name.startswith('E003/') else 'same pilot budget'})
    done = sum(len(groups['E004/'+arm]) for arm in cfg['public_arms'])
    expected = len(cases)*len(cfg['public_arms'])
    stages, errors = Counter(), Counter()
    for path in (root/'states').glob('*.json'):
        st = s.load(path)
        stages[f"round{st['round']}/{st['stage']}"] += 1
        errors.update(str(e.get('error')) for e in st.get('errors',[]))
    result = {'experiment_id':'E004-1.2','status':'complete' if done==expected else 'partial',
              'completed':done,'expected':expected,'instances':len(cases),'seed':0,'time':time.time(),
              'rows':rows,'paired':comparisons,'per_case':dict(groups),'state_stages':dict(stages),
              'error_events':dict(errors),'shared_initial_generation_cost_once':dict(initial_cost),
              'scope':'18-case retrospective exploratory screen. Pending rows are not failures. '
                      'E003 uses 20s without revision; pilot uses 40s plus up to four LLM calls. '
                      'Canonical uses the correct model; it is not an end-to-end baseline. '
                      'One instance per stratum and one seed cannot establish stable superiority; no degenerate singleton bootstrap CI. '
                      'Costs cover completed records, not total GPU-hours or energy.'}
    s.save(root/'summary.json',result)
    lines=['# E004-1.2 feedback pilot','',f'{result["status"]}: {done}/{expected} trajectories; {len(cases)} instances; seed 0.',
           '', '| Method | Completed | Feasible count | Optimal hit count | Gap ≤5% count |',
           '|---|---:|---:|---:|---:|']
    for r in rows:
        lines.append(f'| {r["method"]} | {r["completed"]}/{r["expected"]} | {r["counts"]["original_feasible"]} | {r["counts"]["optimal_hit"]} | {r["counts"]["quality5"]} |')
    lines += ['',result['scope'],'','Paired wins/ties/losses, per-case scores, costs and errors: summary.json.']
    (root/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return result


if __name__ == '__main__':
    from .adapter import default_root
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,default=default_root());a=p.parse_args()
    r=report(a.run);print(f'{r["status"]}: {r["completed"]}/{r["expected"]}; {a.run / "report.md"}')
