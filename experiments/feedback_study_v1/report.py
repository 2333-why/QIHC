"""Independent aggregate/paired reports; failures remain in full denominators."""
import argparse
from collections import defaultdict
from pathlib import Path
import time
import numpy as np
from .common import s, default_root, configurations, identity

PAIRS = [('U2','C1'),('C3','C2'),('M1','C3'),('M2','C3'),('M3','C3'),('M4','C3'),
         ('U1','C3'),('U2','U1'),('U2','B1'),('U2','B2')]
_CACHE = {}


def holm(values):
    order = sorted(range(len(values)),key=lambda i:values[i]); result = [1.]*len(values); previous = 0.
    for rank,i in enumerate(order):
        previous = max(previous,min(1.,values[i]*(len(values)-rank))); result[i]=previous
    return result


def paired(rows, split, replicates=2000):
    grouped = defaultdict(dict); strata = {}
    for row in rows:
        if row['split'] != split:
            continue
        grouped[row['case_id']].setdefault(row['arm'],[]).append(row)
        strata[row['case_id']] = (row['dataset'], row.get('size'))
    results = []; rng = np.random.default_rng(20261101)
    for left,right in PAIRS:
        for metric in ('original_feasible','quality5_reference' if split=='extension' else 'quality_5_percent'):
            blocks = defaultdict(list)
            for case_id, arms in grouped.items():
                if left in arms and right in arms:
                    l,r = arms[left],arms[right]
                    if {v['seed'] for v in l}!={v['seed'] for v in r}:
                        continue
                    blocks[strata[case_id]].append(np.mean([v[metric] for v in l])-np.mean([v[metric] for v in r]))
            if not blocks:
                continue
            delta = np.concatenate([np.array(v) for v in blocks.values()]); observed=float(delta.mean())
            boot = np.zeros(replicates)
            for values in blocks.values():
                a=np.array(values); boot+=rng.choice(a,(replicates,len(a)),replace=True).sum(axis=1)/len(delta)
            null = (rng.choice([-1,1],(replicates,len(delta)))*delta).mean(axis=1)
            p=(1+int((abs(null)>=abs(observed)-1e-12).sum()))/(replicates+1)
            results.append({'left':left,'right':right,'metric':metric,'paired_instances':len(delta),
                'delta':observed,'ci95':np.quantile(boot,[.025,.975]).tolist(),'p_raw':p})
    for row,p in zip(results,holm([r['p_raw'] for r in results])):
        row['p_holm']=p
    return results


def report(root, pairs=False):
    m=s.load(root/'manifest.json'); cfg=m['config']; cases=s.load(root/'cases.public.json')
    expected=defaultdict(int); metadata={}
    for case in cases:
        for arm,seed in configurations(case,cfg):
            key=identity(case['id'],arm,seed)
            metadata[key]=(case,arm,seed); expected[(case['split'],arm)]+=1
    cache=_CACHE.setdefault(str(root.resolve()),{})
    for path in (root/'evaluation'/'scores').glob('*.json'):
        if path.stem in cache:
            continue
        c,arm,seed=metadata[path.stem]; row=s.load(path)
        st=s.load(root/'states'/path.name)
        row.update(case_id=c['id'],arm=arm,seed=seed,split=c['split'],dataset=c['dataset'],size=c.get('size'))
        row.setdefault('original_feasible',False); row.setdefault('reference_proven',False)
        row['quality_5_percent']=bool(row['original_feasible'] and row['reference_proven'] and
                                     row.get('gap_percent') is not None and row['gap_percent']<=5.+1e-8)
        row['quality5_reference']=bool(row['original_feasible'] and row.get('gap_percent') is not None and row['gap_percent']<=5.+1e-8)
        for field in ('input_tokens','output_tokens','generation_attempts','attributed_seconds','cpu_seconds','searched_windows','cost_incomplete'):
            row[field]=st.get(field,0)
        row['execution_errors']=len(st.get('errors',[]))
        cache[path.stem]=row
    grouped=defaultdict(list)
    for r in cache.values():
        grouped[(r['split'],r['arm'])].append(r)
    summary=[]
    for (split,arm),total in sorted(expected.items()):
        rows=grouped[(split,arm)]; completed=len(rows)
        sumfield=lambda field:sum(float(r.get(field) or 0) for r in rows)
        summary.append({'split':split,'arm':arm,'completed':completed,'expected':total,
            'feasible_all_assigned':sumfield('original_feasible')/total,
            'feasible_completed':sumfield('original_feasible')/completed if completed else None,
            'quality5_all_assigned':sumfield('quality_5_percent')/total,
            'quality5_reference_all_assigned':sumfield('quality5_reference')/total,
            'optimal_all_assigned':sumfield('optimal_hit')/total,
            'proven_reference_fraction_completed':sumfield('reference_proven')/completed if completed else None,
            'calls':sumfield('generation_attempts'),'input_tokens':sumfield('input_tokens'),
            'output_tokens':sumfield('output_tokens'),'attributed_generation_seconds':sumfield('attributed_seconds'),
            'cpu_task_seconds':sumfield('cpu_seconds'),'rows_with_unknown_cost':sumfield('cost_incomplete'),
            'initial_feasible_count':sumfield('initial_candidate_feasible'),
            'repaired_count':sumfield('repaired_feasibility'),'regressed_count':sumfield('regressed_feasibility'),
            'repair_rate_from_initial_failure':sumfield('repaired_feasibility')/max(1,completed-sumfield('initial_candidate_feasible')),
            'regression_rate_from_initial_success':sumfield('regressed_feasibility')/max(1,sumfield('initial_candidate_feasible')),
            'execution_errors':sumfield('execution_errors')})
    total=sum(expected.values()); done=len(cache)
    obj={'experiment_id':'E004-1.1','status':'complete' if done==total else 'partial','completed':done,'expected':total,
         'rows':summary,'time':time.time(),'scope':'Pending rows are not measured failures; assigned-denominator rates are lower bounds until complete. '
         'Formal and development splits separate; quality5 requires proven references, quality5_reference uses the recorded best-known feasible reference (not necessarily optimal). '
         'Cost excludes shared initial generation and separate evaluation.'}
    initial_cost=defaultdict(float)
    for p in (root/'initial').glob('*.json'):
        for k,v in s.load(p).get('generation_cost',{}).items():
            if isinstance(v,(int,float)):
                initial_cost[k]+=v
    obj['shared_initial_generation_cost']=dict(initial_cost)
    s.save(root/'summary.json',obj)
    lines=['# E004-1.1 feedback study','',f'Status: {obj["status"]}; {done}/{total} trajectories (including development).','',
           '| Split | Arm | Completed | Original feasible / assigned | Proven optimum gap ≤5% / assigned | Reference gap ≤5% / assigned | Calls | Output tokens |',
           '|---|---|---:|---:|---:|---:|---:|---:|']
    for r in summary:
        lines.append(f'| {r["split"]} | {r["arm"]} | {r["completed"]}/{r["expected"]} | {r["feasible_all_assigned"]:.2%} | {r["quality5_all_assigned"]:.2%} | {r["quality5_reference_all_assigned"]:.2%} | {r["calls"]:.0f} | {r["output_tokens"]:.0f} |')
    lines+=['',obj['scope'],'','Feasibility is evaluated against original mathematics, not the generated model. '
            'No hardware-energy or universal solver-superiority claims. Partial paired statistics are descriptive only.']
    (root/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    if pairs:
        for split in ('public','extension'):
            values=paired(list(cache.values()),split)
            s.save(root/f'paired_{split}.json',{'comparisons':values,'complete':all(r['completed']==r['expected'] for r in summary if r['split']==split),
                'unit':'instance; seeds averaged, task/size-stratified bootstrap; paired sign permutation; Holm across reported primary comparisons'})
    return obj


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--run',type=Path,default=default_root()); p.add_argument('--pairs',action='store_true')
    a=p.parse_args(); print(report(a.run,a.pairs))
