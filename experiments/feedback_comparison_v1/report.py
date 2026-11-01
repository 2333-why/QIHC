"""E003/E004 comparison: original-question metrics, shared seeds, no new search."""
import argparse
from collections import defaultdict
from pathlib import Path
import math
import subprocess
import time
import numpy as np
from qihc.experiment_runtime import resilient_storage as s

METRICS=('original_feasible','optimal_hit','quality1','quality5')
ARMS=('C0','C1','C3','U2','B1','B2')


def metrics(feasible, value, case, reference):
    feasible=bool(feasible)
    if feasible and (value is None or not math.isfinite(float(value))):
        raise ValueError('Feasible score lacks a finite original objective')
    ref=reference.get('result',{})
    proven=bool(ref.get('optimality_proven') and ref.get('feasible') and ref.get('objective') is not None)
    gap=None
    if feasible and proven:
        sign=1 if case['gold']['objective']['sense']=='min' else -1
        gap=100*sign*(value-ref['objective'])/max(1.,abs(ref['objective']))
        if gap < -1e-6:
            raise ValueError('Score improves a purported proven optimum; inspect reference/scoring')
    return {'original_feasible':feasible,'optimal_hit':bool(feasible and proven and abs(value-ref['objective'])<1e-7),
            'quality1':bool(gap is not None and gap<=1.+1e-8),
            'quality5':bool(gap is not None and gap<=5.+1e-8),'gap_percent':gap,
            'proven_reference':proven}


def aggregate(rows, cases, seeds):
    total=len(cases)*len(seeds); done=len(rows)
    out={'completed':done,'expected':total,'status':'complete' if done==total else 'partial'}
    for metric in METRICS:
        n=sum(r[metric] for r in rows.values())
        out[metric+'_assigned']=n/total if total else None
        out[metric+'_completed']=n/done if done else None
    # Gap is conditional and must never replace all-assigned success metrics.
    per_case=[]
    for case in cases:
        rs=[rows.get((case['id'],seed)) for seed in seeds]
        if all(r is not None for r in rs):
            gaps=[r['gap_percent'] for r in rs if r['gap_percent'] is not None]
            if gaps:
                per_case.append(float(np.mean(gaps)))
    out.update(feasible_gap_case_mean=float(np.mean(per_case)) if per_case else None,
               feasible_gap_cases=len(per_case),proven_reference_rows=sum(r['proven_reference'] for r in rows.values()))
    return out


def compare(left, right, cases, seeds, replicates=2000):
    paired=[c for c in cases if all((c['id'],seed) in left and (c['id'],seed) in right for seed in seeds)]
    out={'paired_instances':len(paired),'expected_instances':len(cases),
         'complete':len(paired)==len(cases),'metrics':{},'repaired_runs':0,'regressed_runs':0}
    if not paired:
        return out
    blocks=defaultdict(list)
    for c in paired:
        blocks[(c['dataset'],c.get('tier',c.get('size')))].append(c)
        for seed in seeds:
            a,b=left[c['id'],seed],right[c['id'],seed]
            out['repaired_runs']+=int(a['original_feasible'] and not b['original_feasible'])
            out['regressed_runs']+=int(b['original_feasible'] and not a['original_feasible'])
    rng=np.random.default_rng(20261101)
    for metric in METRICS:
        all_d=[];boot=np.zeros(replicates)
        for group in blocks.values():
            ds=np.array([np.mean([float(left[c['id'],se][metric])-float(right[c['id'],se][metric]) for se in seeds]) for c in group])
            all_d.extend(ds)
            boot+=rng.choice(ds,(replicates,len(ds)),replace=True).sum(axis=1)/len(paired)
        ds=np.array(all_d); delta=float(ds.mean())
        null=(rng.choice([-1,1],(replicates,len(ds)))*ds).mean(axis=1)
        out['metrics'][metric]={'delta':delta,'ci95':np.quantile(boot,[.025,.975]).tolist(),
             'p_raw':float((1+(abs(null)>=abs(delta)-1e-12).sum())/(replicates+1))}
    gaps=[]
    for c in paired:
        ds=[left[c['id'],se]['gap_percent']-right[c['id'],se]['gap_percent'] for se in seeds
            if left[c['id'],se]['gap_percent'] is not None and right[c['id'],se]['gap_percent'] is not None]
        if ds:
            gaps.append(float(np.mean(ds)))
    out['paired_feasible_gap_delta_pp']=float(np.mean(gaps)) if gaps else None
    out['paired_feasible_gap_cases']=len(gaps)
    return out


def build(e003,e004,output,replicates=2000):
    # This tool must never write into either running experiment tree.
    for source in (e003,e004):
        if output.resolve()==source.resolve() or source.resolve() in output.resolve().parents:
            raise ValueError('Comparison output must be outside both experiment trees')
    a=s.load(e003/'manifest.json');b=s.load(e004/'manifest.json')
    if s.digest(e003/'manifest.json')!=b['source_manifest_sha256']:
        raise ValueError('E003 source manifest identity mismatch')
    cases=s.load(e003/'cases.CPU_ONLY.json')
    if s.value_digest(cases)!=a['cases_digest']:
        raise ValueError('E003 case identity mismatch')
    public_list=[c for c in s.load(e004/'evaluation/cases.CPU_ONLY.json') if c['split']=='public']
    public={c['id']:c for c in public_list}
    if len(public)!=len(public_list) or len({c['id'] for c in cases})!=len(cases):
        raise ValueError('Duplicate public case IDs')
    if set(public)!={c['id'] for c in cases}:
        raise ValueError('Public case IDs differ; cannot silently intersect instances')
    seeds=sorted(set(a['config']['seeds'])&set(b['config']['seeds']))
    if not seeds or seeds!=sorted(b['config']['seeds']) or len(set(a['config']['seeds']))!=len(a['config']['seeds']):
        raise ValueError('Every E004 seed must have a corresponding E003 seed')
    if 20 not in a['config']['seconds']:
        raise ValueError('Predeclared E003 20-second comparator is missing')
    if 'U2' not in b['config']['public_arms']:
        raise ValueError('Predeclared primary method U2 is missing')
    groups=defaultdict(dict);costs=defaultdict(lambda:defaultdict(float))
    legacy_shared=defaultdict(float)
    snapshots={}
    def load(path):
        row=s.load(path);snapshots[str(path)]=s.digest(path);return row
    for case in cases:
        cid=case['id'];gold_hash=s.value_digest(case['gold'])
        if gold_hash!=s.value_digest(public[cid]['gold']) or case['question']!=public[cid]['question']:
            raise ValueError('Original problem changed: '+cid)
        ref=load(e003/'references'/(s.key(cid)+'.json'))
        if ref['gold_sha256']!=gold_hash:
            raise ValueError('Reference identity mismatch: '+cid)
        ini=load(e004/'initial'/(s.key(cid)+'.json'))
        frozen=e003/'models'/(s.key(cid,'llm_typed_flat')+'.json')
        if ini['source_record_sha256']!=s.digest(frozen):
            raise ValueError('Initial model differs from E003: '+cid)
        for key,value in ini.get('generation_cost',{}).items():
            if isinstance(value,(int,float)):
                legacy_shared[key]+=value
        for track in ('llm_typed_flat','canonical'):
            model=load(e003/'models'/(s.key(cid,track)+'.json'))
            if model.get('model') is not None and s.value_digest(model['model'])!=model.get('model_sha256'):
                raise ValueError('Frozen mathematics checksum mismatch')
            for backend in a['config']['backends']:
                name=f'E003/{track}/{backend}/20s';groups[name]
                for seed in seeds:
                    path=e003/'scores'/s.key(cid,track)/f'{backend}-20-{seed}.json'
                    if not path.exists():continue
                    r=load(path)
                    if (r['case_id'],r['source_arm'],r['backend'],r['budget_seconds'],r['seed'])!=(cid,track,backend,20,seed):
                        raise ValueError('E003 score identity mismatch')
                    if r['response_sha256']!=model['response_sha256'] or r.get('model_sha256')!=model.get('model_sha256'):
                        raise ValueError('E003 score/model mismatch')
                    groups[name][cid,seed]=metrics(r.get('verified_feasible'),r.get('verified_objective'),case,ref)
                    costs[name]['recorded_solver_wall_seconds']+=r.get('raw_metrics',{}).get('wall_seconds',0.) or 0.
        for arm in ARMS:
            if arm not in b['config']['public_arms']:continue
            name='E004/'+arm;groups[name]
            for seed in seeds:
                key=s.key(cid,arm,str(seed));path=e004/'evaluation/scores'/(key+'.json')
                if not path.exists():continue
                r=load(path);job=e004/'evaluation/jobs'/(key+'.json')
                if r.get('job_sha256')!=s.digest(job):
                    raise ValueError('E004 score/job mismatch')
                j=load(job)
                if (j['case']['id'],j['arm'],j['seed'])!=(cid,arm,seed) or s.value_digest(j['case']['gold'])!=gold_hash:
                    raise ValueError('E004 evaluator identity mismatch')
                groups[name][cid,seed]=metrics(r.get('original_feasible'),r.get('objective'),case,ref)
                st=load(e004/'states'/(key+'.json'))
                for field in ('generation_attempts','input_tokens','output_tokens','attributed_seconds','cpu_seconds','cost_incomplete'):
                    costs[name][field]+=st.get(field,0) or 0
    rows=[]
    for name,values in groups.items():
        row={'method':name,**aggregate(values,cases,seeds),'costs_completed_only':dict(costs[name])}
        row['datasets']={d:aggregate({k:v for k,v in values.items() if public[k[0]]['dataset']==d},
                                    [c for c in cases if c['dataset']==d],seeds) for d in sorted({c['dataset'] for c in cases})}
        rows.append(row)
    pairs=[]
    for name in groups:
        if name.startswith('E003/llm_typed_flat/') or name in ('E004/C0','E004/C1','E004/C3','E004/B1','E004/B2'):
            pairs.append({'left':'E004/U2','right':name,
                          'comparison':'historical_extra_feedback_budget' if name.startswith('E003') else 'within_E004_protocol',
                          **compare(groups['E004/U2'],groups[name],cases,seeds,replicates)})
    stats=[metric for pair in pairs for metric in pair['metrics'].values()]
    previous=0.
    for rank,metric in enumerate(sorted(stats,key=lambda r:r['p_raw'])):
        previous=max(previous,min(1.,metric['p_raw']*(len(stats)-rank)));metric['p_holm']=previous
    result={'experiment_id':'E004-1.1','time':time.time(),'seeds':seeds,'instances':len(cases),
            'analysis_script_sha256':s.digest(Path(__file__)),
            'analysis_code_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=Path(__file__).resolve().parents[2],text=True).strip(),
            'status':'complete' if all(r['status']=='complete' for r in rows) else 'partial',
            'rows':rows,'paired':pairs,'shared_E003_initial_generation_cost_once':dict(legacy_shared),
            'source_manifests':{'E003':s.digest(e003/'manifest.json'),'E004':s.digest(e004/'manifest.json')},
            'budget':{'E003_seconds':20,'E004_rounds':b['config']['rounds'],
                      'E004_non_llm_cap_seconds':b['config']['rounds']*sum(b['config'][k] for k in ('base_seconds','probe_seconds','final_seconds'))},
            'scope':'Read-only shared-instance/shared-seed comparison. E003 historical contrasts have unequal total budgets; '
                    'canonical is an oracle-model solver reference, not an end-to-end LLM baseline. '
                    'Partial assigned rates are lower bounds, pending rows are not failures. All terminal failures stay in denominators. '
                    'Optimal/quality hits require a proven original optimum; feasibility is not semantic modeling accuracy. '
                    'Instance means over seeds; task/size stratified bootstrap, sign permutation, Holm over all reported tests; partial tests descriptive only. '
                    'Costs cover completed rows only, exclude separate diagnostics and overhead, and are not total energy/GPU-hours.'}
    output.mkdir(parents=True,exist_ok=True)
    s.save(output/'comparison.json',result);s.save(output/'source_files.json',snapshots)
    fmt=lambda v:'—' if v is None else f'{100*v:.2f}%'
    lines=['# E004-1.1 vs E003-1.1','',f'Status: {result["status"]}; {len(cases)} instances; shared seeds {seeds}.','',
           '| Method | Completed | Original feasible / assigned | Optimal hit / assigned | Gap ≤1% / assigned | Gap ≤5% / assigned |',
           '|---|---:|---:|---:|---:|---:|']
    for r in rows:
        lines.append('| '+r['method']+' | '+f'{r["completed"]}/{r["expected"]}'+' | '+' | '.join(fmt(r[k+'_assigned']) for k in METRICS)+' |')
    lines+=['','Positive paired rate deltas favor U2; negative paired feasible gap deltas favor U2.',
            'Full pairing, confidence intervals, multiplicity adjustment, per-task results and recorded costs: comparison.json.','',result['scope']]
    (output/'comparison.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--e003',type=Path,required=True);p.add_argument('--e004',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    result=build(a.e003,a.e004,a.output)
    print(f'{result["status"]}: {a.output / "comparison.md"}')


if __name__=='__main__':main()
