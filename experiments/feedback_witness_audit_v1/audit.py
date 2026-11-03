"""Read-only E004-1.4 archive audit; never modifies the frozen runtime."""
import argparse
from collections import Counter,defaultdict
import hashlib
import json
from pathlib import Path
import tarfile
from datetime import datetime,timezone,timedelta
import numpy as np
from qihc.generic.bounded import objective,violations
from experiments.feedback_comparison_v1.report import metrics


def digest(v):return hashlib.sha256(json.dumps(v,ensure_ascii=False,allow_nan=False).encode()).hexdigest()
def key(*v):return hashlib.sha256('\0'.join(v).encode()).hexdigest()


def audit(archive):
    d={};raw={}
    with tarfile.open(archive) as t:
        for m in t:
            if m.isfile() and m.name.endswith('.json'):
                data=t.extractfile(m).read();name=m.name.split('/',1)[1]
                d[name]=json.loads(data);raw[name]=hashlib.sha256(data).hexdigest()
    manifest=d['manifest.json'];cfg=manifest['config'];cases={c['id']:c for c in d['evaluation/cases.CPU_ONLY.json']}
    assert manifest['config_digest']==digest(cfg)
    for name,h in manifest['files'].items():assert raw[name]==h,name
    for path,row in d.items():
        if path.startswith('outputs/'):
            job='jobs/'+path.split('/')[-1];assert row['job_sha256']==raw[job],path
        if path.startswith('responses/'):
            job='requests/'+path.split('/')[-1];assert row['request_sha256']==raw[job],path
    refs={cid:d['evaluation/references/'+key(cid)+'.json'] for cid in cases}
    for cid,c in cases.items():assert refs[cid]['gold_sha256']==digest(c['gold'])
    states={(s['case_id'],s['arm']):s for k,s in d.items() if k.startswith('states/')}
    diffs=[];endpoints={};sem=defaultdict(Counter);witnesses=0
    def score(cid,model,x):
        feasible=x is not None and bool(violations(cases[cid]['gold'],x).sum()==0)
        val=float(objective(cases[cid]['gold'],x)) if x is not None else None
        pred=x is not None and model is not None and bool(violations(model,x).sum()==0)
        return dict(metrics(feasible,val,cases[cid],refs[cid]),objective=val,predicted_feasible=pred)
    fields=('original_feasible','optimal_hit','quality1','quality5','gap_percent','objective','predicted_feasible')
    for row in d['summary.json']['per_case']:
        cid,arm=row['case_id'],row['arm'];st=states[cid,arm]
        calc=score(cid,st['model'],st['x']);endpoints[cid,arm]=calc
        for f in fields:
            if calc[f]!=row['metrics'][f]:diffs.append((cid,arm,f,calc[f],row['metrics'][f]))
        job={'stage':'semantic','config':cfg,'model':st['model'],'gold':cases[cid]['gold']}
        out=d['outputs/'+digest(job)+'.json'];sem[arm][out['status']]+=1
        if out['status']=='disproved':
            w=out.get('constraint_counterexample');pair=out.get('objective_counterexample');proved=False
            if w is not None:
                proved=bool(violations(cases[cid]['gold'],w).sum()==0)!=bool(violations(st['model'],w).sum()==0)
            if pair:
                gf=objective(cases[cid]['gold'],pair)*(1 if cases[cid]['gold']['objective']['sense']=='min' else -1)
                mf=objective(st['model'],pair)*(1 if st['model']['objective']['sense']=='min' else -1)
                proved=proved or bool(np.sign(gf[1]-gf[0])!=np.sign(mf[1]-mf[0]))
            assert proved,(cid,arm,out);witnesses+=1
    corrected=[];cross_diffs=[];status_changes=Counter();backend_counts=defaultdict(Counter);cells=defaultdict(Counter)
    for row in d['cross_matrix.json']:
        cid,arm,backend=row['case_id'],row['arm'],row['backend'];st=states[cid,arm]
        assert row['model_sha256']==digest(st['model'])
        job={'stage':'cross','config':cfg,'model':st['model'],'backend':backend,'seed':0}
        out=d['outputs/'+digest(job)+'.json']
        original=score(cid,st['model'],row['raw_solver'].get('x'))
        for f in fields:
            if original[f]!=row['metrics'][f]:cross_diffs.append((cid,arm,backend,f))
        status_changes[row['status']+' -> '+out['status']]+=1
        m=score(cid,st['model'],out.get('x'))
        corrected.append({'case_id':cid,'arm':arm,'backend':backend,'model_sha256':row['model_sha256'],
                          'original_status':row['status'],'recovered_status':out['status'],'metrics':m,
                          'x':out.get('x'),'wall_seconds':out.get('wall_seconds'),
                          'predicted_objective':out.get('objective'),'optimality_proven':out.get('optimality_proven')})
        for f in ('original_feasible','optimal_hit','quality5','predicted_feasible'):
            backend_counts[backend][f]+=bool(m[f]);cells[arm+'/'+backend][f]+=bool(m[f])
        backend_counts[backend]['model_valid']+=st['model'] is not None
        backend_counts[backend]['rows']+=1
    groups={a:{f:sum(bool(endpoints[cid,a][f]) for cid in cases) for f in ('original_feasible','optimal_hit','quality5')} for a in cfg['arms']}
    contracts={cid:d['contracts/'+key(cid)+'.json'] for cid in cases}
    contracts_summary=Counter(r['status'] for r in contracts.values())
    errors=Counter(r.get('error') for r in contracts.values() if r['status']!='complete')
    cards=Counter();probe_status=Counter();guard_status=Counter();guard_errors=Counter();changed=Counter();cross_models=defaultdict(set)
    for (cid,arm),st in states.items():
        changed[arm]+=st['model']!=d['initial/'+key(cid)+'.json']['model']
        cross_models[cid].add(digest(st['model']))
        for r in st['rounds']:
            probe_status[r['probe']['status']]+=1;guard_status[r['guard']['status']]+=1
            if r['guard'].get('error'):guard_errors[r['guard']['error']]+=1
            cards[arm]+=len(r['probe'].get('cards',[]))
    semantic_changes=Counter()
    for k,meta in d['evaluation/index.json']['semantic'].items():
        snap=d['evaluation/semantic/'+k+'.json'];job={'stage':'semantic','config':cfg,
            'model':d['jobs/'+next(p.split('/')[-1] for p,v in d.items() if p.startswith('jobs/') and v.get('stage')=='semantic' and v.get('gold')==cases[meta['case_id']]['gold'] and digest(v.get('model'))==meta['model_sha256'])]['model'],
            'gold':cases[meta['case_id']]['gold']}
        final=d['outputs/'+digest(job)+'.json'];semantic_changes[snap['status']+' -> '+final['status']]+=1
    # Native optimal results and heuristic results are paired on the identical model row.
    cm={(r['case_id'],r['arm'],r['backend']):r for r in corrected};pair=Counter()
    for cid,arm in states:
        p=cm[cid,arm,'pbit'];x=cm[cid,arm,'scip']
        if not p['metrics']['predicted_feasible'] or not x['metrics']['predicted_feasible']:pair['not_both_model_feasible']+=1;continue
        sign=1 if states[cid,arm]['model']['objective']['sense']=='min' else -1
        delta=sign*(p['predicted_objective']-x['predicted_objective'])
        pair['pbit_better' if delta < -1e-7 else 'pbit_worse' if delta > 1e-7 else 'tie']+=1
    tz=timezone(timedelta(hours=8));clock=d['clock.json'];stop=d['status.json']['time']
    audit={'experiment_id':'E004-1.4','archive_sha256':hashlib.sha256(Path(archive).read_bytes()).hexdigest(),
           'archive_path':str(archive),'code_commit':manifest['code_commit'],'source_hash':manifest['source_hash'],
           'start_shanghai':datetime.fromtimestamp(clock['started_at'],tz).isoformat(),'stop_shanghai':datetime.fromtimestamp(stop,tz).isoformat(),
           'launch_to_stopped_seconds':stop-clock['started_at'],'summary_elapsed_not_job_duration':d['summary.json']['elapsed_since_original_start'],
           'trajectory_score_discrepancies':diffs,'original_cross_score_discrepancies':cross_diffs,
           'groups':groups,'final_semantics_recovered':{a:dict(v) for a,v in sem.items()},'disproved_witnesses_rechecked':witnesses,
           'contracts':dict(contracts_summary),'contract_errors':dict(errors),'probe_status':dict(probe_status),'cards':dict(cards),
           'guard_status':dict(guard_status),'guard_errors':dict(guard_errors),'changed_final_models':dict(changed),
           'cross_status_recovery':dict(status_changes),'semantic_status_recovery':dict(semantic_changes),
           'corrected_backend_counts':{b:dict(v) for b,v in backend_counts.items()},'corrected_arm_backend_cells':{b:dict(v) for b,v in cells.items()},
           'pbit_vs_scip_predicted_model_objective':dict(pair),'unique_final_model_count_including_invalid':sum(map(len,cross_models.values())),
           'response_status':d['summary.json']['response_status_counts'],'costs':d['summary.json']['costs'],
           'historical_baselines':d['summary.json']['historical_baselines']}
    return audit,corrected


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('archive',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    result,matrix=audit(a.archive)
    a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    a.output.with_name(a.output.stem+'_corrected_cross.json').write_text(json.dumps(matrix,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in ('historical_baselines','corrected_arm_backend_cells')},ensure_ascii=False,indent=2))
