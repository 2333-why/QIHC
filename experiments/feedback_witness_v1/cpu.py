"""One bounded job per process. Gold is supplied only to offline evaluation jobs."""
import argparse
import json
import time
from pathlib import Path
import numpy as np
from .common import s
from .protocol import parse_model,guard


def equivalent(model,gold,seconds):
    from qihc.generic.bounded import objective,violations
    if model is None:return {'status':'invalid','compiled':False}
    if model['variables']!=gold['variables']:
        return {'status':'unknown','compiled':True,'reason':'domain_or_mapping_differs'}
    def normalized(m):
        return sorted(json.dumps({k:v for k,v in c.items() if k!='id'},sort_keys=True) for c in m['constraints']+m['logic'])
    def coefficients(m):
        v=m['variables'];n=len(v);a=np.zeros(n+n*n);a[:n]=m['objective']['linear']
        for i,j,c in m['objective']['quadratic']:
            i,j=int(i),int(j)
            if i==j and (v[i]['lower'],v[i]['upper'])==(0,1):a[i]+=c
            else:a[n+min(i,j)*n+max(i,j)]+=c
        return a*(1 if m['objective']['sense']=='min' else -1)
    a,b=coefficients(model),coefficients(gold)
    nz=np.flatnonzero(b)
    scale=float(a[nz[0]]/b[nz[0]]) if len(nz) else 1.
    obj_equal=bool(scale>0 and np.allclose(a,scale*b,rtol=0,atol=1e-9))
    same_constraints=normalized(model)==normalized(gold)
    if obj_equal and same_constraints:
        return {'status':'verified_correct','compiled':True,'proof':'normalized_factor_identity',
                'objective_positive_scale':scale,'constraints_status':'verified_correct','objective_status':'verified_correct'}
    start=time.monotonic();variables=gold['variables'];n=len(variables)
    widths=[v['upper']-v['lower']+1 for v in variables];total=1
    for w in widths:total*=w
    rng=np.random.default_rng(20261104)
    random=np.column_stack([rng.integers(v['lower'],v['upper']+1,256) for v in variables])
    cursor=0;checked=0;constraint_error=None;objective_error=None
    while time.monotonic()-start<seconds:
        if checked==0:xs=random
        else:
            if cursor>=total:break
            ids=np.arange(cursor,min(total,cursor+2048),dtype=np.int64);remaining=ids.copy()
            cols=[]
            for v,w in zip(variables,widths):cols.append(remaining%w+v['lower']);remaining//=w
            xs=np.column_stack(cols);cursor+=len(xs)
        mg=violations(gold,xs).sum(axis=1)==0;mm=violations(model,xs).sum(axis=1)==0
        mismatch=np.flatnonzero(mg!=mm)
        if len(mismatch) and constraint_error is None:constraint_error=xs[int(mismatch[0])].tolist()
        # Only a reversed ordering on two jointly gold-feasible states disproves objective semantics.
        feasible=xs[mg]
        if not obj_equal and len(feasible)>1:
            gf=objective(gold,feasible)*(1 if gold['objective']['sense']=='min' else -1)
            mf=objective(model,feasible)*(1 if model['objective']['sense']=='min' else -1)
            bad=np.flatnonzero(np.sign(np.diff(gf))!=np.sign(np.diff(mf)))
            if len(bad) and objective_error is None:
                i=int(bad[0]);objective_error=[feasible[i].tolist(),feasible[i+1].tolist()]
        checked+=len(xs)
        if constraint_error is not None or objective_error is not None:break
    con_status='disproved' if constraint_error is not None else ('verified_correct' if same_constraints or cursor>=total else 'unknown')
    obj_status='verified_correct' if obj_equal else ('disproved' if objective_error is not None else 'unknown')
    status='disproved' if 'disproved' in (con_status,obj_status) else ('verified_correct' if con_status==obj_status=='verified_correct' else 'unknown')
    return {'status':status,'compiled':True,'constraints_status':con_status,'objective_status':obj_status,
            'objective_positive_scale':scale if obj_equal else None,'states_checked':checked,'enumeration_completed':cursor>=total,
            'constraint_counterexample':constraint_error,'objective_counterexample':objective_error,
            'data_binding_status':'not_separately_certified','seconds':time.monotonic()-start}


def execute(job):
    from qihc.generic.feedback import run_window
    stage=job['stage'];cfg=job['config'];spec=job.get('model');seed=job.get('seed',0)
    if stage=='contract_compile':
        value,model=parse_model(job['response'],job['case'])
        return {'status':'complete','model':model,'quotes':value['quotes']}
    if stage=='guard':return guard(job['case'],spec,job['response'],job.get('contract'),job.get('evidence'))
    if stage in ('base','final'):
        if spec is None:return {'status':'invalid_model','x':None}
        row=run_window(spec,cfg[stage+'_seconds'],seed,'pbit',cfg['chains'])
        row['status']='complete';return row
    if stage=='probe':
        from .evidence import probe
        return probe(spec,job.get('contract'),job['arm'],job.get('base',{}),cfg['probe_seconds'],seed,cfg['chains'],job.get('quotes',[]))
    if stage=='cross':
        if spec is None:return {'status':'not_run_invalid_model','x':None}
        from qihc.generic.solver_baselines.registry import solve
        return solve(spec,job['backend'],cfg['cross_solver_seconds'],seed=seed,chains=cfg['chains'])
    if stage=='semantic':return equivalent(spec,job['gold'],cfg['semantic_check_seconds'])
    raise ValueError('Unknown CPU stage')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('job',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
    started=time.monotonic()
    try:row=execute(s.load(a.job))
    except Exception as exc:row={'status':'execution_error','error':repr(exc),'x':None}
    row.update(job_sha256=s.digest(a.job),worker_seconds=time.monotonic()-started)
    s.save(a.output,row)
