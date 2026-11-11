"""Shared same-model terminal cache, kept strictly inside this new iteration."""
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from qihc.experiment_runtime.storage import load,save,worker_lock,value_digest
from qihc.generic.feedback.verification import equivalent,score
from experiments.feedback_active_v3.cpu import execute
from experiments.feedback_active_v3.runtime import seed_for
from experiments.posttraining_v1.evaluate import worker_lock_wait
from .common import CFG,RUN,DATA,cases

_GOLD=None
_META={}
def one(path):
    global _GOLD,_META
    path=Path(path)
    if _GOLD is None:
        rows=load(DATA/'evaluation/cases.gold.json')+load(DATA/'evaluation/cases.public_external.gold.json')
        _GOLD={r['id']:r['gold'] for r in rows}
        _META={r['id']:r.get('family','public') for r in rows}
    p=load(path); cid=p['case_id']; gold=_GOLD[cid]; model=p['model']
    cache=RUN/'score_cache'; ref_path=cache/'references'/(cid+'.json')
    with worker_lock_wait(ref_path):
        if not ref_path.exists(): save(ref_path,execute(dict(stage='reference',gold=gold,seconds=120,seed=seed_for(cid,'reference'))))
        ref=load(ref_path)
    result=cache/'models'/(value_digest([cid,model,CFG['solver_seconds']])+'.json')
    with worker_lock_wait(result):
        if not result.exists():
            cross={}
            for backend in ('scip','pbit'):
                row=execute(dict(stage='solve',model=model,backend=backend,seconds=CFG['solver_seconds'],chains=CFG['chains'],seed=seed_for(cid,'terminal')))
                if row.get('status')=='execution_error': raise RuntimeError(str(row))
                cross[backend]=dict(row,score=score(gold,row,ref))
            save(result,dict(semantic=equivalent(model,gold),cross=cross))
        metrics=load(result)
    save(path.parent.parent/'evaluations'/path.name,dict(metrics,stage=p['stage'],case_id=cid,
         family=_META.get(cid,'unspecified'),n_variables=len(gold['variables'])))

def watch(mode):
    folder=RUN/mode
    seeds=[17] if mode=='pilot' else CFG['seeds']
    expected=sum(len(cases(s,mode)) for s in seeds)*len(CFG['arms'])
    with worker_lock(folder/'score.lock'),ProcessPoolExecutor(max_workers=CFG['cpu_workers']) as pool:
        while True:
            pending=[p for p in folder.glob('models/*/*/predictions/*.json') if not (p.parent.parent/'evaluations'/p.name).exists()]
            if pending: list(pool.map(one,pending))
            elif (folder/'generated.json').exists():
                count=len(list(folder.glob('models/*/*/evaluations/*.json')))
                if count!=expected: raise RuntimeError(f'Missing evaluations {count}/{expected}')
                break
            else: time.sleep(3)
    from .report import report
    report(mode)

if __name__=='__main__': watch(sys.argv[1])
