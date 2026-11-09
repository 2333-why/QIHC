"""Score using E004 seeds, reference convention and terminal limits."""
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from qihc.experiment_runtime.storage import load,save,worker_lock,value_digest
from qihc.generic.feedback.verification import equivalent,score
from experiments.feedback_active_v3.cpu import execute
from experiments.feedback_active_v3.runtime import seed_for
from experiments.posttraining_v1.evaluate import worker_lock_wait
from .common import DATA,root,ARMS,donor

_GOLD=None
def one(path):
    global _GOLD
    if _GOLD is None:
        _GOLD={r["id"]:r["gold"] for r in load(DATA/"evaluation/cases.gold.json")+load(DATA/"evaluation/cases.public_external.gold.json")}
    p=load(path);cid=p["case_id"];gold=_GOLD[cid];model=p["model"]
    cache=path.parents[5]/"score_cache";ref_path=cache/(cid+".json")
    with worker_lock_wait(ref_path):
        if not ref_path.exists():save(ref_path,execute(dict(stage="reference",gold=gold,seconds=120,seed=seed_for(cid,"reference"))))
        ref=load(ref_path)
    result=cache/(value_digest([cid,model])+".json")
    with worker_lock_wait(result):
        if not result.exists():
            cross={}
            for backend in ("scip","pbit"):
                row=execute(dict(stage="solve",model=model,backend=backend,seconds=20,chains=32,seed=seed_for(cid,"terminal")))
                cross[backend]=dict(row,score=score(gold,row,ref))
            save(result,dict(semantic=equivalent(model,gold),cross=cross))
        metrics=load(result)
    seed=int(path.parts[-4])
    initial=load(donor(seed,"sft")/"predictions"/path.name)["model"]
    initial_semantic=metrics["semantic"] if initial==model else equivalent(initial,gold)
    save(path.parent.parent/"evaluations"/path.name,dict(metrics,stage=p["stage"],case_id=cid,
         initial_status=initial_semantic["status"],decision=p.get("decision")))

def watch(job,mode):
    folder=root(job)/mode
    with worker_lock(folder/"score.lock"),ProcessPoolExecutor(max_workers=8) as pool:
        while True:
            pending=[p for p in folder.glob("models/*/*/predictions/*.json") if not (p.parent.parent/"evaluations"/p.name).exists()]
            if pending:list(pool.map(one,pending))
            elif (folder/"generated.json").exists():
                actual=len(list(folder.glob("models/*/*/evaluations/*.json")))
                goal=len(ARMS[job])*(24 if mode=="pilot" else 2180)
                if actual==goal:break
                if any(not (p.parent.parent/"evaluations"/p.name).exists() for p in folder.glob("models/*/*/predictions/*.json")):continue
                raise RuntimeError(f"Missing evaluations: {actual}/{goal}")
            else:time.sleep(3)
    from .report import report
    report(job,mode)

if __name__=="__main__":watch(*sys.argv[1:])
