import argparse
import torch
from qihc.experiment_runtime.storage import load,save,worker_lock
from experiments.feedback_active_v3.runtime import seed_for
from experiments.posttraining_v1.engine import Engine
from experiments.posttraining_v1.learn import parsed
from qihc.modeling.active_protocol_v2 import model_prompt
from .common import root,donor,HISTORY,MODEL,cases,generated
from .feedback import decide
from .training import fit

def run(job,arm,seed,mode):
    out=root(job)/mode/"models"/str(seed)/arm
    with worker_lock(out/"lock"):
        torch.manual_seed(seed)
        ref=torch.load(donor(seed,"sft")/"adapter.pt",map_location="cpu",weights_only=True)
        engine=None
        if arm in ("pair_rl","pair_local"):
            engine=Engine(MODEL)
            rows=load(HISTORY/"train.json")
            if mode=="pilot":rows=sorted(rows,key=lambda r:seed_for(r["case"]["id"],"pilot"))[:12]
            fit(engine,arm,seed,mode,out,rows,ref)
        frozen=load(root(job)/"manifest.json")["donor_files"]
        cfg=load(__import__("pathlib").Path(__file__).resolve().parents[1]/"feedback_active_v4/config.json")
        cfg["probe_seconds"]=20
        for case in cases(seed,mode):
            cid=case["id"]; path=out/"predictions"/(cid+".json")
            if path.exists():continue
            folder=donor(seed,arm if job=="train" else "sft")
            stored=folder/"predictions"/path.name
            if job=="feedback":
                draft=load(stored)
                if engine is None and arm!="sft":
                    engine=Engine(MODEL);engine.restore(ref)
                state=decide(engine,arm,case,draft,root(job)/mode/"shared"/str(seed),seed,cfg)
            elif arm in ("sft","solver_rl","pvcd") and str(stored) in frozen:
                state=dict(load(stored),origin="frozen_donor")
            else:
                if engine is None:
                    engine=Engine(MODEL)
                    engine.restore(torch.load(folder/"adapter.pt",map_location="cpu",weights_only=True))
                text=generated(engine,model_prompt(case),seed_for(cid,0,"initial"))
                model,error=parsed(text,case)
                state=dict(model=model,raw=text,error=error,origin="new_generation")
            save(path,dict(state,case_id=cid,stage=case["stage"]))
        save(out/"done.json",dict(count=len(cases(seed,mode))))
if __name__=="__main__":
    import sys
    job,arm,seed,mode=sys.argv[1:];run(job,arm,int(seed),mode)
