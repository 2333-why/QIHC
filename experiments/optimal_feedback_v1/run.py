"""Run four single-GPU base-model workers and an independent CPU scorer."""
import argparse
import os
import signal
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from qihc.experiment_runtime.storage import load,save,value_digest,worker_lock
from qihc.experiment_runtime.identity import model_identity
from .data import wait_lock
from experiments.feedback_active_v3.runtime import seed_for
from .data import RUN,MODEL,ARMS,DATA,sources
from .policy import shared,decide

def cohort(mode):
    items=load(RUN/"inputs.json")
    if mode=="formal":return items
    dev=sorted((i for i in items if i["stage"]=="development"),key=lambda i:seed_for(i["case"]["id"],"pilot"))
    return dev[:24]

def work(mode,rank):
    from qihc.modeling.backend import LocalModel
    from experiments.feedback_active_v3.local_server import generate
    pending=[r for r in cohort(mode)[rank::4] if any(not (RUN/mode/"models"/str(r["seed"])/a/"predictions"/(r["case"]["id"]+".json")).exists() for a in ARMS)]
    if not pending:return
    local=LocalModel(str(MODEL),device="cuda:0")
    def ask(prompt,seed):
        try:r=generate(local,dict(messages=[dict(content=prompt)],seed=seed,max_tokens=4096,temperature=0))
        except ValueError as e:
            if "Context limit exceeded" in str(e):return ""
            raise
        calls.append(r["usage"])
        return r["choices"][0]["message"]["content"] if r["choices"][0]["finish_reason"]=="stop" else ""
    for item in pending:
        cid=item["case"]["id"];sid=str(item["seed"]);folder=RUN/mode
        def dest(arm):return folder/"models"/sid/arm/"predictions"/(cid+".json")
        if all(dest(a).exists() for a in ARMS):continue
        calls=[];basepath=folder/"shared"/sid/(cid+".json")
        if not basepath.exists():
            base=shared(item,ask);base["usage"]=list(calls);save(basepath,base)
        base=load(basepath)
        for arm in ARMS:
            if dest(arm).exists():continue
            calls.clear();started=time.monotonic()
            row=decide(item,arm,base,ask)
            save(dest(arm),dict(row,case_id=cid,stage=item["stage"],seed=item["seed"],arm=arm,
                                seconds=time.monotonic()-started,usage=list(calls)))
        print(cid,sid,flush=True)

_GOLD=None
def evaluate(path):
    from experiments.feedback_active_v3.cpu import execute
    from qihc.generic.feedback.verification import score
    global _GOLD
    if _GOLD is None:
        _GOLD={c["id"]:c["gold"] for c in load(DATA/"evaluation/cases.gold.json")+load(DATA/"evaluation/cases.public_external.gold.json")}
    row=load(path);cid=row["case_id"];spec=row["model"];gold=_GOLD[cid]
    cache=RUN/"cache";refpath=cache/"references"/path.name
    with wait_lock(refpath):
        if not refpath.exists():save(refpath,execute(dict(stage="reference",gold=gold,seconds=120,seed=seed_for(cid,"reference"))))
        ref=load(refpath)
    results={}
    for sec in ([20,40] if row["arm"] in ("A","B") else [20]):
        p=cache/(value_digest([cid,spec,sec])+".json")
        with wait_lock(p):
            if not p.exists():
                result={}
                for backend in ("scip","pbit"):
                    solved=execute(dict(stage="solve",model=spec,backend=backend,seconds=sec,seed=seed_for(cid,"terminal"),chains=32))
                    result[backend]=score(gold,solved,ref)
                save(p,result)
            results[str(sec)]=load(p)
    sempath=cache/"semantic"/(value_digest([cid,spec])+".json")
    with wait_lock(sempath):
        if not sempath.exists():save(sempath,execute(dict(stage="semantic",model=spec,gold=gold)))
        sem=load(sempath)["status"]
    save(path.parent.parent/"evaluations"/path.name,dict(case_id=cid,seed=row["seed"],arm=row["arm"],stage=row["stage"],
         action=row["action"],semantic=sem,scores=results))

def score_loop(mode):
    folder=RUN/mode
    with ProcessPoolExecutor(max_workers=8) as pool:
        while True:
            todo=[p for p in folder.glob("models/*/*/predictions/*.json") if not (p.parent.parent/"evaluations"/p.name).exists()]
            if todo:list(pool.map(evaluate,todo))
            elif (folder/"generated.json").exists():
                count=len(list(folder.glob("models/*/*/evaluations/*.json")))
                if count==len(cohort(mode))*4:break
                if any(not (p.parent.parent/"evaluations"/p.name).exists() for p in folder.glob("models/*/*/predictions/*.json")):continue
                raise RuntimeError(f"Incomplete evaluations: {count}")
            else:time.sleep(2)
    from .report import report
    report(mode)

def launch(mode):
    import torch
    manifest=load(RUN/"manifest.json")
    if manifest["source"]!=sources() or value_digest(load(RUN/"inputs.json"))!=manifest["cohort"]:raise ValueError("Prepared code/data mismatch")
    attrs=lambda files:[(f["path"],f["bytes"],f["mtime_ns"]) for f in files]
    if attrs(model_identity(MODEL))!=attrs(manifest["model_files"]):raise ValueError("Model changed")
    ids=os.getenv("CUDA_VISIBLE_DEVICES","0,1,2,3").split(",")
    if torch.cuda.device_count()!=4 or len(set(ids))!=4 or len(ids)!=4 or not torch.cuda.is_bf16_supported():raise ValueError("Allocate four BF16 GPUs")
    folder=RUN/mode; logs=folder/"logs";logs.mkdir(parents=True,exist_ok=True)
    cmd=[sys.executable,"-m","experiments.optimal_feedback_v1.run",mode]
    with worker_lock(RUN/"launch.lock"):
        children=[];handles=[]
        try:
            scorer=subprocess.Popen(cmd+["score"]);children.append(scorer)
            workers=[]
            for i,g in enumerate(ids):
                log=(logs/f"worker{i}.log").open("a");handles.append(log)
                p=subprocess.Popen(cmd+[str(i)],env=dict(os.environ,CUDA_VISIBLE_DEVICES=g),stdout=log,stderr=subprocess.STDOUT)
                workers.append(p);children.append(p)
            while any(p.poll() is None for p in workers):
                if any(p.poll() not in (None,0) for p in children):raise RuntimeError(f"Worker failed; inspect {logs}")
                time.sleep(2)
            if any(p.returncode for p in workers):raise RuntimeError(f"Worker failed; inspect {logs}")
            save(folder/"generated.json",{"complete":True})
            if scorer.wait():raise RuntimeError("Scoring failed")
        finally:
            for p in children:
                if p.poll() is None:p.terminate()
            for p in children:p.wait()
            for h in handles:h.close()

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("mode",choices=["pilot","formal"]);p.add_argument("role",nargs="?",default="launch");a=p.parse_args()
    if a.role=="launch":
        def stop(*args):raise KeyboardInterrupt
        signal.signal(signal.SIGTERM,stop)
        launch(a.mode)
    elif a.role=="score":score_loop(a.mode)
    else:work(a.mode,int(a.role))
