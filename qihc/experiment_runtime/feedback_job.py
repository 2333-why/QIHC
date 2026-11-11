"""Reusable four-GPU feedback job; historical entry points remain immutable."""
import importlib
import argparse
import os
import signal
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor,ThreadPoolExecutor
from threading import Lock,local as thread_local
from pathlib import Path
from qihc.experiment_runtime.storage import load,save,value_digest,worker_lock
from qihc.experiment_runtime.identity import model_identity
from experiments.feedback_active_v3.runtime import seed_for
def configure(package):
    global PACKAGE, RUN, MODEL, ARMS, DATA, sources, wait_lock, shared, decide, AUDIT
    PACKAGE = package
    data = importlib.import_module(package+'.data')
    policy = importlib.import_module(package+'.policy')
    RUN, MODEL, ARMS, DATA = data.RUN, data.MODEL, data.ARMS, data.DATA
    sources, wait_lock = data.sources, data.wait_lock
    shared, decide = policy.shared, policy.decide
    module = getattr(data, 'AUDIT_MODULE', None)
    AUDIT = importlib.import_module(module).audit if module else None


def initialize(package):
    configure(package)


def cohort(mode):
    items=load(RUN/"inputs.json")
    if mode=="formal":return items
    dev=sorted((i for i in items if i["stage"]=="development"),key=lambda i:seed_for(i["case"]["id"],"pilot"))
    return dev[:24]

def work(mode,rank):
    from qihc.modeling.backend import LocalModel
    from experiments.feedback_active_v3.local_server import generate
    pending=[r for r in cohort(mode) if any(not (RUN/mode/"models"/str(r["seed"])/a/"predictions"/(r["case"]["id"]+".json")).exists() for a in ARMS)]
    if not pending:return
    local=LocalModel(str(MODEL),device="cuda:0")
    inference_lock=Lock();ctx=thread_local();active_seconds=0.;started_worker=time.monotonic()
    def ask(prompt,seed):
        nonlocal active_seconds
        try:
            with inference_lock:
                started=time.monotonic()
                r=generate(local,dict(messages=[dict(content=prompt)],seed=seed,max_tokens=4096,temperature=0))
                active_seconds+=time.monotonic()-started
        except ValueError as e:
            if "Context limit exceeded" in str(e):
                ctx.calls.append(dict(error="context_limit", detail=str(e)))
                return ""
            raise
        ctx.calls.append(dict(usage=r["usage"], finish_reason=r["choices"][0]["finish_reason"], text=r["choices"][0]["message"]["content"]))
        return r["choices"][0]["message"]["content"] if r["choices"][0]["finish_reason"]=="stop" else ""
    def process(item):
        cid=item["case"]["id"];sid=str(item["seed"]);folder=RUN/mode
        def dest(arm):return folder/"models"/sid/arm/"predictions"/(cid+".json")
        if all(dest(a).exists() for a in ARMS):return
        calls=[];ctx.calls=calls;cache={};basepath=folder/"shared"/sid/(cid+".json")
        if not basepath.exists():
            base=shared(item,ask);base["usage"]=list(calls);save(basepath,base)
        base=load(basepath)
        for arm in ARMS:
            if dest(arm).exists():continue
            calls.clear();started=time.monotonic()
            row=decide(item,arm,base,ask,cache=cache)
            save(dest(arm),dict(row,case_id=cid,stage=item["stage"],seed=item["seed"],arm=arm,
                                seconds=time.monotonic()-started,usage=list(calls)))
        print(cid,sid,flush=True)

    # Four rotated queues with nonblocking per-item claims. A faster GPU can
    # consume unfinished items initially nearest another worker.
    offset=len(pending)*rank//4
    def claim(item):
        lock=worker_lock(RUN/mode/"claims"/str(item["seed"])/(item["case"]["id"]+".lock"))
        try:lock.__enter__()
        except BlockingIOError:return
        try:process(item)
        finally:lock.__exit__(None,None,None)
    with ThreadPoolExecutor(max_workers=int(os.getenv("QIHC_PROBE_LANES","2"))) as lanes:
        for _ in lanes.map(claim,pending[offset:]+pending[:offset]):pass
    save(RUN/mode/"timing"/f"worker{rank}.json",dict(inference_service_seconds=active_seconds,
         elapsed_after_loading=time.monotonic()-started_worker,lanes=int(os.getenv("QIHC_PROBE_LANES","2"))))

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
    calls = list(row.get('usage', []))
    if row['arm'] not in ('A', 'B'):
        calls += load(path.parents[4]/'shared'/str(row['seed'])/path.name).get('usage', [])
    usage = {name: sum(c.get('usage', {}).get(name, 0) for c in calls)
             for name in ('prompt_tokens', 'completion_tokens')}
    usage['calls'] = len(calls)
    save(path.parent.parent/"evaluations"/path.name,dict(case_id=cid,seed=row["seed"],arm=row["arm"],stage=row["stage"],
         action=row["action"],semantic=sem,scores=results,
         audit=AUDIT(row,gold) if AUDIT else {}, usage=usage))

def score_loop(mode):
    folder=RUN/mode
    with ProcessPoolExecutor(max_workers=int(os.getenv("QIHC_SCORE_WORKERS","8")), initializer=initialize, initargs=(PACKAGE,)) as pool:
        while True:
            todo=[p for p in folder.glob("models/*/*/predictions/*.json") if not (p.parent.parent/"evaluations"/p.name).exists()]
            if todo:list(pool.map(evaluate,todo))
            elif (folder/"generated.json").exists():
                count=len(list(folder.glob("models/*/*/evaluations/*.json")))
                if count==len(cohort(mode))*len(ARMS):break
                if any(not (p.parent.parent/"evaluations"/p.name).exists() for p in folder.glob("models/*/*/predictions/*.json")):continue
                raise RuntimeError(f"Incomplete evaluations: {count}")
            else:time.sleep(2)
    importlib.import_module(PACKAGE+".report").report(mode)

def launch(mode):
    import torch
    manifest=load(RUN/"manifest.json")
    if manifest["source"]!=sources() or value_digest(load(RUN/"inputs.json"))!=manifest["cohort"]:raise ValueError("Prepared code/data mismatch")
    attrs=lambda files:[(f["path"],f["bytes"],f["mtime_ns"]) for f in files]
    if attrs(model_identity(MODEL))!=attrs(manifest["model_files"]):raise ValueError("Model changed")
    ids=os.getenv("CUDA_VISIBLE_DEVICES","0,1,2,3").split(",")
    if torch.cuda.device_count()!=4 or len(set(ids))!=4 or len(ids)!=4 or not torch.cuda.is_bf16_supported():raise ValueError("Allocate four BF16 GPUs")
    folder=RUN/mode; logs=folder/"logs";logs.mkdir(parents=True,exist_ok=True)
    cmd=[sys.executable,"-m",PACKAGE+".run",mode]
    with worker_lock(RUN/"launch.lock"):
        children=[];handles=[]
        try:
            next_sample=0.
            profiler=True
            scorer=subprocess.Popen(cmd+["score"]);children.append(scorer)
            workers=[]
            for i,g in enumerate(ids):
                log=(logs/f"worker{i}.log").open("a");handles.append(log)
                p=subprocess.Popen(cmd+[str(i)],env=dict(os.environ,CUDA_VISIBLE_DEVICES=g),stdout=log,stderr=subprocess.STDOUT)
                workers.append(p);children.append(p)
            while any(p.poll() is None for p in workers):
                if any(p.poll() not in (None,0) for p in children):raise RuntimeError(f"Worker failed; inspect {logs}")
                if profiler and time.monotonic()>=next_sample:
                    next_sample=time.monotonic()+30
                    try:
                        sample=subprocess.run(["nvidia-smi","-i",",".join(ids),"--query-gpu=timestamp,uuid,utilization.gpu,memory.used",
                            "--format=csv,noheader,nounits"],capture_output=True,text=True,timeout=4)
                        if sample.returncode:profiler=False
                        else:
                            with (logs/"gpu.csv").open("a") as f:f.write(sample.stdout)
                    except (OSError,subprocess.TimeoutExpired):profiler=False
                time.sleep(2)
            if any(p.returncode for p in workers):raise RuntimeError(f"Worker failed; inspect {logs}")
            save(folder/"generated.json",{"complete":True})
            if scorer.wait():raise RuntimeError("Scoring failed")
        finally:
            for p in children:
                if p.poll() is None:p.terminate()
            for p in children:p.wait()
            for h in handles:h.close()

def main(package):
    configure(package)
    p=argparse.ArgumentParser();p.add_argument("mode",choices=["pilot","formal"]);p.add_argument("role",nargs="?",default="launch");a=p.parse_args()
    if a.role=="launch":
        def stop(*args):raise KeyboardInterrupt
        signal.signal(signal.SIGTERM,stop)
        launch(a.mode)
    elif a.role=="score":score_loop(a.mode)
    else:work(a.mode,int(a.role))
