import hashlib
import os
from pathlib import Path
from qihc.experiment_runtime.storage import load,save,key,value_digest,digest,worker_lock
from qihc.experiment_runtime.identity import model_identity
from experiments.feedback_active_v3.check_local_model import check
from experiments.feedback_active_v3.runtime import public_cases
from experiments.feedback_active_v3.data import public_view
from experiments.feedback_active_v3.assets import restore

ROOT=Path(__file__).resolve().parents[2]
BASE=Path("/inspire/hdd/global_user/czxs25210022/why")
DATA=ROOT/"experiments/feedback_active_v3/assets/public_dataset"
HISTORY=Path(os.getenv("QIHC_E004_RUN",BASE/"results/E004_2.2_active_feedback"))
RUN=Path(os.getenv("QIHC_OPT_RUN",BASE/"results/E004_2.4_base_optimal_feedback"))
MODEL=Path(os.getenv("QIHC_MODEL",BASE/"models/Qwen3-32B"))
SEEDS={"development":[17],"main":[17,29,43],"scale":[17,29],"public":[17,29]}
ARMS=("A","B","C","D")
def sources():
    return {p.name:digest(p) for p in Path(__file__).parent.glob("*.py")}

def freeze(files):
    cases=public_cases(DATA)
    items=[]
    for c in cases:
        stage=c["split"]
        for seed in SEEDS[stage]:
            cid=c["id"]
            identity=key("public",cid,"C0",str(seed)) if stage=="public" else cid
            initial=load(HISTORY/"initial"/(key(identity)+".json"))
            if initial["case_id"]!=cid:raise ValueError(cid)
            raw=""
            if initial["model"] is None:
                request=HISTORY/"requests"/(initial["initial_request"]+".json")
                if request.exists():raw=load(request).get("text") or ""
            items.append(dict(case=public_view(c),stage=stage,seed=seed,model=initial["model"],
                              checks=initial["checks"],error=initial["error"],raw=raw))
    if len(cases)!=910 or len(items)!=2180:raise ValueError("Wrong cohort")
    manifest=dict(experiment="E004_2.4",cohort=value_digest(items),source=sources(),model_files=files,
                  dataset=digest(DATA/"public.manifest.json"),arms=list(ARMS))
    for name,obj in [("manifest",manifest),("inputs",items)]:
        p=RUN/(name+".json")
        if p.exists() and load(p)!=obj:raise ValueError("Frozen run changed: "+str(p))
        save(p,obj)

def prepare():
    if RUN.resolve()==HISTORY.resolve():raise ValueError("History and output must differ")
    restore(DATA)
    history=load(HISTORY/"manifest.json")
    if history["config"]["experiment_id"]!="E004_2.2" or history.get("smoke") or history["data_manifest"]!=load(DATA/"manifest.json"):
        raise ValueError("Incompatible historical experiment")
    check(MODEL)
    files=model_identity(MODEL)
    for r in files:
        if r["sha256"] is None:
            with (MODEL/r["path"]).open("rb") as f:r["sha256"]=hashlib.file_digest(f,"sha256").hexdigest()
    if value_digest(files)!=history["model_identity"]:raise ValueError("Base model identity mismatch")
    from experiments.feedback_active_v3.cpu import execute
    g=load(DATA/"evaluation/cases.gold.json")[0]["gold"]
    if execute(dict(stage="solve",model=g,backend="scip",seconds=2,seed=17,chains=32)).get("x") is None:
        raise RuntimeError("SCIP check failed")
    with worker_lock(RUN/"prepare.lock"):freeze(files)
    print("E004_2.4_CPU_READY",RUN)

if __name__=="__main__":prepare()


from contextlib import contextmanager
@contextmanager
def wait_lock(path):
    import time
    while True:
        cm=worker_lock(path.with_suffix(".lock"))
        try:
            cm.__enter__()
            break
        except BlockingIOError:
            time.sleep(.1)
    try:yield
    finally:cm.__exit__(None,None,None)
