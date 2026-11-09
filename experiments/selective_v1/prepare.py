"""Validate the shared read-only donor and prepare both independent result roots."""
import hashlib
import json
import sys
from pathlib import Path
from qihc.experiment_runtime.storage import digest,load,save,worker_lock
from qihc.experiment_runtime.identity import model_identity
from experiments.feedback_active_v3.check_local_model import check
from .common import HISTORY,MODEL,DATA,SEEDS,IDS,root,donor,cases

def required_ids(seed):
    return {r["id"] for r in cases(seed,"formal")}

def main():
    from . import training,feedback
    from qihc.generic.solver_baselines.registry import solve
    from experiments.feedback_active_v3.assets import restore
    restore(DATA)
    split=load(HISTORY/"split.json"); prepared=load(HISTORY/"prepared.json")
    if split["experiment"]!="E005-1.7" or split["source"]!=digest(DATA/"public.manifest.json"):
        raise ValueError("Wrong donor split")
    for name,h in split["files"].items():
        if hashlib.sha256(json.dumps(load(HISTORY/(name+".json")),sort_keys=True).encode()).hexdigest()!=h:
            raise ValueError("Changed donor split "+name)
    check(MODEL)
    files=model_identity(MODEL)
    for r in files:
        if r["sha256"] is None:
            with (MODEL/r["path"]).open("rb") as f:r["sha256"]=hashlib.file_digest(f,"sha256").hexdigest()
    if files!=prepared["files"]:raise ValueError("Base model does not match donor")
    truth=load(HISTORY/"train.json")[0]["gold"]
    if solve(truth,"scip",2,17)["x"] is None:raise RuntimeError("SCIP preflight")
    assets={}
    for seed in SEEDS:
        for arm in ("sft","solver_rl","pvcd"):
            folder=donor(seed,arm)
            adapter=folder/"adapter.pt"
            assets[str(adapter)]=digest(adapter)
            for cid in required_ids(seed):
                p=folder/"predictions"/(cid+".json")
                if p.exists():assets[str(p)]=digest(p)
                elif arm=="sft":raise ValueError("Missing shared SFT initial: "+str(p))
    source={p.name:digest(p) for p in Path(__file__).parent.glob("*.py")}
    for job in IDS:
        out=root(job)
        with worker_lock(out/"prepare.lock"):
            manifest=dict(job=job,history=str(HISTORY),split=split["files"],model=prepared["identity"],source=source,
                          donor_files=assets,fingerprints=files)
            p=out/"manifest.json"
            if p.exists() and load(p)!=manifest:raise RuntimeError("Run identity changed: "+str(out))
            save(p,manifest)
    print("BOTH_JOBS_READY",*(str(root(j)) for j in IDS),flush=True)

if __name__=="__main__": main()
