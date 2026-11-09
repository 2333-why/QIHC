"""One independent worker per assigned CUDA device, without shared network ports."""
import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from qihc.experiment_runtime.storage import load,save,digest,worker_lock
from qihc.experiment_runtime.identity import model_identity
from .common import root,ARMS,SEEDS,HISTORY,MODEL

def launch(job,mode):
    import torch
    out=root(job); manifest=load(out/"manifest.json")
    if manifest["job"]!=job or manifest["history"]!=str(HISTORY):raise ValueError("Incorrect preparation manifest")
    if manifest["source"]!={p.name:digest(p) for p in Path(__file__).parent.glob("*.py")}:raise ValueError("Code changed after CPU preparation")
    attrs=lambda files:[(r["path"],r["bytes"],r["mtime_ns"]) for r in files]
    if attrs(model_identity(MODEL))!=attrs(manifest["fingerprints"]):raise ValueError("Base model changed")
    import hashlib,json
    for name,h in manifest["split"].items():
        if hashlib.sha256(json.dumps(load(HISTORY/(name+".json")),sort_keys=True).encode()).hexdigest()!=h:raise ValueError(name)
    if any(digest(Path(k))!=h for k,h in manifest["donor_files"].items() if job=="train" or "/sft/" in k):raise ValueError("Frozen donor changed")
    if torch.cuda.device_count()!=4 or not torch.cuda.is_bf16_supported():raise RuntimeError("Allocate four BF16 GPUs")
    ids=os.getenv("CUDA_VISIBLE_DEVICES","0,1,2,3").split(",")
    if len(ids)!=4 or len(set(ids))!=4:raise ValueError("Ambiguous GPU assignment")
    folder=out/mode
    tasks=[(a,s) for s in ([17] if mode=="pilot" else SEEDS) for a in ARMS[job]
           if not (folder/"models"/str(s)/a/"done.json").exists()]
    live={}
    with worker_lock(out/"launcher.lock"):
        watcher=subprocess.Popen([sys.executable,"-m","experiments.selective_v1.score",job,mode])
        try:
            while tasks or live:
                if watcher.poll() not in (None,0):raise RuntimeError("CPU evaluation failed")
                for gpu,(proc,log) in list(live.items()):
                    if proc.poll() is None:continue
                    log.close();del live[gpu]
                    if proc.returncode:raise RuntimeError(f"Worker failed: {log.name}")
                for gpu in ids:
                    if gpu in live or not tasks:continue
                    a,s=tasks.pop(0);path=folder/"logs"/f"{s}-{a}.log";path.parent.mkdir(parents=True,exist_ok=True)
                    log=path.open("a")
                    proc=subprocess.Popen([sys.executable,"-m","experiments.selective_v1.worker",job,a,str(s),mode],
                                env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu),stdout=log,stderr=subprocess.STDOUT)
                    live[gpu]=proc,log
                time.sleep(2)
            save(folder/"generated.json",dict(arms=ARMS[job]))
            if watcher.wait():raise RuntimeError("Scoring failed")
        finally:
            for proc,log in live.values():
                if proc.poll() is None:proc.terminate()
                proc.wait();log.close()
            if watcher.poll() is None:watcher.terminate();watcher.wait()

if __name__=="__main__":
    def stop(*args):raise KeyboardInterrupt
    signal.signal(signal.SIGTERM,stop)
    p=argparse.ArgumentParser();p.add_argument("job",choices=ARMS);p.add_argument("mode",choices=["pilot","formal"])
    args=p.parse_args();launch(args.job,args.mode)
