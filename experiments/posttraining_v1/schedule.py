"""Schedule independent seed/method jobs on the granted physical GPUs."""
import argparse
import os
import signal
import subprocess
import sys
import time
from qihc.experiment_runtime.identity import model_identity
from qihc.experiment_runtime.storage import worker_lock, load, save
from .data import RUN, MODEL, METHODS, prepare


def launch(mode):
    import torch
    prepared=load(RUN/"prepared.json")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("CUDA BF16 required")
    visible=os.environ.get("CUDA_VISIBLE_DEVICES")
    ids=visible.split(",") if visible else [str(i) for i in range(torch.cuda.device_count())]
    if len(ids)!=4 or torch.cuda.device_count()!=4: raise RuntimeError("Allocate four GPUs")
    stat_files=model_identity(MODEL)
    if (str(MODEL.resolve())!=prepared["model"] or
        [(r["path"],r["bytes"],r["mtime_ns"]) for r in stat_files]!=
        [(r["path"],r["bytes"],r["mtime_ns"]) for r in prepared["files"]]):
        raise RuntimeError("Local model changed since CPU preparation")
    root=RUN/mode
    seeds=[17] if mode=="pilot" else [17,29,43]
    pending=[(method,seed) for seed in seeds for method in METHODS
             if not (root/"models"/str(seed)/method/"done.json").exists()]
    live={}
    evaluator=None
    try:
        evaluator=subprocess.Popen([sys.executable,"-m","experiments.posttraining_v1.evaluate",mode,"--watch"])
        while pending or live:
            if evaluator.poll() is not None and evaluator.returncode!=0:
                raise RuntimeError("Evaluation failed; inspect evaluator output")
            for gpu,(proc,log,job) in list(live.items()):
                if proc.poll() is None: continue
                log.close()
                del live[gpu]
                if proc.returncode: raise RuntimeError(f"{job} failed: see {log.name}")
            for gpu in ids:
                if gpu in live: continue
                job=next(((m,s) for m,s in pending if m=="sft" or (root/"models"/str(s)/"sft/adapter.pt").exists()),None)
                if job is None: continue
                method,seed=job
                path=root/"logs"/f"{seed}-{method}.log"
                path.parent.mkdir(parents=True,exist_ok=True)
                log=path.open("a")
                proc=subprocess.Popen([sys.executable,"-m","experiments.posttraining_v1.learn",method,str(seed),mode],
                                     env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu),stdout=log,stderr=subprocess.STDOUT)
                live[gpu]=proc,log,job
                pending.remove(job)
            time.sleep(2)
        save(root/"trained.json",dict(methods=METHODS,seeds=seeds))
        if evaluator.wait()!=0: raise RuntimeError("Evaluation failed")
    finally:
        for proc,log,_ in live.values():
            if proc.poll() is None: proc.terminate()
            proc.wait()
            log.close()
        if evaluator and evaluator.poll() is None:
            evaluator.terminate()
            evaluator.wait()


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("mode",choices=["pilot","formal"])
    mode=p.parse_args().mode
    def stop(*_):
        raise KeyboardInterrupt("GPU scheduler terminated")
    signal.signal(signal.SIGTERM,stop)
    with worker_lock(RUN/"gpu.lock"):
        split=load(RUN/"split.json")
        from .data import HERE
        from qihc.experiment_runtime.storage import digest
        import hashlib, json
        if split["code"] != {p.name:digest(p) for p in sorted(HERE.glob("*.py"))}:
            raise RuntimeError("Prepared code differs")
        for name,expected in split["files"].items():
            if hashlib.sha256(json.dumps(load(RUN/(name+".json")),sort_keys=True).encode()).hexdigest()!=expected:
                raise RuntimeError("Prepared split changed: "+name)
        launch(mode)
