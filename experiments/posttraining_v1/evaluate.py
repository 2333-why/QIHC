"""Evaluate new predictions with E004 terminal/reference seeds and scoring."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import time
from qihc.experiment_runtime.storage import load, save, worker_lock
from qihc.generic.feedback.verification import score
from experiments.feedback_active_v3.cpu import execute
from experiments.feedback_active_v3.runtime import seed_for
from .data import DATA,RUN,METHODS

_GOLD=None


def evaluate(path):
    global _GOLD
    if _GOLD is None: _GOLD={r["id"]:r["gold"] for r in load(DATA/"evaluation/cases.gold.json")}
    pred=load(path)
    cid=pred["case_id"]
    gold=_GOLD[cid]
    ref_path=RUN/"references"/(cid+".json")
    with worker_lock_wait(ref_path):
        if not ref_path.exists():
            save(ref_path,execute(dict(stage="reference",gold=gold,seconds=120,seed=seed_for(cid,"reference"))))
        ref=load(ref_path)
    semantic=execute(dict(stage="semantic",model=pred["model"],gold=gold))
    cross={}
    for backend in ("scip","pbit"):
        result=execute(dict(stage="solve",model=pred["model"],backend=backend,seconds=20,chains=32,seed=seed_for(cid,"terminal")))
        cross[backend]=dict(result,score=score(gold,result,ref))
    save(path.parent.parent/"evaluations"/path.name,dict(case_id=cid,semantic=semantic,cross=cross))


from contextlib import contextmanager
@contextmanager
def worker_lock_wait(path):
    lock=worker_lock(path.with_suffix(".lock"))
    while True:
        try:
            lock.__enter__()
            break
        except BlockingIOError:
            time.sleep(.2)
            lock=worker_lock(path.with_suffix(".lock"))
    try: yield
    finally: lock.__exit__(None,None,None)


def run(mode,watch):
    root=RUN/mode
    with worker_lock(root/"evaluation.lock"),ProcessPoolExecutor(max_workers=8) as pool:
        while True:
            pending=[p for p in root.glob("models/*/*/predictions/*.json")
                     if not (p.parent.parent/"evaluations"/p.name).exists()]
            if pending:
                list(pool.map(evaluate,pending))
            elif (root/"trained.json").exists() or not watch:
                if watch:
                    count=len(list(root.glob("models/*/*/evaluations/*.json")))
                    expected=(24 if mode=="pilot" else 480)*(1 if mode=="pilot" else 3)*len(METHODS)
                    if count!=expected:
                        if any(not (p.parent.parent/"evaluations"/p.name).exists()
                               for p in root.glob("models/*/*/predictions/*.json")):
                            continue
                        raise RuntimeError(f"Incomplete terminal evaluations: {count}/{expected}")
                break
            else: time.sleep(4)
    from .report import report
    report(mode)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("mode",choices=["pilot","formal"])
    p.add_argument("--watch",action="store_true")
    a=p.parse_args()
    run(a.mode,a.watch)
