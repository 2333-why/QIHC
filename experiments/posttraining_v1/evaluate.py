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
    if _GOLD is None:
        cases=load(DATA/"evaluation/cases.gold.json")+load(DATA/"evaluation/cases.public_external.gold.json")
        _GOLD={r["id"]:r["gold"] for r in cases}
    pred=load(path)
    cid=pred["case_id"]; spec=pred["model"]
    method=path.parts[-3]; seed=int(path.parts[-4]); root=path.parents[4]
    gold=_GOLD[cid]
    ref_path=RUN/"references"/(cid+".json")
    with worker_lock_wait(ref_path):
        if not ref_path.exists():
            save(ref_path,execute(dict(stage="reference",gold=gold,seconds=120,seed=seed_for(cid,"reference"))))
        ref=load(ref_path)
    settings=load(RUN/"energy/settings.json"); base=settings["base"]
    from .energy import KEEP,ENERGY_METHODS,diagnostic,weights,walk,best_of,rule
    from qihc.generic.feedback.search import sample
    # A/B are CPU-only, generated from the identical per-seed SFT prediction.
    if method=="sft" or (root.name=="gold" and method=="best_lm"):
        diag=diagnostic(root,cid,seed,spec,base) if spec is not None else None
        for arm in ("uniform_weight","rule_weight"):
            action=rule(diag,settings["delta"]) if diag is not None and arm=="rule_weight" else KEEP
            try:
                if spec is not None: weights(spec,base,action)
            except ValueError: action=KEEP
            save(root/"models"/str(seed)/arm/"predictions"/path.name,
                 dict(case_id=cid,stage=pred["stage"],model=spec,action=action))
    semantic=execute(dict(stage="semantic",model=spec,gold=gold))
    cross={}
    action=pred.get("action",KEEP)
    fixed={"x":None}
    if spec is not None:
        w=weights(spec,base if method in ENERGY_METHODS else 3.,action)
        # Baselines without adaptive feedback also spend the probe budget searching.
        if method in ENERGY_METHODS:
            diag=diagnostic(root,cid,seed,spec,base)
        else:
            from .energy import diagnose
            diag=diagnose(spec,3.,cid,seed)
        final=walk(spec,w,seed_for(cid,seed,"finish"),(1.,.5,.25),512)
        best=best_of(diag["best"],final["best"],spec)
        fixed=dict(x=best["x"],seconds=sum(r["seconds"] for r in diag["windows"])+final["seconds"])
    for backend in ("scip","pbit"):
        if backend=="pbit" and method in ENERGY_METHODS and spec is not None:
            probe_time=sum(r["seconds"] for r in diag["windows"])
            budget=max(0.,20.-probe_time)
            incumbent=diag["best"]
            if budget:
                row=sample(spec,budget,seed_for(cid,"terminal"),chains=32,temperature=.5,weights=w,read_q=False)
                from qihc.generic.bounded import violations,objective
                x=row["x"]
                if x is not None:
                    incumbent=best_of(incumbent,dict(x=x,violation=float(violations(spec,x).sum()),f=float(objective(spec,x))),spec)
            result=dict(x=incumbent["x"],probe_seconds=probe_time,terminal_seconds=budget)
        else:
            from qihc.experiment_runtime.storage import value_digest
            signature=value_digest([cid,spec,backend,20,32,seed_for(cid,"terminal")])
            cached=RUN/"terminal_cache"/(signature+".json")
            with worker_lock_wait(cached):
                if not cached.exists():
                    save(cached,execute(dict(stage="solve",model=spec,backend=backend,seconds=20,chains=32,seed=seed_for(cid,"terminal"))))
                result=load(cached)
        cross[backend]=dict(result,score=score(gold,result,ref))
    save(path.parent.parent/"evaluations"/path.name,dict(case_id=cid,stage=pred["stage"],semantic=semantic,
         fixed=dict(fixed,score=score(gold,fixed,ref)),cross=cross))


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
            pending=[p for parent in (root,root/"gold") for p in parent.glob("models/*/*/predictions/*.json")
                     if not (p.parent.parent/"evaluations"/p.name).exists()]
            if pending:
                list(pool.map(evaluate,pending))
            elif (root/"trained.json").exists() or not watch:
                if watch:
                    count=sum(len(list(parent.glob("models/*/*/evaluations/*.json"))) for parent in (root,root/"gold"))
                    expected=(24 if mode=="pilot" else 2180)*(len(METHODS)+4)
                    if count!=expected:
                        if any(not (p.parent.parent/"evaluations"/p.name).exists()
                               for parent in (root,root/"gold") for p in parent.glob("models/*/*/predictions/*.json")):
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
