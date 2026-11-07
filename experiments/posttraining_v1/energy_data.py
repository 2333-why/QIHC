"""Fit classical controls and form paired preferences using training IDs only."""
import os
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from qihc.experiment_runtime.storage import load,save,worker_lock
from qihc.generic.feedback.verification import score
from experiments.feedback_active_v3.cpu import execute
from experiments.feedback_active_v3.runtime import seed_for
from .data import RUN,HERE
from .energy import KEEP,diagnose,walk,weights,rule,best_of


def grade(spec,ref,diag,action,seeds,cid):
    w=weights(spec,diag["base"],action)
    results=[]
    hits=[]
    chain_hits=[]
    for s in seeds:
        final=walk(spec,w,seed_for(cid,s,"finish"),(1.,.5,.25),512)
        best=best_of(diag["best"],final["best"],spec)
        results.append(score(spec,{"x":best["x"]},ref))
        hit=next((r["step"] for r in final["improvements"] if r["violation"]==0 and ref.get("optimality_proven")
                  and ref.get("objective") is not None and abs(r["f"]-ref["objective"])<1e-8),513)
        hits.append(hit)
        chain_hits.append(sum(v is not None and bool(ref.get("optimality_proven")) and ref.get("objective") is not None
                          and abs(v-ref["objective"])<1e-8 for v in final["chain_objectives"])/32)
    return (np.mean([r["optimal_hit"] for r in results]),np.mean([r["original_feasible"] for r in results]),np.mean(chain_hits),-np.mean(hits))


def reference(row):
    return execute(dict(stage="reference",gold=row["gold"],seconds=120,seed=seed_for(row["case"]["id"],"reference")))


def calibration(row):
    cid=row["case"]["id"]; spec=row["gold"]
    path=RUN/"energy/calibration"/(cid+".json")
    if path.exists(): return load(path)
    ref=reference(row)
    scores=[]
    for base in (2.25,3.,4.5,6.):
        diag=diagnose(spec,base,cid,101)
        cached={}
        for delta in (0,1,2,4,8):
            action=rule(diag,delta) if delta else KEEP
            try: weights(spec,base,action)
            except ValueError: action=KEEP
            signature=tuple(map(tuple,action["transfers"]))
            if signature not in cached: cached[signature]=grade(spec,ref,diag,action,[201,202],cid)
            scores.append(dict(base=base,delta=delta,score=cached[signature]))
    save(path,scores)
    return scores


def preference(job):
    row,settings=job
    cid=row["case"]["id"]; spec=row["gold"]
    path=RUN/"energy/preferences"/(cid+".json")
    if path.exists(): return load(path)
    diag=diagnose(spec,settings["base"],cid,101)
    ref=reference(row)
    n=len(spec["constraints"])+len(spec["logic"])
    candidates=[KEEP]
    rng=np.random.default_rng(seed_for(cid,"candidate"))
    raw=[{"profile":k} for k in ("dense_up","dense_down","rhs_up","rhs_down","degree_up","degree_down")]+[rule(diag,settings["delta"])]
    if n>=2:
        for _ in range(64):
            a,b=rng.choice(n,size=2,replace=False).tolist()
            edits=[[a,b,int(rng.choice([1,2,4,8]))]]
            if rng.random()<.5:
                j,k=rng.choice(n,size=2,replace=False).tolist()
                edits.append([j,k,int(rng.choice([1,2,4,8]))])
            raw.append({"transfers":edits})
    for action in raw:
        try: vector=weights(spec,settings["base"],action)
        except ValueError: continue
        if all(not np.array_equal(vector,weights(spec,settings["base"],c)) for c in candidates): candidates.append(action)
        if len(candidates)>=8: break
    scores=[grade(spec,ref,diag,a,range(201,205),cid) for a in candidates]
    top=max(range(len(scores)),key=scores.__getitem__)
    low=min(range(len(scores)),key=scores.__getitem__)
    pair=None
    if scores[top]>scores[low]:
        confirm=[grade(spec,ref,diag,candidates[i],range(301,309),cid) for i in [top,low]]
        if confirm[0]>confirm[1]:
            pair=dict(chosen=candidates[top],rejected=candidates[low],confirmation=confirm)
    result=dict(id=cid,diag=diag,candidates=candidates,scores=scores,pair=pair)
    save(path,result)
    return result


def prepare_energy():
    train=load(RUN/"train.json")
    if not os.getenv("QIHC_REBUILD_PREFS"):
        import hashlib
        bundle=HERE/"energy_labels.json"
        content=bundle.read_bytes().replace(b"\r\n",b"\n")
        if hashlib.sha256(content).hexdigest()!="ddd4ba493e611c6ad5f7baf1052690b5d01a8286aa797a85447249f4cdc6073b":
            raise ValueError("Energy label bundle checksum mismatch")
        frozen=load(bundle)
        ids={r["case"]["id"] for r in train}
        if (frozen["train_sha256"]!=load(RUN/"split.json")["files"]["train"]
                or {r["id"] for r in frozen["preferences"]}!=ids):
            raise ValueError("Frozen labels do not match the current training cohort")
        with worker_lock(RUN/"energy/setup.lock"):
            for r in frozen["preferences"]: save(RUN/"energy/preferences"/(r["id"]+".json"),r)
            for cid,grid in frozen["calibration"].items(): save(RUN/"energy/calibration"/(cid+".json"),grid)
            save(RUN/"energy/settings.json",frozen["settings"])
        print("Loaded frozen preferences:",len(frozen["settings"]["pair_ids"]),"/",len(train))
        return
    with worker_lock(RUN/"energy/setup.lock"),ProcessPoolExecutor(max_workers=int(os.getenv("QIHC_PREP_WORKERS","12"))) as pool:
        grid=list(pool.map(calibration,train))
        means={ (a["base"],a["delta"]): tuple(np.mean([b[i]["score"] for b in grid],axis=0))
                for i,a in enumerate(grid[0])}
        base=max((k[0] for k in means if k[1]==0),key=lambda b:(means[b,0],-abs(b-3)))
        delta=max((1,2,4,8),key=lambda d:means[base,d])
        settings=dict(base=base,delta=delta)
        prefs=list(pool.map(preference,[(r,settings) for r in train]))
        settings["pair_ids"]=[r["id"] for r in prefs if r["pair"]]
        save(RUN/"energy/settings.json",settings)
        print("Matched C/D preferences:",len(settings["pair_ids"]),"/",len(train),flush=True)
        if not settings["pair_ids"]:
            raise RuntimeError("No independently confirmed training preference; C/D cannot be trained on this dataset.")
