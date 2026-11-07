"""Validated energy transfers and fixed-update p-bit trajectory measurements."""
import json
import time
import numpy as np
from qihc.generic.factor_search import encoding
from qihc.generic.feedback.search import Energy, conditional_probability
from qihc.modeling.active_protocol_v2 import compact
from experiments.feedback_active_v3.runtime import seed_for

KEEP = {"transfers":[]}
ENERGY_METHODS = ("uniform_weight","rule_weight","best_lm","trajectory_lm")


def weights(spec, base, action):
    n = len(spec["constraints"])+len(spec["logic"])
    out = np.full(n,base,dtype=float)
    if isinstance(action,dict) and set(action)=={"profile"}:
        profile=action["profile"]
        if profile not in ("dense_up","dense_down","rhs_up","rhs_down","degree_up","degree_down"):
            raise ValueError("Unknown profile")
        ranks=[abs(c["rhs"]) if profile.startswith("rhs") else sum(v!=0 for v in c["a"]) for c in spec["constraints"]]
        if profile.startswith("degree") and spec["constraints"]:
            incidence=np.asarray([c["a"] for c in spec["constraints"]])!=0
            ranks=(incidence @ incidence.sum(axis=0)).tolist()
        ranks.extend([0]*len(spec["logic"]))
        order=sorted(range(n),key=lambda i:(ranks[i],i))
        width=n//2
        if width and ranks[order[0]]!=ranks[order[-1]]:
            change=min(base-2.125,8-base)*(1 if profile.endswith("up") else -1)
            out[order[:width]]-=change
            out[order[-width:]]+=change
        return out
    if not isinstance(action,dict) or set(action)!={"transfers"}:
        raise ValueError("Expected transfers")
    edits=action["transfers"]
    if not isinstance(edits,list) or len(edits)>2:
        raise ValueError("At most two transfers")
    for edit in edits:
        if not isinstance(edit,list) or len(edit)!=3: raise ValueError("Invalid edit")
        source,dest,units=edit
        if any(type(v)!=int for v in edit) or source==dest or not (0<=source<n and 0<=dest<n) or units not in (1,2,4,8):
            raise ValueError("Invalid row or transfer amount")
        out[source]-=units/8
        out[dest]+=units/8
    if np.any(out<2.125) or np.any(out>8): raise ValueError("Unsafe weight")
    return out


def best_of(a,b,spec):
    def rank(item):
        return (item["violation"]>0,item["violation"],item["f"]*(1 if spec["objective"]["sense"]=="min" else -1))
    return min((x for x in (a,b) if x is not None),key=rank)


def walk(spec,w,seed,temp,steps):
    start=time.monotonic()
    rng=np.random.default_rng(seed)
    energy=Energy(spec,weights=w)
    sites,lower=encoding(spec)
    bits=rng.integers(2,size=(32,len(sites)))
    x=np.tile(lower,(32,1))
    for k,(j,v) in enumerate(sites): x[:,j]+=bits[:,k]*v
    f,r,e=energy.values(x)
    best=None
    chain_best=np.full(32,np.inf)
    improvements=[]
    rates=[]; masses=[]; signatures=[]; values=[]
    for t in range(steps+1):
        violated=r.sum(axis=1)
        chain_best=np.minimum(chain_best,np.where(violated==0,energy.sign*f,np.inf))
        i=np.lexsort((energy.sign*f,violated,violated>0))[0]
        improved=best_of(best,dict(x=x[i].tolist(),violation=float(violated[i]),f=float(f[i])),spec)
        if improved is not best:
            best=improved
            improvements.append(dict(step=t,**best))
        if t and t%(steps//4)==0:
            rates.append((r>0).mean(axis=0)); masses.append(float((violated==0).mean()))
            signatures.append(len(np.unique(x,axis=0))/32)
            good=f[violated==0]
            values.append(float(np.median(good)) if good.size else None)
        if t==steps: break
        if not sites: continue
        k=int(rng.integers(len(sites)))
        j,v=sites[k]
        trial=x.copy(); trial[:,j]+=(1-2*bits[:,k])*v
        nf,nr,ne=energy.values(trial)
        is_zero=bits[:,k]==0
        temperature=temp if np.isscalar(temp) else temp[min(len(temp)-1,t*len(temp)//steps)]
        q=conditional_probability(np.where(is_zero,e,ne),np.where(is_zero,ne,e),temperature)
        change=(rng.random(32)<q)!=bits[:,k]
        bits[change,k]=1-bits[change,k]
        x[change]=trial[change]; f[change]=nf[change]; r[change]=nr[change]; e[change]=ne[change]
    return dict(best=best,improvements=improvements,chain_objectives=[float(v*energy.sign) if np.isfinite(v) else None for v in chain_best],rate=np.mean(rates,axis=0).round(4).tolist(),mass=masses,
                diversity=signatures, feasible_medians=values,seconds=time.monotonic()-start)


def diagnose(spec,base,cid,seed):
    w=weights(spec,base,KEEP)
    windows=[walk(spec,w,seed_for(cid,seed,i,"probe"),t,128) for i,t in enumerate((.25,.5,1.))]
    best=None
    for row in windows: best=best_of(best,row["best"],spec)
    return dict(best=best,windows=windows,base=base)


def rule(diag,delta):
    rates=np.mean([row["rate"] for row in diag["windows"]],axis=0)
    if len(rates)<2 or max(rates)==min(rates): return KEEP
    return {"transfers":[[int(np.argmin(rates)),int(np.argmax(rates)),delta]]}


def prompt(case,spec,diag,method):
    context=dict(question=case["question"],model=compact(spec),base=diag["base"],best=diag["best"])
    if method=="trajectory_lm":
        context["windows"]=[dict(temperature=t,**{k:r[k] for k in ("rate","mass","diversity","feasible_medians")})
                            for t,r in zip((.25,.5,1.),diag["windows"])]
    return ('Select p-bit penalty transfers. JSON only: {"transfers":[[from,to,eighths],...]}. '
            'Rows numbered from zero: constraints followed by logic. Use zero, one or two transfers, '
            'units 1,2,4,8. Or return {"profile":"dense_up"}, dense_down, rhs_up/down or degree_up/down to transfer mass between halves ranked by row size, absolute RHS or summed variable incidence. Keep with {"transfers":[]}.\n'
            +json.dumps(context,ensure_ascii=False,separators=(",",":")))


def diagnostic(root,cid,seed,model,base):
    # All energy policies receive the same frozen SFT input and probe samples.
    path=root/"diagnostics"/str(seed)/(cid+".json")
    from .evaluate import worker_lock_wait
    from qihc.experiment_runtime.storage import save,load
    with worker_lock_wait(path):
        if not path.exists():
            save(path,diagnose(model,base,cid,seed))
    return load(path)
