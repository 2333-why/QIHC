"""Only source-supported objective edits can enter an already executable draft."""
import copy
import json
import time
import itertools
import numpy as np
from qihc.generic.bounded import objective,violations
from qihc.generic.feedback.search import sample
from qihc.generic.solver_baselines.registry import solve
from qihc.modeling.active_protocol_v2 import compact,model_prompt,parse_model,parse_checks,source_prompt
from qihc.modeling.protocol import load_object
from experiments.feedback_active_v3.runtime import seed_for

def objective_check(checks):
    candidates=[r for r in checks.get("checks",[]) if r["kind"]=="objective"]
    if not candidates:return None
    fields=("sense","linear","quadratic")
    if any(any(c[k]!=candidates[0][k] for k in fields) for c in candidates):return None
    return candidates[0]

def shared(item,ask):
    case=item["case"]; cid=case["id"]; seed=item["seed"]; model=item["model"]
    if model is None:
        prompt=model_prompt(case)+"\nRepair the invalid initial attempt. Parser: "+str(item["error"])[:1000]+"\nAttempt: "+item["raw"][:12000]
        text=ask(prompt,seed_for(cid,seed,"repair"))
        try:m=parse_model(text,case)
        except (ValueError,TypeError,KeyError,IndexError):m=None
        return dict(model=m,action="compile_repair",text=text)
    c=objective_check(item["checks"])
    if c is None:
        text=ask(source_prompt(case)+"\nFocus only on the complete objective. Return one objective check with a verbatim quote or an empty checks list.",seed_for(cid,seed,"objective_check"))
        try:c=objective_check(parse_checks(text,case))
        except (ValueError,TypeError,KeyError,IndexError):c=None
    return dict(model=model,action="unchanged",check=c)

def values(model,xs):
    return objective(model,np.asarray(xs))*(1 if model["objective"]["sense"]=="min" else -1)

def pool(model,source,arm,seed,seconds,holdout):
    rng=np.random.default_rng(seed)
    end=time.monotonic()+seconds
    states=[]
    def influence(spec):
        obj=spec["objective"]; v=np.array(obj["linear"],dtype=float)
        for j,k,c in obj["quadratic"]:
            v[j]+=c
            if j!=k:v[k]+=c
        return v*(1 if obj["sense"]=="min" else -1)
    focus=int(np.argmax(abs(influence(model)-influence(source))))
    for i in range(4):
        left=end-time.monotonic()
        if left<=.01:break
        spec=copy.deepcopy(model if i%2==0 else source)
        if holdout:
            # Different search directions on the same unchanged feasible region.
            spec["objective"]=dict(sense="min",linear=rng.integers(-4,5,len(model["variables"])).tolist(),quadratic=[])
        elif i>=2:
            spec["objective"]["sense"]="min" if spec["objective"]["sense"]=="max" else "max"
        if arm=="C":
            x=solve(spec,"scip",left/(4-i),seed+i).get("x")
            if x is not None:states.append(x)
        else:
            clamp=None
            if i>=2 and not holdout:
                clamp={focus:model["variables"][focus]["lower" if i==2 else "upper"]}
            states+=sample(spec,left/(4-i),seed+i,chains=32,temperature=(.25,.5,1.,.5)[i],clamps=clamp,read_q=False)["final"]
    if not states:return []
    xs=np.unique(states,axis=0)
    return xs[violations(model,xs).sum(axis=1)==0].tolist()

def comparisons(xs,model,source):
    if len(xs)<2:return []
    a,b=values(model,xs),values(source,xs)
    return [(xs[i],xs[j],int(np.sign(a[i]-a[j])),int(np.sign(b[i]-b[j])))
            for i,j in itertools.combinations(range(len(xs)),2)]

def amend(case,model,fields,text):
    edit=load_object(text)
    if not isinstance(edit,dict) or set(edit)!={"field","value"} or edit["field"] not in fields:
        raise ValueError("Expected one objective field")
    m=compact(model);m["objective"][edit["field"]]=edit["value"]
    return parse_model(json.dumps({"model":m}),case)

def decide(item,arm,base,ask,seconds=20):
    model=item["model"]
    if arm=="A":return dict(model=model,action="base")
    if model is None:return base
    if arm=="B":return dict(model=model,action="compile_only")
    c=base.get("check")
    if not c:return dict(model=model,action="no_objective_check")
    obj={k:c[k] for k in ("sense","linear","quadratic")}
    fields=[k for k in obj if model["objective"][k]!=obj[k]]
    if not fields:return dict(model=model,action="objective_agrees")
    source=dict(model,objective=obj);cid=item["case"]["id"];seed=item["seed"]
    shown=next((v for v in comparisons(pool(model,source,arm,seed_for(cid,seed,"find"),seconds/2,False),model,source)
                if v[2]!=v[3]),None)
    if shown is None:return dict(model=model,action="no_conflict")
    validation=[v for v in comparisons(pool(model,source,arm,seed_for(cid,seed,"validate"),seconds/2,True),model,source)
                if v[:2]!=shown[:2]]
    state=dict(model=model,action="no_validation_pair",shown=shown,holdout_pairs=len(validation))
    if not validation:return state
    prompt=(model_prompt(item["case"])+'\nReturn ONLY {"field":"sense|linear|quadratic","value":...} '
            'or {"keep":true}. Correct at most one objective field supported by the original problem. '
            'Orders are signs of cost(x)-cost(y) after converting to minimization.\n'+
            json.dumps(dict(current=compact(model)["objective"],quote=c["quote"],hypothesized_objective=obj,
                            permitted_fields=fields,comparison=shown),separators=(",",":")))
    text=ask(prompt,seed_for(cid,seed,"objective_patch"));state["text"]=text
    try:new=amend(item["case"],model,fields,text)
    except (ValueError,TypeError,KeyError,IndexError):return dict(state,action="keep_or_invalid")
    # Require the edit to match the source hypothesis on every retained check.
    xs=sorted(set(tuple(x) for v in [shown]+validation for x in v[:2]))
    rank=dict(zip(xs,values(new,xs)))
    if any(int(np.sign(rank[tuple(x)]-rank[tuple(y)]))!=correct for x,y,old,correct in [shown]+validation):
        return dict(state,action="rollback")
    return dict(state,model=new,action="accepted")
