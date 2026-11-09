"""Single structural edit, validated through the frozen E004 protocol."""
import copy
import json
import numpy as np
from qihc.modeling.active_protocol_v2 import compact,parse_model
from qihc.modeling.protocol import load_object
from qihc.generic.bounded import violations,objective
from qihc.generic.feedback.active import check_residual

def edit_prompt(prompt,model,cards):
    return prompt+'\nInstead of a full model return one local edit: {"action":"keep"}, or '+\
        '{"action":"append","row":ROW}, {"action":"replace","id":ID,"row":ROW}, '+\
        '{"action":"delete","id":ID}, or {"action":"objective","field":"sense|linear|quadratic","value":VALUE}. '+\
        'Confirm the evidence from the original question.\nMODEL:'+json.dumps(compact(model),separators=(',',':'))+'\nEVIDENCE:'+json.dumps(cards,separators=(',',':'))

def edit(text,old,case):
    a=load_object(text)
    m=copy.deepcopy(compact(old)); op=a.get("action")
    if a=={"action":"keep"}: return old
    if op=="objective" and set(a)=={"action","field","value"} and a["field"] in ("sense","linear","quadratic"):
        m["objective"][a["field"]]=a["value"]
    elif op=="append" and set(a)=={"action","row"}: m["constraints"].append(a["row"])
    elif op in ("replace","delete") and set(a)==({"action","id","row"} if op=="replace" else {"action","id"}):
        i=next((i for i,r in enumerate(m["constraints"]) if r["id"]==a["id"]),None)
        if i is None: raise ValueError("Unknown row")
        if op=="delete": m["constraints"].pop(i)
        else: m["constraints"][i]=a["row"]
    else: raise ValueError("Invalid local edit")
    return parse_model(json.dumps({"model":m}),case)

def witnessed(model,checks,cards):
    by_id={c["id"]:c for c in checks["checks"] if c["kind"]=="constraint"}
    return [c for c in cards if c.get("check_id") in by_id and c.get("x") is not None and
            violations(model,c["x"]).sum()==0 and
            np.asarray(check_residual(by_id[c["check_id"]],np.asarray([c["x"]]))).item()>0]

def order(spec,x,y):
    sign=1 if spec["objective"]["sense"]=="min" else -1
    return int(np.sign(sign*(objective(spec,x)-objective(spec,y))))

def objective_card(model,checks,arm,seconds,seed):
    import time
    from qihc.generic.feedback.search import sample
    from qihc.generic.solver_baselines.registry import solve
    c=next((c for c in checks["checks"] if c["kind"]=="objective"),None)
    if c is None:return []
    source=dict(model,objective={k:c[k] for k in ("sense","linear","quadratic")})
    states=[]; deadline=time.monotonic()+seconds
    for i,(m,reverse) in enumerate(((model,False),(source,False),(model,True),(source,True))):
        left=deadline-time.monotonic()
        if left<=.01:break
        m=copy.deepcopy(m)
        if reverse:m["objective"]["sense"]="max" if m["objective"]["sense"]=="min" else "min"
        budget=left/(4-i)
        if arm=="pbit_local":
            states.extend(sample(m,budget,seed+i,chains=32,temperature=(.5,1.,.25,.5)[i],read_q=False)["final"])
        else:
            x=solve(m,"scip",budget,seed+i).get("x")
            if x is not None:states.append(x)
    if not states:return []
    xs=np.unique(states,axis=0);xs=xs[violations(model,xs).sum(axis=1)==0]
    for i,x in enumerate(xs):
        for y in xs[i+1:]:
            expected=order(source,x,y)
            if expected!=order(model,x,y):
                return [dict(kind="objective_pair",check_id=c["id"],quote=c["quote"],x=x.tolist(),y=y.tolist(),source_order=expected)]
    return []

def protected(old,new,checks,cards):
    if not cards:return False
    for c in cards:
        if c.get("kind")=="objective_pair":
            if violations(new,[c["x"],c["y"]]).sum()!=0 or order(new,c["x"],c["y"])!=c["source_order"]:return False
        elif violations(new,c["x"]).sum()==0:return False
    for c in checks["checks"]:
        if c["kind"]=="constraint":
            def present(m): return any(all(r[k]==c[k] for k in ("a","rhs","op")) for r in m["constraints"])
            if present(old) and not present(new): return False
        elif c["kind"]=="objective":
            fields=("sense","linear","quadratic")
            if all(old["objective"][k]==c[k] for k in fields) and new["objective"]!=old["objective"]:return False
    return True
