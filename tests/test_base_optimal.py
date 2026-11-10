import copy
import json
from experiments.optimal_feedback_v1 import policy

def fixture():
    names=["x0","x1","x2"]
    obj=dict(sense="max",linear=[1,2,3],quadratic=[])
    m=dict(schema="bounded-integer-v2",variables=[dict(name=n,lower=0,upper=1) for n in names],
           objective=dict(obj,sense="min"),constraints=[],logic=[])
    c=dict(kind="objective",quote="maximize",**obj)
    return dict(case=dict(id="fixture",question="maximize x0 + 2*x1 + 3*x2",variable_names=names),
                model=m,seed=17,checks=dict(checks=[c]),raw="",error=None)

def test_accept_and_rollback(monkeypatch):
    i=fixture();ask=lambda *_:json.dumps(dict(field="sense",value="max"))
    base=policy.shared(i,ask)
    # Hold back different assignments from the demonstrated counterexample.
    monkeypatch.setattr(policy,"pool",lambda m,s,a,k,b,hold: [[0,0,0],[0,1,0]] if hold else [[0,0,0],[1,0,0]])
    for a in ("C","D"):
        y=policy.decide(i,a,base,ask)
        assert y["action"]=="accepted" and y["model"]["objective"]["sense"]=="max"
        assert i["model"]["objective"]["sense"]=="min"
        y=policy.decide(i,a,base,lambda *_:'{"field":"sense","value":"min"}')
        assert y["action"]=="rollback" and y["model"]==i["model"]
    assert policy.decide(i,"B",base,ask)["model"]==i["model"]
    monkeypatch.setattr(policy,"pool",lambda *args:[[0,0,0],[1,0,0]])
    assert policy.decide(i,"D",base,ask)["action"]=="no_validation_pair"

def test_actual_probes_and_shared_compilation():
    i=fixture()
    base=policy.shared(i,lambda *_:"")
    for arm in ("C","D"):
        xs=policy.pool(i["model"],dict(i["model"],objective={k:base["check"][k] for k in ("sense","linear","quadratic")}),arm,17,.8,False)
        assert len(xs)>=2
    broken=dict(i,model=None,error="bad")
    answer=json.dumps({"model":policy.compact(i["model"])})
    repair=policy.shared(broken,lambda *_:answer)
    assert repair["model"]==i["model"]
    for arm in ("B","C","D"): assert policy.decide(broken,arm,repair,None)==repair
    assert policy.decide(broken,"A",repair,None)["model"] is None
