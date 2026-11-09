"""Gold-blind feedback using one frozen SFT draft and common repairs/checks."""
from qihc.experiment_runtime.storage import load,save
from experiments.posttraining_v1.learn import parsed
from experiments.posttraining_v1.evaluate import worker_lock_wait
from experiments.feedback_active_v3.runtime import seed_for
from qihc.modeling.active_protocol_v2 import model_prompt,source_prompt,parse_checks
from qihc.generic.feedback.active import probe
from .protocol import edit_prompt,edit,witnessed,protected,objective_card
import json
from .common import generated

def cached(path,make):
    with worker_lock_wait(path):
        if not path.exists():save(path,make())
        return load(path)

def decide(engine,arm,case,draft,folder,seed,cfg):
    initial=draft["model"]; prompt=model_prompt(case); cid=case["id"]
    def repair():
        message=prompt+"\nCorrect this unparseable attempt and return a full model:\n"+draft.get("raw","")+"\nParser: "+str(draft.get("error"))
        text=generated(engine,message,seed_for(cid,seed,"repair"))
        model,error=parsed(text,case)
        return dict(model=model,error=error,raw=text)
    if arm=="sft":return dict(model=initial,decision="base")
    if initial is None:
        return dict(cached(folder/"repair"/(cid+".json"),repair),decision="compile_repair")
    if arm=="compile":return dict(model=initial,decision="keep")
    def checks():
        text=generated(engine,source_prompt(case),seed_for(cid,seed,"source"))
        try:return parse_checks(text,case)
        except (ValueError,TypeError,KeyError):return dict(checks=[],error="invalid source checks")
    src=cached(folder/"checks"/(cid+".json"),checks)
    if not src["checks"]:return dict(model=initial,decision="abstain_no_checks")
    has_obj=any(c["kind"]=="objective" for c in src["checks"])
    budget=dict(cfg,probe_seconds=cfg["probe_seconds"]/(2 if has_obj else 1))
    result=probe(initial,src,"X" if arm=="scip_local" else "P2",budget,seed_for(cid,seed,"probe"))
    cards=witnessed(initial,src,result["cards"])[:4]
    if has_obj:cards=(objective_card(initial,src,arm,budget["probe_seconds"],seed_for(cid,seed,"objective"))+cards)[:4]
    while cards and len(engine.encode(json.dumps(cards,separators=(",",":"))))>1024:cards.pop()
    state=dict(model=initial,decision="abstain",cards=cards,probe_status=result["status"])
    if cards:
        text=generated(engine,edit_prompt(prompt,initial,cards),seed_for(cid,seed,"edit"))
        state["edit"]=text
        try:
            new=edit(text,initial,case)
            if protected(initial,new,src,cards):state.update(model=new,decision="accepted")
            else:state["decision"]="rolled_back"
        except (ValueError,TypeError,KeyError,IndexError) as e:state.update(decision="invalid_edit",error=str(e))
    return state
