"""SFT then frozen-reference DPO on the same confirmed pairs for C and D."""
import argparse
import json
import random
import time
import torch
from torch.nn import functional as F
from qihc.experiment_runtime.storage import load,save,worker_lock
from qihc.modeling.protocol import load_object
from experiments.feedback_active_v3.runtime import seed_for
from .data import RUN,MODEL,DATA,cohort
from .energy import KEEP,weights,diagnostic,prompt
from .engine import Engine
from .learn import put_checkpoint


def paired_loss(chosen,rejected,ref_chosen,ref_rejected):
    return -F.logsigmoid(.1*((chosen-rejected)-(ref_chosen-ref_rejected)))


def logprob(engine,p,action):
    text=json.dumps(action,separators=(",",":"))
    ids=engine.encode(text)+[engine.tok.eos_token_id]
    h=engine.hidden(p,ids)
    lp=engine.distribution(h)
    return lp.gather(1,torch.tensor(ids,device=h.device)[:,None]).sum()


def gold_inputs():
    rows=load(DATA/"evaluation/cases.gold.json")+load(DATA/"evaluation/cases.public_external.gold.json")
    return {r["id"]:r["gold"] for r in rows}


def run(method,seed,mode):
    root=RUN/mode
    out=root/"models"/str(seed)/method
    settings=load(RUN/"energy/settings.json")
    with worker_lock(out/"lock"):
        torch.manual_seed(seed)
        engine=Engine(MODEL)
        rows={r["case"]["id"]:r for r in load(RUN/"train.json")}
        ids=list(settings["pair_ids"])
        if mode=="pilot": ids=sorted(ids,key=lambda i:seed_for(i,"pilot"))[:12]
        if not ids: raise RuntimeError("No confirmed preference labels")
        examples={}
        for cid in ids:
            data=load(RUN/"energy/preferences"/(cid+".json"))
            examples[cid]=(prompt(rows[cid]["case"],rows[cid]["gold"],data["diag"],method),data["pair"])
        final=out/"adapter.pt"
        ck=out/"checkpoint.pt"
        refpath=out/"energy_sft.pt"
        if not final.exists():
            order=[]
            for epoch in range(2 if mode=="pilot" else 4):
                order_ids=ids.copy()
                random.Random(seed+epoch).shuffle(order_ids)
                phase="sft" if epoch<(1 if mode=="pilot" else 2) else "dpo"
                order.extend((phase,c) for c in order_ids)
            optimizer=torch.optim.AdamW(engine.params.values(),lr=2e-5)
            index=0
            if ck.exists():
                state=torch.load(ck,map_location="cpu",weights_only=True)
                engine.restore(state["weights"])
                optimizer.load_state_dict(state["optimizer"])
                index=state["step"]
            for step in range(index,len(order)):
                started=time.monotonic()
                phase,cid=order[step]
                p,pair=examples[cid]
                optimizer.zero_grad(set_to_none=True)
                if phase=="sft":
                    engine.net.train()
                    loss=-logprob(engine,p,pair["chosen"])/len(engine.encode(json.dumps(pair["chosen"],separators=(",",":"))))
                else:
                    if not refpath.exists(): put_checkpoint(refpath,engine.snapshot())
                    for g in optimizer.param_groups: g["lr"]=5e-6
                    active=engine.snapshot()
                    engine.restore(torch.load(refpath,map_location="cpu",weights_only=True))
                    engine.net.eval()
                    with torch.no_grad():
                        rc=logprob(engine,p,pair["chosen"])
                        rr=logprob(engine,p,pair["rejected"])
                    engine.restore(active); engine.net.train()
                    loss=paired_loss(logprob(engine,p,pair["chosen"]),logprob(engine,p,pair["rejected"]),rc,rr)
                if not torch.isfinite(loss): raise RuntimeError("Nonfinite energy loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(list(engine.params.values()),1.,error_if_nonfinite=True)
                optimizer.step()
                save(out/"steps"/f"{step:04d}.json",dict(phase=phase,case_id=cid,loss=float(loss.detach()),seconds=time.monotonic()-started))
                if (step+1)%8==0 or step+1==len(order) or order[step+1][0]!=phase:
                    put_checkpoint(ck,dict(weights=engine.snapshot(),optimizer=optimizer.state_dict(),step=step+1))
            put_checkpoint(final,engine.snapshot())
        engine.restore(torch.load(final,map_location="cpu",weights_only=True))
        for track in ("end_to_end","gold"):
            track_root=root if track=="end_to_end" else root/"gold"
            fixed_models=gold_inputs() if track=="gold" else {}
            for case in cohort(mode,seed):
                cid=case["id"]; path=track_root/"models"/str(seed)/method/"predictions"/(cid+".json")
                if path.exists(): continue
                model=(load(root/"models"/str(seed)/"sft/predictions"/path.name)["model"]
                       if track=="end_to_end" else fixed_models[cid]); action=KEEP; raw=None; error=None
                started=time.monotonic()
                if model is not None and len(model["constraints"])+len(model["logic"])>=2:
                    diag=diagnostic(track_root,cid,seed,model,settings["base"])
                    response=engine.evaluate(prompt(case,model,diag,method),seed_for(cid,seed,"energy"))
                    raw=response["choices"][0]["message"]["content"]
                    try:
                        if response["choices"][0]["finish_reason"]!="stop": raise ValueError("incomplete")
                        candidate=load_object(raw)
                        weights(model,settings["base"],candidate)
                        action=candidate
                    except (ValueError,TypeError,KeyError) as exc: error=str(exc)
                save(path,dict(case_id=cid,stage=case["stage"],model=model,action=action,raw=raw,error=error,
                               decision_seconds=time.monotonic()-started))
        save(out/"done.json",dict(seed=seed,method=method,count=len(cohort(mode,seed))))


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("method",choices=["best_lm","trajectory_lm"])
    p.add_argument("seed",type=int)
    p.add_argument("mode",choices=["pilot","formal"])
    a=p.parse_args()
    run(a.method,a.seed,a.mode)
