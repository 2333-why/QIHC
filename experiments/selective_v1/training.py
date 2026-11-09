"""Outcome reward augmented with paired assignments, optionally local distillation."""
import json
import numpy as np
import torch
from qihc.experiment_runtime.storage import save
from qihc.generic.bounded import violations,objective
from qihc.generic.feedback.verification import equivalent,score
from qihc.modeling.active_protocol_v2 import model_prompt
from experiments.feedback_active_v3.cpu import execute
from experiments.feedback_active_v3.runtime import seed_for
from experiments.posttraining_v1.learn import parsed,put_checkpoint,repair_weights
from experiments.posttraining_v1.data import target
from experiments.posttraining_v1.witness import probe
from .protocol import edit,edit_prompt

def agreement(model,gold,states):
    gf=violations(gold,states).sum(axis=1)==0
    mf=violations(model,states).sum(axis=1)==0
    acc=float(np.mean([np.mean(mf[gf==label]==label) for label in (False,True) if np.any(gf==label)]))
    common=states[gf & mf]
    if len(common)<2:return acc
    def order(spec):
        values=objective(spec,common)*(1 if spec["objective"]["sense"]=="min" else -1)
        return np.sign(values[:,None]-values[None,:])[np.triu_indices(len(values),1)]
    return .5*(acc+float(np.mean(order(gold)==order(model))))

def step(engine,arm,row,seed,reference,out,i):
    case,gold=row["case"],row["gold"]; p=model_prompt(case)
    search_seed=seed_for(seed,i)
    ref=execute(dict(stage="reference",gold=gold,seconds=120,seed=seed_for(case["id"],"reference")))
    lo=np.array([v["lower"] for v in gold["variables"]]); hi=np.array([v["upper"] for v in gold["variables"]])
    states=list(np.random.default_rng(search_seed).integers(lo,hi+1,size=(32,len(lo))))
    if ref.get("x") is not None:states.append(ref["x"])
    samples=[]; probes=0
    for k in range(4):
        ids,text=engine.sample(p,search_seed+k)
        m,err=parsed(text,case)
        sol=execute(dict(stage="solve",model=m,backend="scip",seconds=2,chains=32,seed=search_seed+k))
        s=score(gold,sol,ref)
        if sol.get("x") is not None:states.append(sol["x"])
        cards=[]
        if m is not None and equivalent(m,gold)["status"]!="verified_correct" and probes<2:
            cards,_=probe(m,gold,search_seed+100*(k+1));probes+=1
            for c in cards:
                states.append(c["x"])
                if "y" in c:states.append(c["y"])
        samples.append(dict(ids=ids,model=m,text=text,cards=cards,error=err,
                        outcome=(int(s["original_feasible"])+int(s["optimal_hit"]))/2))
    xs=np.unique(states,axis=0)
    for s in samples:
        s["reward"]=-1. if s["model"] is None else .5*s["outcome"]+.5*agreement(s["model"],gold,xs)
    rewards=np.array([s["reward"] for s in samples])
    if rewards.std()>1e-8:
        for s,a in zip(samples,(rewards-rewards.mean())/rewards.std()):
            engine.backward(p,s["ids"],"pg",scale=.25,advantage=float(a),reference=reference)
    repairs=[]
    if arm=="pair_local":
        for k,s in enumerate(samples):
            if not s["cards"]:continue
            tp=edit_prompt(p,s["model"],s["cards"])
            _,text=engine.sample(tp,search_seed+500+k)
            try:new=edit(text,s["model"],case)
            except (ValueError,KeyError,IndexError,TypeError):new=None
            status=equivalent(new,gold)["status"]
            s.update(edit=text,edit_status=status)
            if status=="verified_correct":
                answer=target(new)
                weights=repair_weights(engine.tok,answer,s["model"],new,s["cards"])
                # Teacher predicts a FULL model in this distillation context.
                tp=p+"\nReturn the corrected full model. Previous attempt: "+s["text"]+"\nEvidence: "+json.dumps(s["cards"])
                repairs.append((tp,answer,weights))
        for tp,answer,weights in repairs:
            engine.backward(p,engine.encode(answer),"vcd",scale=.5/len(repairs),teacher_prompt=tp,weights=weights)
    engine.backward(p,engine.encode(row["answer"])+[engine.tok.eos_token_id],"nll",scale=.1)
    save(out/"trace"/f"{i:04}.json",dict(cid=case["id"],states=xs.tolist(),samples=samples,verified=len(repairs)))

def fit(engine,arm,seed,mode,out,rows,reference):
    import random,time
    final=out/"adapter.pt"; ck=out/"checkpoint.pt"
    if not final.exists():
        engine.restore(reference)
        optimizer=torch.optim.AdamW(engine.params.values(),lr=5e-6)
        index=0
        if ck.exists():
            state=torch.load(ck,map_location="cpu",weights_only=True)
            engine.restore(state["weights"]);optimizer.load_state_dict(state["optimizer"]);index=state["step"]
        order=[]
        for epoch in range(1 if mode=="pilot" else 2):
            chunk=list(rows);random.Random(seed+epoch).shuffle(chunk);order+=chunk
        for i in range(index,len(order)):
            started=time.monotonic()
            optimizer.zero_grad(set_to_none=True)
            step(engine,arm,order[i],seed,reference,out,i)
            norm=torch.nn.utils.clip_grad_norm_(list(engine.params.values()),1.,error_if_nonfinite=True)
            optimizer.step()
            save(out/"steps"/f"{i:04}.json",dict(step=i+1,seconds=time.monotonic()-started,gradient_norm=float(norm)))
            if (i+1)%8==0 or i+1==len(order):
                put_checkpoint(ck,dict(weights=engine.snapshot(),optimizer=optimizer.state_dict(),step=i+1))
        put_checkpoint(final,engine.snapshot())
    engine.restore(torch.load(final,map_location="cpu",weights_only=True))
