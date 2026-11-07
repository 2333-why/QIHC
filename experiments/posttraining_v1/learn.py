"""One seed/method per GPU. Resume at atomic optimizer-state checkpoints."""
import argparse
import json
import os
import random
import time
from pathlib import Path
import numpy as np
import torch
from qihc.experiment_runtime.storage import load, save, worker_lock
from qihc.generic.feedback.verification import equivalent, score
from qihc.generic.bounded import violations
from qihc.modeling.active_protocol_v2 import model_prompt, parse_model, compact
from experiments.feedback_active_v3.cpu import execute
from experiments.feedback_active_v3.runtime import seed_for
from .data import RUN, MODEL, METHODS, target
from .engine import Engine
from .witness import probe


def parsed(text, case):
    try: return parse_model(text, case), None
    except (ValueError, KeyError, TypeError, IndexError, OverflowError) as e: return None, str(e)[:500]


def feedback_prompt(prompt, attempt, feedback):
    return (prompt + "\nReview this attempt using the following training feedback. "
            "Return a complete revised JSON model.\nAttempt: " + attempt + "\nFeedback: "
            + json.dumps(feedback, separators=(",",":")))


def repair_weights(tok, answer, old, new, cards):
    # Only uniquely supported changed nodes are upweighted. Ignore ambiguous mappings.
    spans = set()
    for card in cards:
        if card["kind"] == "objective" and old["objective"] != new["objective"]:
            node = compact(new)["objective"]
        elif card["kind"] == "feasibility" and not card["correct_feasible"]:
            x = np.asarray(card["x"])
            residual = violations(new,x)
            old_rows = [{k:v for k,v in c.items() if k!="id"} for c in old["constraints"]]
            indices = [i for i,c in enumerate(new["constraints"]) if residual[1+i]>0
                       and {k:v for k,v in c.items() if k!="id"} not in old_rows]
            if len(indices)!=1: continue
            node = compact(new)["constraints"][indices[0]]
        else: continue
        fragment = json.dumps(node, ensure_ascii=False,separators=(",",":"))
        if answer.count(fragment)==1:
            start=answer.index(fragment)
            spans.add((start,start+len(fragment)))
    offsets=tok(answer, add_special_tokens=False,return_offsets_mapping=True)["offset_mapping"]
    return [2. if any(start<b and end>a for a,b in spans) else 1. for start,end in offsets]


def train_one(engine, method, row, seed, ref_state, out, step):
    prompt=model_prompt(row["case"])
    if method=="sft":
        ids=engine.encode(row["answer"])+[engine.tok.eos_token_id]
        return {"anchor":engine.backward(prompt,ids,"nll")}
    case, gold=row["case"],row["gold"]
    samples=[]
    reference=None
    if method=="solver_rl":
        reference=execute(dict(stage="reference",gold=gold,seconds=120,seed=seed_for(case["id"],"reference")))
    for i in range(4):
        ids,text=engine.sample(prompt,seed+i)
        model,error=parsed(text,case)
        sem=equivalent(model,gold)
        reward={"invalid":-1.,"disproved":-1.,"verified_correct":1.}.get(sem["status"])
        diagnostic=error or sem
        if method=="solver_rl":
            solution=execute(dict(stage="solve",model=model,backend="scip",seconds=2,chains=32,seed=seed+i))
            if solution.get("status")=="execution_error": raise RuntimeError(str(solution))
            scoring=score(gold,solution,reference)
            reward=(-1. if model is None else (float(scoring["original_feasible"])+float(scoring["optimal_hit"]))/2)
            diagnostic=scoring
        samples.append(dict(ids=ids,text=text,model=model,error=error,semantic=sem,reward=reward,diagnostic=diagnostic))
    known=[s for s in samples if s["reward"] is not None]
    if method in ("pvcd","solver_rl") and len(known)>1:
        rewards=np.array([s["reward"] for s in known])
        if rewards.std()>1e-8:
            for sample,adv in zip(known,(rewards-rewards.mean())/rewards.std()):
                sample["advantage"]=float(adv)
                engine.backward(prompt,sample["ids"],"pg",scale=1/len(known),advantage=float(adv),reference=ref_state)
    if method=="sdpo":
        for sample in samples:
            tp=feedback_prompt(prompt,sample["text"],sample["diagnostic"])
            engine.backward(prompt,sample["ids"],"sdpo",scale=.25,teacher_prompt=tp,reference=ref_state)
    if method=="pvcd":
        for sample in samples:
            if "advantage" not in sample:
                engine.backward(prompt,sample["ids"],"ref",scale=.25,reference=ref_state)
        eligible=[s for s in samples if s["model"] and s["semantic"]["status"]!="verified_correct"][:2]
        repairs=[]
        for i,s in enumerate(eligible):
            cards,trace=probe(s["model"],gold,seed+100*(i+1))
            s.update(cards=cards,probe_trace=trace)
            if not cards: continue
            tp=feedback_prompt(prompt,s["text"],cards)
            repair_ids,repair=engine.sample(tp,seed+500+i)
            repaired,error=parsed(repair,case)
            check=equivalent(repaired,gold)
            s.update(repair=repair,repair_tokens=len(repair_ids),repair_semantic=check,repair_error=error)
            if check["status"]=="verified_correct":
                answer=target(repaired)
                repairs.append((tp,answer,repair_weights(engine.tok,answer,s["model"],repaired,cards)))
        for tp,answer,weights in repairs:
            engine.backward(prompt,engine.encode(answer),"vcd",scale=.5/len(repairs),teacher_prompt=tp,weights=weights)
    anchor=engine.backward(prompt,engine.encode(row["answer"])+[engine.tok.eos_token_id],"nll",scale=.1)
    save(out/"trace"/f"{step:04d}.json",dict(id=case["id"],seed=seed,samples=samples))
    return {"anchor":anchor,"signals":sum("advantage" in s for s in samples)}


def put_checkpoint(path, obj):
    tmp=path.with_suffix(".tmp")
    torch.save(obj,tmp)
    os.replace(tmp,path)


def work(method, seed, mode):
    out=RUN/mode/"models"/str(seed)/method
    with worker_lock(out/"lock"):
        torch.manual_seed(seed)
        engine=Engine(MODEL)
        train=load(RUN/"train.json")
        if mode=="pilot": train=sorted(train,key=lambda r:seed_for(r["case"]["id"],"pilot"))[:12]
        state_path=out/"checkpoint.pt"
        final=out/"adapter.pt"
        if not final.exists():
            reference=None
            if method!="sft":
                reference=torch.load(out.parent/"sft"/"adapter.pt",map_location="cpu",weights_only=True)
                engine.restore(reference)
            optim=torch.optim.AdamW(engine.params.values(),lr=2e-5 if method=="sft" else 5e-6)
            index=0
            if state_path.exists():
                saved=torch.load(state_path,map_location="cpu",weights_only=True)
                engine.restore(saved["weights"])
                optim.load_state_dict(saved["optimizer"])
                index=saved["step"]
            schedule=[]
            epochs=1 if mode=="pilot" else (3 if method=="sft" else 2)
            for epoch in range(epochs):
                order=list(range(len(train)))
                random.Random(seed+epoch).shuffle(order)
                schedule.extend(order)
            for step in range(index,len(schedule)):
                started=time.monotonic()
                optim.zero_grad(set_to_none=True)
                info=train_one(engine,method,train[schedule[step]],seed_for(seed,step),reference,out,step)
                norm=torch.nn.utils.clip_grad_norm_(list(engine.params.values()),1.,error_if_nonfinite=True)
                optim.step()
                save(out/"steps"/f"{step:04d}.json",dict(step=step,seconds=time.monotonic()-started,
                     gpu_peak_bytes=torch.cuda.max_memory_allocated(),gradient_norm=float(norm),**info))
                if (step+1)%8==0 or step+1==len(schedule):
                    put_checkpoint(state_path,dict(weights=engine.snapshot(),optimizer=optim.state_dict(),step=step+1))
                print(method,seed,step+1,len(schedule),float(norm),info,flush=True)
            weights=engine.snapshot()
            if reference is not None and all(torch.equal(weights[k],reference[k]) for k in weights):
                raise RuntimeError("No adapter update")
            if method=="sft" and not any(t.abs().max()>0 for k,t in weights.items() if k.endswith(".b")):
                raise RuntimeError("SFT did not update")
            put_checkpoint(final,weights)
        if method!="sft":
            signal={"groups":0,"reward_updates":0,"distilled_repairs":0,"witnessed_candidates":0}
            for path in (out/"trace").glob("*.json"):
                samples=load(path)["samples"]
                signal["groups"]+=1
                signal["reward_updates"]+=int(any("advantage" in s for s in samples))
                signal["distilled_repairs"]+=sum(s.get("repair_semantic",{}).get("status")=="verified_correct" for s in samples)
                signal["witnessed_candidates"]+=sum(bool(s.get("cards")) for s in samples)
            save(out/"training_signals.json",signal)
        engine.restore(torch.load(final,map_location="cpu",weights_only=True))
        cases=([r["case"] for r in load(RUN/"validation.json")] if mode=="pilot"
               else load(RUN/"test_public.json"))
        for case in cases:
            path=out/"predictions"/(case["id"]+".json")
            if path.exists(): continue
            response=engine.evaluate(model_prompt(case),seed_for(case["id"],0,"initial"))
            text=response["choices"][0]["message"]["content"]
            model,error=parsed(text,case) if response["choices"][0]["finish_reason"]=="stop" else (None,"length")
            save(path,dict(case_id=case["id"],model=model,error=error,raw=text,usage=response["usage"]))
        save(out/"done.json",dict(count=len(cases),method=method,seed=seed))


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("method",choices=METHODS)
    p.add_argument("seed",type=int)
    p.add_argument("mode",choices=["pilot","formal"])
    a=p.parse_args()
    work(a.method,a.seed,a.mode)
