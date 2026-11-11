"""Keep P-VCD's certified reward; enhance evidence, repair loop and preservation."""
import json
import random
import time
import torch
import numpy as np
from qihc.experiment_runtime.storage import load, save, value_digest
from qihc.generic.feedback.verification import equivalent
from qihc.modeling.active_protocol_v2 import model_prompt
from experiments.posttraining_v1.learn import parsed, repair_weights, put_checkpoint
from experiments.posttraining_v1.data import target
from experiments.feedback_active_v3.runtime import seed_for
from .common import CFG, RUN, donor, training_rows
from .evidence import collect, cards_from_states

def repair_prompt(prompt, text, cards, error=None):
    return (prompt+'\nTraining-only verified counterexamples follow. Correct both over-permissive and '
        'over-restrictive constraints and objective ordering (including ties). Preserve already correct '
        'parts. Return the COMPLETE corrected JSON model, not a patch.\nAttempt: '+text+
        '\nEvidence: '+json.dumps(cards,ensure_ascii=False,separators=(',',':'))+
        ('\nParse failure: '+error if error else ''))

def repair_chain(engine, case, gold, sample, seed, arm, cfg):
    original=sample['model']; current=original; text=sample['text']; error=None
    bank=[]; attempts=[]
    for round_id in range(cfg['repair_rounds']):
        if arm=='cycle_static': current=original; text=sample['text']; error=None
        # If parsing failed, retain the last executable model for refreshed evidence.
        states,search=collect(current,gold,seed+100*round_id,cfg,arm=='cycle_uniform')
        bank.extend(states)
        cards=cards_from_states(current,gold,bank,cfg['cards'])
        if not cards and error is None:
            attempts.append(dict(round=round_id,search=search,status='no_evidence',cards=[]))
            continue
        tp=repair_prompt(model_prompt(case),text,cards,error)
        ids,reply=engine.sample(tp,seed+1000+round_id)
        new,error=parsed(reply,case); sem=equivalent(new,gold)
        attempts.append(dict(round=round_id,search=search,cards=cards,reply=reply,tokens=len(ids),
                             semantic=sem,error=error,status='generated'))
        if sem['status']=='verified_correct':
            answer=target(new)
            return dict(answer=answer,teacher_prompt=tp,
                        weights=repair_weights(engine.tok,answer,current,new,cards)), attempts
        if new is not None:
            current=new; text=reply
        # On parse failure both the next Attempt and its cards continue to refer
        # to the last executable model; the failed reply remains in the trace.
    return None,attempts

def step(engine, arm, row, seed, reference, parent, cfg=CFG):
    prompt=model_prompt(row['case']); samples=[]
    for k in range(cfg['rollouts']):
        ids,text=engine.sample(prompt,seed+k)
        model,error=parsed(text,row['case']); sem=equivalent(model,row['gold'])
        reward={'invalid':-1.,'disproved':-1.,'verified_correct':1.}.get(sem['status'])
        samples.append(dict(ids=ids,text=text,model=model,error=error,semantic=sem,reward=reward))
    known=[s for s in samples if s['reward'] is not None]
    rewards=np.array([s['reward'] for s in known]); updated=len(known)>1 and rewards.std()>1e-8
    if updated:
        for s,adv in zip(known,(rewards-rewards.mean())/rewards.std()):
            s['advantage']=float(adv)
            engine.backward(prompt,s['ids'],'pg',scale=1/len(known),advantage=float(adv),reference=reference)
    for s in samples:
        if 'advantage' not in s and s['ids']:
            engine.backward(prompt,s['ids'],'ref',scale=1/cfg['rollouts'],reference=reference)
    repairs={}; verified=0
    distinct={}
    for s in samples:
        if s['model'] is not None and s['semantic']['status']!='verified_correct':
            distinct.setdefault(value_digest(s['model']),s)
    eligible=list(distinct.values())[:cfg['repair_candidates']]
    for i,s in enumerate(eligible):
        repair,trace=repair_chain(engine,row['case'],row['gold'],s,seed+2000*(i+1),arm,cfg)
        s['repair_trace']=trace
        if repair:
            verified+=1
            repairs.setdefault(value_digest(repair['answer']),repair)
    for r in repairs.values():
        engine.backward(prompt,engine.encode(r['answer']),'vcd',scale=cfg['distill_scale']/len(repairs),
                        teacher_prompt=r['teacher_prompt'],weights=r['weights'])
    guard=False
    if arm!='cycle_no_guard' and parent is not None and equivalent(parent,row['gold'])['status']=='verified_correct':
        engine.backward(prompt,engine.encode(target(parent))+[engine.tok.eos_token_id],
                        'nll',scale=cfg['guard_scale'])
        guard=True
    anchor=engine.backward(prompt,engine.encode(row['answer'])+[engine.tok.eos_token_id],'nll',scale=cfg['anchor_scale'])
    return dict(case_id=row['case']['id'],seed=seed,samples=samples,reward_update=bool(updated),
                verified_repairs=verified,unique_targets=len(repairs),guard=guard,anchor=anchor)

def fit(engine, arm, seed, mode, out, reference):
    rows=training_rows(mode); order=[]
    for epoch in range(1 if mode=='pilot' else CFG['epochs']):
        batch=list(rows); random.Random(seed+epoch).shuffle(batch); order+=batch
    final=out/'adapter.pt'; ck=out/'checkpoint.pt'
    if final.exists():
        engine.restore(torch.load(final,map_location='cpu',weights_only=True)); return
    engine.restore(reference)
    optim=torch.optim.AdamW(engine.params.values(),lr=CFG['learning_rate']); start=0
    if ck.exists():
        state=torch.load(ck,map_location='cpu',weights_only=True)
        engine.restore(state['weights']); optim.load_state_dict(state['optimizer']); start=state['step']
    for i in range(start,len(order)):
        began=time.monotonic(); optim.zero_grad(set_to_none=True)
        parent=load(RUN/'inputs/parents'/str(seed)/(order[i]['case']['id']+'.json'))['model']
        result=step(engine,arm,order[i],seed_for(seed,i),reference,parent)
        norm=torch.nn.utils.clip_grad_norm_(list(engine.params.values()),1.,error_if_nonfinite=True)
        optim.step()
        save(out/'trace'/f'{i:04}.json',result)
        save(out/'steps'/f'{i:04}.json',dict(step=i+1,seconds=time.monotonic()-began,gradient_norm=float(norm),
              gpu_peak_bytes=torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0))
        if (i+1)%CFG['checkpoint_steps']==0 or i+1==len(order):
            put_checkpoint(ck,dict(weights=engine.snapshot(),optimizer=optim.state_dict(),step=i+1))
        print(arm,seed,i+1,len(order),'verified',result['verified_repairs'],'guard',result['guard'],flush=True)
    weights=engine.snapshot()
    if all(torch.equal(weights[k],reference[k]) for k in weights): raise RuntimeError('Adapter did not change')
    put_checkpoint(final,weights)
