import numpy as np
from qihc.generic.feedback.verification import equivalent,normalized
from qihc.modeling.active_protocol_v2 import model_prompt
from experiments.posttraining_v1.learn import parsed,feedback_prompt,repair_weights
from experiments.posttraining_v1.data import target
from experiments.posttraining_v1.witness import probe
from .feedback import explain,prompt as repair_prompt,remaining

def step(engine,arm,row,seed,reference,cfg):
    base=model_prompt(row['case']);gold=row['gold'];samples=[]
    for k in range(cfg['rollouts']):
        ids,text=engine.sample(base,seed+k);model,error=parsed(text,row['case']);sem=equivalent(model,gold)
        reward={'invalid':-1.,'disproved':-1.,'verified_correct':1.}.get(sem['status'])
        samples.append(dict(ids=ids,text=text,model=model,error=error,semantic=sem,reward=reward))
    known=[s for s in samples if s['reward'] is not None]
    rewards=np.asarray([s['reward'] for s in known]);update=len(known)>1 and rewards.std()>1e-8
    if update:
        for s,a in zip(known,(rewards-rewards.mean())/rewards.std()):
            s['advantage']=float(a)
            engine.backward(base,s['ids'],'pg',scale=1/len(known),advantage=float(a),reference=reference)
    for s in samples:
        if 'advantage' not in s:engine.backward(base,s['ids'],'ref',scale=.25,reference=reference)
    # Same candidate count, probe, single repair attempt and coefficients as original P-VCD.
    repairs=[]
    for i,s in enumerate([s for s in samples if s['model'] is not None and s['semantic']['status']!='verified_correct'][:cfg['repair_candidates']]):
        cards,search=probe(s['model'],gold,seed+100*(i+1),seconds=cfg['probe_seconds'])
        s.update(cards=cards,probe_trace=search)
        if not cards:continue
        actual=explain(s['model'],gold,cards) if arm=='localized' else cards
        tp=repair_prompt(base,s['text'],s['model'],actual) if arm=='localized' else feedback_prompt(base,s['text'],cards)
        ids,text=engine.sample(tp,seed+500+i);model,error=parsed(text,row['case']);sem=equivalent(model,gold)
        s.update(feedback=actual,repair=text,repair_tokens=len(ids),repair_semantic=sem,repair_error=error,
                 witness_remaining=remaining(model,gold,actual))
        if sem['status']=='verified_correct':
            answer=target(model);repairs.append((tp,answer,repair_weights(engine.tok,answer,s['model'],model,actual)))
    for tp,answer,weights in repairs:
        engine.backward(base,engine.encode(answer),'vcd',scale=.5/len(repairs),teacher_prompt=tp,weights=weights)
    positives={}
    for s in samples:
        if s['semantic']['status']=='verified_correct':positives.setdefault(repr(normalized(s['model'])),s['model'])
    # One target per original training case/step, never selected by validation accuracy.
    positive_used=arm!='pvcd' and bool(positives)
    if positive_used:
        answer=target(next(iter(positives.values())))
        engine.backward(base,engine.encode(answer)+[engine.tok.eos_token_id],'nll',scale=cfg['positive_scale'])
    anchor=engine.backward(base,engine.encode(row['answer'])+[engine.tok.eos_token_id],'nll',scale=.1)
    return dict(case_id=row['case']['id'],seed=seed,samples=samples,reward_update=bool(update),
                verified_repairs=len(repairs),positive_candidates=sum(s['semantic']['status']=='verified_correct' for s in samples),
                unique_positives=len(positives),positive_used=positive_used,anchor=anchor)
