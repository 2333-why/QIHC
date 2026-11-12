"""Read-only pilot evidence collection; stdlib only, no model loading or training."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

def read(path):
    return json.loads(path.read_text(encoding='utf-8'))

def collect(run, mode='pilot', examples_per_kind=2):
    run=Path(run); folder=run/mode
    inputs=read(run/'inputs/train.json')
    train={r['case']['id']:r for r in inputs}
    result=dict(run=str(run),mode=mode,arms={},examples=[],evidence_sha256={})
    for name in ('manifest.json',f'{mode}/gate.json',f'{mode}/summary.json'):
        path=run/name
        if path.exists():
            result[name]=read(path)
            result['evidence_sha256'][name]=hashlib.sha256(path.read_bytes()).hexdigest()
    paths=sorted(folder.glob('models/*/*/trace/*.json'))
    if not paths: raise ValueError('No training traces found: '+str(folder))
    counts=defaultdict(Counter); by_round=defaultdict(Counter); picked=Counter()
    distinct=defaultdict(set)
    for path in paths:
        arm=path.parts[-3];seed=path.parts[-4];trace=read(path);cid=trace['case_id']
        if cid not in train: raise ValueError('Trace is not a training case: '+cid)
        rel=path.relative_to(run).as_posix()
        result['evidence_sha256'][rel]=hashlib.sha256(path.read_bytes()).hexdigest()
        c=counts[arm];c['groups']+=1;c['verified_repairs_recorded']+=trace['verified_repairs']
        c['reward_updates']+=int(trace['reward_update']);c['guard_groups']+=int(trace['guard'])
        distinct[arm].add(cid)
        for sample_index,s in enumerate(trace['samples']):
            c['initial:'+s['semantic']['status']]+=1
            rounds=s.get('repair_trace',[])
            c['candidates_with_evidence']+=int(any(a['cards'] for a in rounds))
            for a in rounds:
                status=a.get('semantic',{}).get('status',a['status'])
                c['repair:'+status]+=1;by_round[arm][str(a['round'])+':'+status]+=1
                if a['status']=='generated':
                    c['generated']+=1
                    c['at_4096_token_limit']+=int(a.get('tokens',0)>=4096)
                    c['same_raw_reply_as_initial']+=int(a.get('reply','').strip()==s['text'].strip())
                sem=a.get('semantic',{});proof=sem.get('proof',sem.get('reason','unspecified'))
                key=(arm,status,proof)
                if picked[key]>=examples_per_kind: continue
                picked[key]+=1
                result['examples'].append(dict(arm=arm,seed=seed,case_id=cid,trace=rel,
                    sample_index=sample_index,selected_round=a['round'],failure_kind=[status,proof],
                    training_input=train[cid],initial={k:s.get(k) for k in ('text','model','error','semantic','reward')},
                    repair_trace=rounds))
    for arm,c in counts.items():
        result['arms'][arm]=dict(counts=dict(c),by_round=dict(by_round[arm]),distinct_train_cases=len(distinct[arm]))
    result['interpretation_limits']=[
        'Repair attempts may repeat the same training case; do not treat them as independent trials.',
        'At 4096 tokens is a truncation indicator, not proof; generation can also hit time limits.',
        'This collector does not rerun the verifier or infer failure causes beyond recorded fields.',
        'Historical P-VCD is fully trained; new pilot arms used 12 training steps. Not equal training budgets.']
    return result

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--mode',choices=['pilot','formal'],default='pilot')
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();result=collect(a.run,a.mode)
    # Exclusive output creation prevents overwriting any existing evidence or checkpoints.
    with a.out.open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2)
    print(json.dumps(result['arms'],ensure_ascii=False,indent=2))
    print('DIAGNOSTIC_SAVED',a.out)

if __name__=='__main__':main()
