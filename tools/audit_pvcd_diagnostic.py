"""Replay the returned representative chains; deduplicate selected-round examples."""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from qihc.modeling.active_protocol_v2 import parse_model
from qihc.generic.feedback.verification import equivalent
from experiments.pvcd_verified_v1.feedback import remaining

def audit(path):
    data=json.loads(Path(path).read_text(encoding='utf-8-sig'));chains={}
    for e in data['examples']:chains.setdefault((e['arm'],e['trace'],e['sample_index']),e)
    rows=[]
    for e in chains.values():
        gold=e['training_input']['gold'];case=e['training_input']['case'];current=e['initial']['model']
        for a in e['repair_trace']:
            if a['status']!='generated':continue
            try:model=parse_model(a['reply'],case);error=None
            except (ValueError,KeyError,TypeError,IndexError,OverflowError) as exc:model=None;error=str(exc)
            sem=equivalent(model,gold)
            rows.append(dict(arm=e['arm'],case_id=e['case_id'],trace=e['trace'],sample=e['sample_index'],round=a['round'],
                logged_status=a['semantic']['status'],replayed_status=sem['status'],error=error,
                cards=len(a['cards']),remaining=remaining(model,gold,a['cards']),
                identical_model=(model==current),proof=sem.get('proof'),tokens=a.get('tokens')))
            if model is not None:current=model
    return dict(source_sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(),
        run_sha=data['manifest.json']['code_sha'],selected_examples=len(data['examples']),
        unique_chains=len(chains),unique_cases=len({e['case_id'] for e in chains.values()}),
        replayed_rounds=len(rows),statuses=dict(Counter(r['replayed_status'] for r in rows)),
        all_statuses_match=all(r['logged_status']==r['replayed_status'] for r in rows),
        valid_rounds_with_remaining_witness=sum(r['remaining'] is not None and r['remaining']>0 for r in rows),
        rows=rows,scope='Selected representative chains, not all 94 repair attempts; repeated examples deduplicated.')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('input',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
    result=audit(a.input);a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='rows'},ensure_ascii=False,indent=2))
