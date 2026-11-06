"""Read-only counterfactual format audit; never rewrites experiment records."""
import argparse
from collections import Counter
import copy
import json
from pathlib import Path

from qihc.modeling.protocol import load_object
from qihc.modeling.active_protocol import parse_model


def normalize(obj):
    obj=copy.deepcopy(obj); changes=[]
    model=obj.get('model')
    if isinstance(model,dict):
        objective=model.get('objective')
        if isinstance(objective,dict) and 'quadratic' not in objective:
            objective['quadratic']=[];changes.append('omitted_quadratic_as_empty')
        if model.get('logic')=={}:
            model['logic']=[];changes.append('empty_logic_object_as_list')
    return obj,changes


def audit(run,data):
    cases={c['id']:c for c in json.loads((data/'cases.public.json').read_text())}
    source_counts=Counter();errors=Counter();rows=[]
    for path in sorted((run/'initial').glob('*.json')):
        initial=json.loads(path.read_text());case=cases.get(initial['case_id'])
        if case is None: continue
        source_path=run/'requests'/(initial['source_request']+'.json')
        if source_path.exists():
            source=json.loads(source_path.read_text())
            try:
                obj=load_object(source.get('text',''));checks=obj.get('checks')
                category='over_16' if isinstance(checks,list) and len(checks)>16 else 'within_16' if isinstance(checks,list) else 'wrong_checks_type'
                if set(obj)!={'checks'}:category+=';extra_or_missing_keys'
                source_counts[category]+=1
            except ValueError:source_counts['invalid_json']+=1
        if initial.get('model') is not None:continue
        req=json.loads((run/'requests'/(initial['initial_request']+'.json')).read_text())
        row=dict(case_id=case['id'],original_error=initial.get('error'),recompiled=False)
        try:
            if req.get('status')!='complete':raise ValueError('request_not_complete')
            obj,changes=normalize(load_object(req.get('text','')))
            row['normalizations']=changes
            parse_model(json.dumps(obj),case)
            row['recompiled']=True
        except (ValueError,TypeError,KeyError) as exc:row['remaining_error']=str(exc);errors[str(exc)]+=1
        rows.append(row)
    return dict(scope='counterfactual_parser_audit_only_not_new_experiment_scores',
                failed_initials_audited=len(rows),recompiled=sum(r['recompiled'] for r in rows),
                semantic_correctness='NOT evaluated; compilable does not mean correct',
                remaining_errors=dict(errors),source_structure=dict(source_counts),cases=rows)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True)
    p.add_argument('--data',type=Path,default=Path('experiments/feedback_active_v3/assets/public_dataset'))
    a=p.parse_args();print(json.dumps(audit(a.run,a.data),ensure_ascii=False,indent=2))
