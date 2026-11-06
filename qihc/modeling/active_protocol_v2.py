"""E004_2.2 format contract; historical active_protocol stays unchanged."""
import copy
import json
from . import active_protocol as legacy
from .protocol import load_object

compact=legacy.compact


def normalize_model(model):
    model=copy.deepcopy(model)
    if not isinstance(model,dict): return model
    if isinstance(model.get('objective'),dict): model['objective'].setdefault('quadratic',[])
    if model.get('logic')=={}: model['logic']=[]
    return model


FORMAT_NOTE='''
FINAL FORMAT CHECK: objective MUST include "quadratic":[] when no quadratic terms.
logic MUST be [] when no logic constraints, never {}. Do not omit either field.
Every JSON key and constraint id must be unique. Sum repeated coefficients before emitting a sparse dictionary.
Use numeric decision indices 0..N-1, NOT original entity labels. Never invent a mapping or use ellipses.
The parser interprets omitted quadratic as zero terms and empty logic {} as []; this does not certify semantics.
'''


def model_prompt(case): return legacy.model_prompt(case)+FORMAT_NOTE


def source_prompt(case):
    return legacy.source_prompt(case)+'''
Emit ONLY {"checks":[...]}, no other top-level keys. Select at most 16 representative local checks;
do NOT enumerate every edge or every rule. Prefer fewer precise checks over an exhaustive list.
For a linear objective include "quadratic":[]. If unsure return {"checks":[]}.
'''


def patch_prompt(case,model,checks,evidence,error=None):
    return legacy.patch_prompt(case,model,checks,evidence,error)+FORMAT_NOTE


def parse_model(text,case,old=None):
    obj=load_object(text)  # Preserve rejection of duplicate keys and malformed JSON.
    if 'model' in obj: obj['model']=normalize_model(obj['model'])
    if set(obj)=={'edits'} and isinstance(obj['edits'],list):
        for edit in obj['edits']:
            if not isinstance(edit,dict): raise ValueError('Patch entries must be objects')
            if edit.get('target')=='objective' and isinstance(edit.get('value'),dict): edit['value'].setdefault('quadratic',[])
            if edit.get('target')=='logic' and edit.get('value')=={}:edit['value']=[]
    return legacy.parse_model(json.dumps(obj),case,old)


def parse_checks(text,case):
    obj=load_object(text)
    if set(obj)!={'checks'} or not isinstance(obj['checks'],list): raise ValueError('Expected checks list and no other envelope keys')
    checks=obj['checks']; selected=copy.deepcopy(checks[:16]); malformed=[]; clean=[]
    for index,row in enumerate(selected):
        if not isinstance(row,dict):malformed.append(f'check {index}: expected object');continue
        if row.get('kind')=='objective':row.setdefault('quadratic',[])
        clean.append(row)
    result=legacy.parse_checks(json.dumps({'checks':clean}),case)
    result['rejected']=malformed+result['rejected']
    result.update(emitted_count=len(checks),budget_dropped=max(0,len(checks)-16),selection='first_16_in_source_output_order_no_gold')
    return result


def guard(case,previous,text):
    try:
        candidate=parse_model(text,case,previous)
        if case.get('public_domains') is not None and [(v['lower'],v['upper']) for v in candidate['variables']]!=[tuple(x) for x in case['public_domains']]:
            raise ValueError('Candidate contradicts public domains')
        return dict(status='accepted',model=candidate,candidate=candidate)
    except (ValueError,KeyError,TypeError) as exc:
        return dict(status='rejected',model=previous,candidate=None,error=str(exc))
