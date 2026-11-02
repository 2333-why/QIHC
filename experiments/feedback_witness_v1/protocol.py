"""Source-only executable hypotheses and atomic objective/constraint-family edits."""
import copy
import json
import numpy as np
from qihc.modeling import typed_schema, typed_sources, typed_builder
from qihc.generic.bounded import objective, violations
from experiments.feedback_pilot_v1.adapter import bounded_schema
from experiments.feedback_study_v1.protocol import compact_model


def syntax(case, stage):
    flat = bounded_schema(typed_schema.schema('flat', case['question'], len(case['variable_names'])))
    model = flat['anyOf'][0]['properties']['model']
    text = {'type':'string','maxLength':320}
    quotes = {'type':'array','items':text,'minItems':1,'maxItems':6}
    obj = typed_schema.obj
    if stage == 'contract':
        body = obj({'model':model,'quotes':quotes})
    else:
        body = {'anyOf':[obj({'action':{'const':'retain'},'reason':text}),
                         obj({'action':{'const':'replace_families'}, 'model':model,
                              'families':{'type':'array','items':{'enum':['objective','constraints']},'minItems':1,'maxItems':2},
                              'quotes':quotes,'reason':text})]}
    body['$defs'] = flat['$defs']
    return body


def request(case, stage, cfg, model=None, contract=None, evidence=None, error=None, remaining=None):
    public = {'source_question':case['question'],'variable_names':case['variable_names'],
              'source_catalog':typed_sources.describe(case['question'])}
    guide = typed_builder.prompt(case['question'],case['variable_names'],'flat')
    # Guide contains the public source and general typed binding syntax only.
    if stage == 'contract':
        instruction = ('Independently translate the ORIGINAL SOURCE into an executable semantic hypothesis. '
                       'Return model and 1-6 exact nonempty source quotations. No existing model or solver answer is available. '
                       'Use forall_rows for universal table relations; never partially enumerate a rule family. '
                       'Be concise. This hypothesis is fallible, not ground truth. ')
    else:
        hypothesis=contract or {}
        hypothesis_summary={'quotes':hypothesis.get('quotes',[]),'status':hypothesis.get('status'),
                            'error':hypothesis.get('error'),
                            'expected_factor_count':len(hypothesis['model']['constraints'])+len(hypothesis['model']['logic']) if hypothesis.get('model') else None}
        public.update(current_model=compact_model(model) if model else None,
                      source_only_hypothesis=hypothesis_summary, evidence='__WITNESS_EVIDENCE__', previous_error=error)
        instruction = ('Repair the model using the original source and fallible evidence. Return action=retain, or '
                       'action=replace_families with a COMPLETE model and families from [objective,constraints] (at most two). '
                       'The constraints family contains ALL linear and logical requirements. Replace it atomically; '
                       'use forall_rows to cover ALL relevant rows, never a short subset. Preserve any unlisted family. '
                       'Never change variable names or domains on an executable model. An invalid initial model may be rebuilt. '
                       'Provide exact nonempty source quotations for changes. Low energy does not prove correct semantics. '
                       'The source-only hypothesis may itself be wrong; do not invent omissions to obtain feasibility. ')
    return {'stage':stage, 'case_id':case['id'], 'prompt':guide+'\nFINAL OUTPUT CONTRACT (supersedes example envelope):\n'+instruction+'\n'+json.dumps(public,separators=(',',':')),
            '_kind':'flat','_schema':syntax(case,stage),
            '_max_new_tokens':cfg['contract_output_tokens'] if stage=='contract' else cfg['patch_output_tokens'],
            'time_limit_seconds':cfg['contract_timeout_seconds'] if stage=='contract' else cfg['patch_timeout_seconds'],
            'input_remaining':remaining if remaining is not None else cfg['request_input_tokens'],
            'evidence_payload':evidence or {'scope':'source-only; no search evidence'}}


def parse_model(response, case):
    if response.get('status') != 'generated':
        raise ValueError('No complete generated object: '+str(response.get('status')))
    value=json.loads(response['response'])
    if value.get('action')=='retain':return value,None
    quotes=value.get('quotes',[])
    if not quotes or any(not isinstance(q,str) or not q.strip() or q not in case['question'] for q in quotes):
        raise ValueError('Every quote must be a nonempty exact source substring')
    model, reason=typed_sources.compile_math(json.dumps({'model':value['model']}),case['question'])
    if model is None:raise ValueError('Cannot compile: '+str(reason))
    if [v['name'] for v in model['variables']] != case['variable_names']:
        raise ValueError('Variable mapping changed')
    return value,model


def holdouts(model, seed=20261103, count=128):
    rng=np.random.default_rng(seed)
    return np.column_stack([rng.integers(v['lower'],v['upper']+1,count) for v in model['variables']])


def relation_failures(model, contract, xs):
    if model['variables'] != contract['variables']:
        raise ValueError('Hypothesis domain differs; cannot compare contexts')
    a=violations(model,xs).sum(axis=1)==0;b=violations(contract,xs).sum(axis=1)==0
    shared=xs[a & b]
    af=objective(model,shared)*(1 if model['objective']['sense']=='min' else -1)
    bf=objective(contract,shared)*(1 if contract['objective']['sense']=='min' else -1)
    return int(np.count_nonzero(a!=b)), int(np.count_nonzero(np.sign(np.diff(af))!=np.sign(np.diff(bf))))


def guard(case, old, response, contract, evidence):
    value,candidate=parse_model(response,case)
    if candidate is None:return {'status':'retained','model':old,'candidate':None}
    families=value.get('families',[])
    if not 1<=len(families)<=2 or len(set(families))!=len(families) or set(families)-{'objective','constraints'}:
        raise ValueError('Expected one or two unique complete rule families')
    if old is not None:
        if old['variables']!=candidate['variables']:raise ValueError('Executable domain cannot be pruned or remapped')
        for name in ('objective','constraints'):
            fields=[name] if name=='objective' else ['constraints','logic']
            if name not in families and any(candidate[k]!=old[k] for k in fields):
                raise ValueError('Changed undeclared family: '+name)
    # Check coverage of the independent executable hypothesis, never canonical/gold.
    # Exact normalized factor inclusion avoids accepting only the first few rows.
    if contract and candidate['variables']==contract['variables']:
        def factors(m):
            return {json.dumps({k:v for k,v in c.items() if k!='id'},sort_keys=True) for c in m['constraints']+m['logic']}
        missing=factors(contract)-factors(candidate)
        if missing:
            return {'status':'guard_rejected','reason':'Hypothesis rule-family coverage incomplete',
                    'missing_factor_count':len(missing),'candidate':candidate,'model':old}
        xs=holdouts(contract)
        cards=(evidence or {}).get('cards',[])
        points=[x for card in cards for x in card.get('states',[])]
        if points:xs=np.concatenate([xs,np.asarray(points)],axis=0)
        mismatch=relation_failures(candidate,contract,xs)
        if any(mismatch):
            return {'status':'guard_rejected','reason':'Independent contract checks conflict (not a gold judgment)',
                    'mismatch_counts':mismatch,'candidate':candidate,'model':old}
    return {'status':'accepted','model':candidate,'candidate':candidate,
            'verification_scope':'source references and finite hypothesis checks; NOT semantic proof'}


def compact_evidence(packet, tokenizer, limit):
    out=copy.deepcopy(packet)
    out['omitted_cards']=0
    size=lambda:len(tokenizer.encode(json.dumps(out,separators=(',',':')),add_special_tokens=False))
    while size()>limit and out.get('cards'):
        out['cards'].pop();out['omitted_cards']+=1
    if size()>limit:
        out={'scope':'evidence envelope exceeded budget','cards':[], 'omitted_cards':len(packet.get('cards',[]))}
    return out
