import copy
import argparse
import itertools
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
from tools.runtime.modeling_v5 import protocol as p, generation as g, formatting as f
from tools.runtime.modeling_v5.sources import catalog


@pytest.fixture(scope='module')
def failures():
    return json.loads((Path(__file__).parent/'fixtures/modeling_v5_failures.json').read_text())['cases']


def accepts(text):
    parser = f.parser()
    for ch in text:
        if ch not in parser.get_allowed_characters():
            return False
        parser = parser.add_character(ch)
    return parser.can_end()


def envelope(n=3):
    return {'model':{'schema':'bounded-integer-v2',
        'variables':{'count':n,'prefix':'x','lower':0,'upper':1},
        'objective':{'sense':'max','linear':[1]*n,'quadratic':[]},'constraints':[],'logic':[]}}


def test_grammar_accepts_long_vectors_and_references():
    obj=envelope(64)
    assert accepts(json.dumps(obj))  # Must not inherit the package default max array length 20.
    obj['model']['objective']['linear']={'source':'d0'}
    obj['model']['constraints']=[{'forall_rows':{'source':'d1','id_prefix':'c','terms':[{'index':{'column':'u'},'coefficient':1}],'op':'ge','rhs':1}}]
    assert accepts(json.dumps(obj))
    assert accepts('{"unsupported":"requires continuous variables"}')
    assert not accepts('{"model":{}}')


def test_actual_bad_json_rejected_by_character_grammar(failures):
    rejected=0
    for case in failures:
        for response in case['responses'].values():
            if response['status']=='incomplete_output':
                assert not accepts(response['response']),case['id']
                row=g.validate_output(copy.deepcopy(response),case['question'])
                assert row['status']=='invalid_json_output'
                assert row['syntax_location']['offset']>=0
                rejected+=1
    assert rejected==5


def test_eos_syntax_error_uses_error_prompt_not_identical_retry(tmp_path,monkeypatch,failures):
    case=next(c for c in failures if c['id'].startswith('balanced_cut'))
    raw=case['responses']['model']['response']
    from tools.benchmarks import comparison as c
    class OOM(Exception):pass
    gen=SimpleNamespace(max_input_tokens=32768,
        _inputs=lambda x:{'input_ids':SimpleNamespace(shape=(1,len(x)))},
        torch=SimpleNamespace(cuda=SimpleNamespace(OutOfMemoryError=OOM,empty_cache=lambda:None)))
    request={'prompt':'ORIGINAL PROBLEM','question':case['question']}
    path=tmp_path/'requests'/'one.json'; c.save(path,request)
    prompts=[]
    def generate(generator,items,retry=False):
        prompts.append(items[0][1]['prompt'])
        return [{'response':raw,'status':'incomplete_output','finish_reason':'eos_without_complete_json','output_tokens':3679}]
    monkeypatch.setattr(g,'generate',generate)
    g.execute(gen,[(path,request)],tmp_path,0,'fixture')
    row=c.load(tmp_path/'responses'/'one.json')
    assert len(prompts)==2 and prompts[0]!=prompts[1]
    assert 'syntax location' in prompts[1] and '6245' in prompts[1]
    assert row['status']=='invalid_json_output' and row['output_tokens']==7358


def test_catalog_provenance_and_no_gold_access(failures):
    for case in failures:
        for row in catalog(case['question']):
            literal=case['question'][row['start']:row['end']]
            value=json.loads(literal)
            if row['path']:
                for part in row['path'].split('/')[1:]:
                    part=part.replace('~1','/').replace('~0','~')
                    value=value[int(part)] if isinstance(value,list) else value[part]
            assert value==row['value']
        modified=copy.deepcopy(case)
        modified['gold']['objective']['linear']=['SECRET_GOLD']
        modified['gold']['constraints']=[]
        assert p.modeling_prompt(case)==p.modeling_prompt(modified)


def test_real_knapsack_copy_error_diagnosed_and_reference_exact(failures):
    case=next(c for c in failures if c['id'].startswith('knapsack-64'))
    raw=case['responses']['model']['response']
    bad,_,audits=p.inspect(raw,case['question'])
    issues=audits['semantic_diagnostics']['suspected_numeric_copy_errors']
    assert any(x['target'].startswith('constraint:') and x['different_entries']>0 for x in issues)
    obj=json.loads(raw)
    weights=case['gold']['constraints'][0]['a']
    ref=next(r for r in catalog(case['question']) if r['value']==weights)
    obj['model']['constraints'][0]['a']={'source':ref['id']}
    fixed,_,audits=p.inspect(json.dumps(obj),case['question'])
    assert fixed['constraints'][0]['a']==weights
    assert bad['constraints'][0]['a']!=weights  # Never silently corrected legacy output.
    assert audits['semantic_diagnostics']['source_bindings']
    with pytest.raises(ValueError,match='unknown source'):
        p.compile_math(json.dumps(obj))


def test_generic_weighted_table_polynomial_and_constraints():
    question='Pairs: [{"a":0,"b":1,"weight":3},{"a":1,"b":2,"weight":2}]. Limits [1,2,3].'
    source=next(r['id'] for r in catalog(question) if r['kind']=='table')
    obj=envelope(); model=obj['model']
    model['objective']['linear']={'sum_rows':{'source':source,'terms':[
        {'index':{'column':'a'},'coefficient':{'column':'weight'}},
        {'index':{'column':'b'},'coefficient':{'column':'weight'}}]}}
    model['objective']['quadratic']={'sum_rows':{'source':source,'terms':[{'i':{'column':'a'},'j':{'column':'b'},'c':-2}]}}
    model['constraints']=[{'forall_rows':{'source':source,'id_prefix':'pair','terms':[
        {'index':{'column':'a'},'coefficient':1},{'index':{'column':'b'},'coefficient':1}],
        'op':'ge','rhs':1}}]
    compiled=p.compile_math(json.dumps(obj),question)[0]
    assert compiled['objective']['linear']==[3,5,2]
    assert compiled['objective']['quadratic']==[[0,1,-2],[1,2,-2]]
    assert [x['a'] for x in compiled['constraints']]==[[1,1,0],[0,1,1]]
    for x in itertools.product((0,1),repeat=3):
        value=sum(a*b for a,b in zip(compiled['objective']['linear'],x))+sum(c*x[i]*x[j] for i,j,c in compiled['objective']['quadratic'])
        assert value==3*x[0]+5*x[1]+2*x[2]-2*x[0]*x[1]-2*x[1]*x[2]


def test_quadratic_dict_normalization_preserves_even_wrong_meaning(failures):
    case=next(c for c in failures if c['id']=='nlco_MAXCUT_L_017')
    raw=case['responses']['review_pbit_selected']['response']
    model,_,audits=p.inspect(raw,case['question'])
    assert all(c>=0 for i,j,c in model['objective']['quadratic'])  # Do not fix signs using task type.
    assert audits['semantic_diagnostics']['single_variable_equalities_total']==25
    assert audits['semantic_diagnostics']['objective_local_probes']
    obj=envelope(2)
    obj['model']['objective']['quadratic']=[{'i':1,'j':0,'c':3},[0,1,-1],[1,1,4]]
    model=p.compile_math(json.dumps(obj))[0]
    assert model['objective']['quadratic']==[[0,1,2],[1,1,4]]


@pytest.mark.parametrize('value',[{'source':'absent'},{'source':'d0','unknown':1},{'source':'d0','scale':True}])
def test_bad_references_fail_closed(value):
    obj=envelope();obj['model']['objective']['linear']=value
    with pytest.raises(ValueError):p.compile_math(json.dumps(obj),'[2,3,4]')


def test_forall_dimension_column_and_duplicates_rejected():
    obj=envelope(2)
    obj['model']['constraints']=[{'forall_rows':{'source':'d0','id_prefix':'c','terms':[{'index':{'column':'missing'},'coefficient':1}],'op':'ge','rhs':1}}]
    with pytest.raises(ValueError,match='column'):p.compile_math(json.dumps(obj),'[{"u":0}]')
    obj['model']['constraints'][0]['forall_rows']['terms'][0]['index']={'column':'u'}
    with pytest.raises(ValueError,match='outside'):p.compile_math(json.dumps(obj),'[{"u":99}]')
    obj['model']['constraints']*=2
    with pytest.raises(ValueError,match='Unique'):p.compile_math(json.dumps(obj),'[{"u":0}]')


def test_flatten_matrix_and_exact_column_data():
    obj=envelope(4);obj['model']['objective']['linear']={'source':'d0','flatten':True}
    model=p.compile_math(json.dumps(obj),'[[3,4],[7,9]]')[0]
    assert model['objective']['linear']==[3,4,7,9]
    obj=envelope(2);obj['model']['objective']['linear']={'source':'d0','column':'cost','scale':-1}
    model=p.compile_math(json.dumps(obj),'[{"cost":3,"u":0},{"cost":4,"u":1}]')[0]
    assert model['objective']['linear']==[-3,-4]
    obj['model']['objective']['linear']={'sum_rows':{'source':'d0','terms':[{'index':{'column':'u'},'coefficient':{'column':'cost','scale':-2}}]}}
    model=p.compile_math(json.dumps(obj),'[{"cost":3,"u":0},{"cost":4,"u":1}]')[0]
    assert model['objective']['linear']==[-6,-8]


def test_token_filter_batched_prefixes_nested_arrays_and_eos():
    from lmformatenforcer.tokenenforcer import TokenEnforcer,TokenEnforcerTokenizerData
    data=TokenEnforcerTokenizerData([(i,chr(i),False) for i in range(32,127)],
                                   lambda ids:''.join(chr(i) for i in ids),0,False,128)
    enforcer=TokenEnforcer(data,f.parser())
    f.restore_parser_limits(enforcer)
    obj=envelope(32)
    obj['requirements']=[{'targets':['variable:x'+str(i) for i in range(32)]}]
    answers=[json.dumps(obj,separators=(',',':')),'{"unsupported":"outside scope"}']
    prefixes=[[],[]]  # Only generated suffixes, shared grammar independent of prompt.
    for step in range(max(map(len,answers))):
        for rank,answer in enumerate(answers):
            if step>=len(answer):continue
            allowed=enforcer.get_allowed_tokens(prefixes[rank]).allowed_tokens
            assert ord(answer[step]) in allowed,(rank,step,answer[max(0,step-20):step+20])
            if step==0:assert 0 not in allowed  # Cannot EOS before a complete envelope.
            prefixes[rank].append(ord(answer[step]))
    for prefix in prefixes:
        assert 0 in enforcer.get_allowed_tokens(prefix).allowed_tokens


def test_all_twelve_arms_with_source_bound_model_and_resume(tmp_path):
    from tools.runtime.modeling_v5 import experiment as e
    from tools.benchmarks import comparison as c
    question='Values [2,3,4]. Choose at most two binary variables.'
    obj=envelope();obj['model']['objective']['linear']={'source':'d0'}
    obj['model']['constraints']=[['cap',[1,1,1],'le',2]]
    case={'id':'source-fixture','dataset':'fixture','question':question,
          'gold':p.compile_math(json.dumps(obj),question)[0],'reference_answer':7,'reference_status':'fixture'}
    source,run,weights=tmp_path/'source',tmp_path/'run',tmp_path/'weights'
    c.save(weights/'config.json',{'fixture':True})
    c.save(source/'cases.CPU_ONLY.json',[case])
    c.save(source/'manifest.json',{'cases_sha256':e.digest(source/'cases.CPU_ONLY.json'),
        'settings':{'model':str(weights),'model_files':e.model_identity(weights)}})
    args=argparse.Namespace(run=run,from_run=source,gate=False,budget=256,chains=2,seeds=1,solver_seconds=.1,limit=0,arms=list(e.ARMS))
    e.prepare(args)
    for _ in range(80):
        for path in (run/'requests').glob('*.json'):
            if not (run/'responses'/path.name).exists():
                request=c.load(path)
                assert request['question']==question and 'SECRET' not in request['prompt']
                c.save(run/'responses'/path.name,{'response':json.dumps(obj),'execution':'MOCK',
                    'status':'generated','request_sha256':e.digest(path),'generation_attempts':1})
        if e.process_case(run,case,e.validate(run)['settings'])=='complete':break
    report=e.summarize(run,[case],e.validate(run)['settings'])
    assert report['status']=='complete' and len(report['datasets']['fixture']['arms'])==12
    assert all(a['source_bound_stage_models']==2 for a in report['model_audits_completed_cases_only'].values())
    e.prepare(args)
    assert e.process_case(run,case,e.validate(run)['settings'])=='complete'
