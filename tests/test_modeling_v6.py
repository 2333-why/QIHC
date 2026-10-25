import argparse
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.benchmarks import comparison as c
from tools.runtime.modeling_v6 import builder as b, experiment as e, formatting as f
from tools.runtime.modeling_v6 import workflow as w, generation as g, report, statistics, eta
from tools.runtime.modeling_v6.protocol import compile_math


QUESTION='Values [2,3,4]. Choose at most two binary variables.'


def req(rid,kind,quote=QUESTION,count=1):
    return {'id':rid,'kind':kind,'quote':quote,'expected_count':count,'explanation':'test interpretation'}


def plan():
    return {'variables':{'count':3,'prefix':'x','lower':0,'upper':1},
            'requirements':[req('value','objective'),req('domain','domain',count=3),req('cap','constraint')]}


def rule(rid='cap',cid='capacity'):
    return {'requirement_id':rid,'constraints':[{'id':cid,'a':[1,1,1],'op':'le','rhs':2}],'logic':[]}


def state():
    s=b.validate_plan(plan(),QUESTION,['x0','x1','x2'])
    s=b.objective_step(s,{'objective':{'sense':'max','linear':{'source':'d0'},'quadratic':[]}},QUESTION)
    return b.rules_step(s,{'items':[rule()]},['cap'],QUESTION)


def fake_generator():
    class OOM(Exception):pass
    return SimpleNamespace(torch=SimpleNamespace(cuda=SimpleNamespace(OutOfMemoryError=OOM,empty_cache=lambda:None)))


def mock_generate(generator,items,**kwargs):
    rows=[]
    for _,task in items:
        kind=task['_kind']
        if kind=='flat':obj=b.assemble(state(),QUESTION)
        elif kind=='plan':obj=plan()
        elif kind=='objective':obj={'objective':state()['objective']}
        elif kind=='rules':obj={'items':[rule()]}
        else:obj={'notes':[],'rule_replacements':[]}
        rows.append({'status':'generated','response':b.dump(obj),'input_tokens':100,'output_tokens':50,
                     'finish_reason':'json_complete','token_budget':task['_max_new_tokens']})
    return rows


def request(tmp_path,frontend='staged',phase='model_staged',previous=None):
    data={'id':'fixture','case_id':'tiny','question':QUESTION,'phase':phase,'frontend':frontend,
          'prompt':b.common(QUESTION,['x0','x1','x2']),'variable_names':['x0','x1','x2']}
    if previous is not None:data['previous_state']=previous
    path=tmp_path/'requests'/('fixture-'+phase+'.json');c.save(path,data)
    return path,data


def test_incremental_assembly_and_source_reference():
    s=state();obj=b.assemble(s,QUESTION)
    model=compile_math(b.dump(obj),QUESTION)[0]
    assert model['objective']['linear']==[2,3,4]
    assert model['constraints'][0]['a']==[1,1,1]
    assert obj['model']['objective']['linear']=={'source':'d0'}
    assert obj['requirements'][-1]['targets']==['constraint:capacity']
    assert 'gold' not in b.common(QUESTION)


def test_local_patch_preserves_other_components_and_is_atomic():
    s=state();original=copy.deepcopy(s)
    patch={'notes':[],'rule_replacements':[dict(rule(),constraints=[{'id':'capacity','a':[1,1,1],'op':'le','rhs':1}])]}
    updated=b.apply_patch(s,patch,QUESTION)
    assert s==original and updated['objective']==s['objective'] and updated['variables']==s['variables']
    assert compile_math(b.finish(updated,QUESTION),QUESTION)[0]['constraints'][0]['rhs']==1
    patch['rule_replacements'][0]['constraints'][0]['a']=[1,1]
    with pytest.raises(ValueError):b.apply_patch(s,patch,QUESTION)
    assert s==original


def test_fix_domain_misread_as_fixed_value_without_erasing_obligation():
    s=state();s['rules'][0]['constraints']=[{'id':'wrong_fixed','a':[1,0,0],'op':'eq','rhs':1}]
    patch={'notes':[{'target':'cap','quote':QUESTION,'reason':'This source clause specifies domain membership, not a fixed value.'}],
           'requirement_updates':[req('cap','domain',count=3)]}
    revised=b.apply_patch(s,patch,QUESTION)
    assert not revised['rules'] and {r['id'] for r in revised['requirements']}=={'value','domain','cap'}
    assert compile_math(b.finish(revised,QUESTION),QUESTION)[0]['constraints']==[]
    patch['notes']=[]
    with pytest.raises(ValueError,match='targeted'):b.apply_patch(s,patch,QUESTION)


@pytest.mark.parametrize('patch',[
    {'notes':[],'rule_replacements':[{'requirement_id':'cap','constraints':[],'logic':[]}]},
    {'notes':[],'rule_replacements':[rule('unknown')]},
    {'notes':[],'rule_replacements':[rule(),rule()]},
    {'notes':[],'variables':{'count':2,'prefix':'z','lower':0,'upper':1}},
    {'notes':[],'delete_requirements':['cap']},
    {'notes':[{'target':'cap'}]},
])
def test_bad_patch_rejected_without_state_mutation(patch):
    s=state();original=copy.deepcopy(s)
    with pytest.raises((ValueError,TypeError)):b.apply_patch(s,patch,QUESTION)
    assert s==original


def test_add_missing_obligation_and_global_duplicate_check():
    s=state();patch={'notes':[],'new_requirements':[{'requirement':req('new','constraint'), 'rule':rule('new','new_constraint')}]}
    revised=b.apply_patch(s,patch,QUESTION)
    assert len(compile_math(b.finish(revised,QUESTION),QUESTION)[0]['constraints'])==2
    patch['new_requirements'][0]['rule']['constraints'][0]['id']='capacity'
    with pytest.raises(ValueError):b.apply_patch(s,patch,QUESTION)


def test_four_bounded_chunks_and_required_coverage():
    p=plan();p['requirements']=p['requirements'][:2]+[req('r'+str(i),'constraint') for i in range(33)]
    s=b.validate_plan(p,QUESTION,['x0','x1','x2'])
    assert len(b.chunks(s))==4 and sum(map(len,b.chunks(s)))==33
    with pytest.raises(ValueError,match='Unmodeled'):b.finish(s,QUESTION)
    with pytest.raises(ValueError,match='EACH'):b.rules_step(s,{'items':[]},['r0'],QUESTION)


def test_stage_grammar_and_actual_failure_samples():
    examples={'flat':b.assemble(state(),QUESTION),'plan':plan(),'objective':{'objective':state()['objective']},
              'rules':{'items':[rule()]},'patch':{'notes':[],'rule_replacements':[]}}
    def accepted(text,kind):
        parser=f.parser(kind)
        for ch in text:
            if ch not in parser.get_allowed_characters():return False
            parser=parser.add_character(ch)
        return parser.can_end()
    for kind,obj in examples.items():
        assert accepted(b.dump(obj),kind)
        assert accepted('{"unsupported":"outside scope"}',kind)
        assert not accepted('{"unexpected":1}',kind)
    fixture=json.loads((Path(__file__).parent/'fixtures/modeling_v5_failures.json').read_text())
    bad=[r['response'] for case in fixture['cases'] for r in case['responses'].values() if r['status']=='incomplete_output']
    assert len(bad)==5 and not any(accepted(raw,'flat') for raw in bad)


def test_real_numeric_copy_failure_fixed_by_local_source_binding():
    from tools.runtime.modeling_v5.sources import catalog
    cases=json.loads((Path(__file__).parent/'fixtures/modeling_v5_failures.json').read_text())['cases']
    case=next(c for c in cases if c['id'].startswith('knapsack-64'))
    model=json.loads(case['responses']['model']['response'])['model']
    original={'variables':model['variables'],'objective':model['objective'],
              'requirements':[req('o','objective',case['question']),req('d','domain',case['question'],64),req('r','constraint',case['question'],len(model['constraints']))],
              'rules':[{'requirement_id':'r','constraints':model['constraints'],'logic':model.get('logic',[])}]}
    bad=compile_math(b.finish(original,case['question']),case['question'])[0]
    weights=case['gold']['constraints'][0]['a']  # Test oracle only; production builder has no gold.
    source=next(r['id'] for r in catalog(case['question']) if r['value']==weights)
    replacement=copy.deepcopy(original['rules'][0]);replacement['constraints'][0]['a']={'source':source}
    fixed=b.apply_patch(original,{'notes':[],'rule_replacements':[replacement]},case['question'])
    good=compile_math(b.finish(fixed,case['question']),case['question'])[0]
    assert bad['constraints'][0]['a']!=weights and good['constraints'][0]['a']==weights
    assert fixed['objective']==original['objective']


def test_batched_workflow_and_step_resume(tmp_path,monkeypatch):
    calls=[]
    def generate(gen,items,**kwargs):
        calls.append([t['_kind'] for _,t in items]);return mock_generate(gen,items)
    monkeypatch.setattr(g,'generate',generate)
    first=request(tmp_path,phase='model_staged');second=request(tmp_path,phase='other_staged')
    w.execute(fake_generator(),[first,second],tmp_path,0,'batch')
    assert calls==[['plan','plan'],['objective','objective'],['rules','rules']]
    result=c.load(tmp_path/'responses'/first[0].name)
    assert result['generation_attempts']==3 and result['output_tokens']==150 and result['builder_state']
    before=copy.deepcopy(result)
    # Replaying the same request must perform zero generation calls.
    calls.clear();w.execute(fake_generator(),[first],tmp_path,0,'resumed')
    assert not calls and c.load(tmp_path/'responses'/first[0].name)==before


def test_failed_step_targeted_repair_cost_and_shared_cap(tmp_path,monkeypatch):
    prompts=[]
    def generate(gen,items,**kwargs):
        prompts.extend(t['prompt'] for _,t in items)
        out=mock_generate(gen,items)
        if len(prompts)==2:out[0]['response']='{"objective":{"sense":"max","linear":{"source":"absent"},"quadratic":[]}}'
        return out
    monkeypatch.setattr(g,'generate',generate)
    path,data=request(tmp_path)
    w.execute(fake_generator(),[(path,data)],tmp_path,0,'fixture')
    row=c.load(tmp_path/'responses'/path.name)
    assert row['status']=='generated' and row['local_repairs']==1 and row['output_tokens']==200
    assert 'unknown source' in prompts[2] and prompts[1]!=prompts[2]
    assert row['output_tokens']<=row['total_output_cap']==b.TOTAL_OUTPUT_TOKENS


def test_permanent_failure_not_published_as_valid_partial_model(tmp_path,monkeypatch):
    def generate(gen,items,**kwargs):
        return [{'status':'generated','response':'{"variables":{},"requirements":[]}','output_tokens':10} for _ in items]
    monkeypatch.setattr(g,'generate',generate)
    path,data=request(tmp_path);w.execute(fake_generator(),[(path,data)],tmp_path,0,'bad')
    row=c.load(tmp_path/'responses'/path.name)
    assert row['mathematical_validity']=='invalid' and row['generation_attempts']==2 and row['output_tokens']==20
    with pytest.raises(ValueError):compile_math(row['response'],QUESTION)


def test_budget_exhaustion_never_gets_additional_tokens():
    job=w.Job(Path('unused'),{'frontend':'staged','question':QUESTION,'prompt':'test'})
    job.records=[{'output_tokens':b.TOTAL_OUTPUT_TOKENS-100}]
    assert job.task() is None and job.done and job.result['status']=='invalid_model_output'


def test_checkpoint_tamper_refused(tmp_path,monkeypatch):
    monkeypatch.setattr(g,'generate',mock_generate)
    path,data=request(tmp_path);w.execute(fake_generator(),[(path,data)],tmp_path,0,'first')
    bad=copy.deepcopy(data);bad['prompt']+=' changed';c.save(path,bad)
    with pytest.raises(ValueError,match='checkpoint input changed'):w.execute(fake_generator(),[(path,bad)],tmp_path,0,'second')


def test_interrupted_after_plan_resumes_without_repeating_plan(tmp_path,monkeypatch):
    gen=fake_generator();calls=[]
    def fail_once(generator,items,**kwargs):
        calls.append(items[0][1]['_kind'])
        if calls[-1]=='objective':raise gen.torch.cuda.OutOfMemoryError('fixture OOM')
        return mock_generate(generator,items)
    monkeypatch.setattr(g,'generate',fail_once)
    path,data=request(tmp_path)
    with pytest.raises(gen.torch.cuda.OutOfMemoryError):w.execute(gen,[(path,data)],tmp_path,0,'interrupted')
    assert not (tmp_path/'responses'/path.name).exists()
    assert len(list((tmp_path/'builder_steps'/path.stem).glob('*.json')))==1
    assert any(c.load(p)['event']=='unpublished_step_oom' for p in (tmp_path/'inference_events').glob('*.json'))
    calls.clear()
    def resume(generator,items,**kwargs):
        calls.extend(t['_kind'] for _,t in items);return mock_generate(generator,items)
    monkeypatch.setattr(g,'generate',resume)
    w.execute(gen,[(path,data)],tmp_path,0,'resumed')
    assert calls==['objective','rules']
    assert c.load(tmp_path/'responses'/path.name)['generation_attempts']==3


def test_step_output_tamper_refused(tmp_path,monkeypatch):
    monkeypatch.setattr(g,'generate',mock_generate)
    path,data=request(tmp_path);w.execute(fake_generator(),[(path,data)],tmp_path,0,'first')
    checkpoint=tmp_path/'builder_steps'/path.stem/'00.json'
    record=c.load(checkpoint);record['generation']['output_tokens']+=1;c.save(checkpoint,record)
    with pytest.raises(ValueError,match='output integrity'):w.execute(fake_generator(),[(path,data)],tmp_path,0,'second')


def test_local_review_is_one_patch_and_flat_is_one_full_response(tmp_path,monkeypatch):
    calls=[]
    def generate(gen,items,**kwargs):
        calls.extend(t['_kind'] for _,t in items);return mock_generate(gen,items)
    monkeypatch.setattr(g,'generate',generate)
    a=request(tmp_path,phase='review_staged',previous=state())
    z=request(tmp_path,frontend='flat',phase='review_flat')
    w.execute(fake_generator(),[a,z],tmp_path,0,'mixed')
    assert set(calls)=={'patch','flat'} and len(calls)==2
    assert c.load(tmp_path/'responses'/a[0].name)['builder_state']['rules']==state()['rules']


def test_batch_oom_splits_real_work_and_retains_results(tmp_path,monkeypatch):
    gen=fake_generator()
    def generate(generator,items,**kwargs):
        if len(items)>1:raise gen.torch.cuda.OutOfMemoryError('batch too big')
        return mock_generate(generator,items)
    monkeypatch.setattr(g,'generate',generate)
    a=request(tmp_path,frontend='flat',phase='first');z=request(tmp_path,frontend='flat',phase='second')
    w.execute(gen,[a,z],tmp_path,0,'split')
    assert all(c.load(tmp_path/'responses'/p.name)['status']=='generated' for p,_ in (a,z))
    assert len(list((tmp_path/'inference_events').glob('*.json')))==1


def test_queue_packs_compatible_schemas(tmp_path):
    from tools.runtime.modeling_v6.inference import pending
    # Production queue names are hashes; use the same identity function here.
    for i,frontend in enumerate(['staged','flat','staged']):
        value={'case_id':str(i),'phase':'model_'+frontend,'frontend':frontend}
        c.save(tmp_path/'requests'/(c.key(str(i),value['phase'])+'.json'),value)
    picked=pending(tmp_path,0,1,2)
    assert len(picked)==2 and all(r['frontend']=='staged' for _,r in picked)


def test_gate_requires_both_frontends_and_actual_search(tmp_path):
    from tools.runtime.modeling_v6.health import gate_check
    arms=['flat_selected','staged_selected','staged_full_review']
    case={'id':'tiny'};c.save(tmp_path/'manifest.json',{'settings':{'gate':True,'arms':arms}})
    c.save(tmp_path/'cases.CPU_ONLY.json',[case])
    import hashlib
    for arm in arms:
        for phase,stage in [(e.initial_phase(arm),'initial'),('review_'+arm,'revised')]:
            path=tmp_path/'requests'/(c.key('tiny',phase)+'.json');c.save(path,{'fixture':True})
            c.save(tmp_path/'responses'/path.name,{'execution':'audited_modeling_v6','status':'generated',
                'format_enforcement':'lm-format-enforcer-0.11.3-json-schema','request_sha256':e.digest(path),'response':'fixture'})
            c.save(tmp_path/'progress'/c.key('tiny','case')/f'{arm}.{stage}.0.json',
                   {'input_response_sha256':hashlib.sha256(b'fixture').hexdigest(),'status':'searched',
                    'mathematical_validity':'valid','runs':[{'candidate_evaluations':100,'verified_feasible':False}]})
    checks=gate_check(tmp_path)
    assert checks['passed'] and len(checks['checks'])==6  # Only tests gate logic, not real inference.
    path=tmp_path/'progress'/c.key('tiny','case')/'staged_full_review.revised.0.json'
    record=c.load(path);record['runs'][0]['candidate_evaluations']=0;c.save(path,record)
    assert not gate_check(tmp_path)['passed']


def setup_experiment(tmp_path):
    source,run,weights=tmp_path/'source',tmp_path/'run',tmp_path/'weights'
    case={'id':'tiny','dataset':'fixture','question':QUESTION,'gold':compile_math(b.finish(state(),QUESTION),QUESTION)[0],
          'reference_answer':7,'reference_status':'fixture'}
    c.save(weights/'config.json',{'fixture':True});c.save(source/'cases.CPU_ONLY.json',[case])
    c.save(source/'manifest.json',{'cases_sha256':e.digest(source/'cases.CPU_ONLY.json'),
          'settings':{'model':str(weights),'model_files':e.model_identity(weights)}})
    args=argparse.Namespace(run=run,from_run=source,gate=False,budget=256,chains=2,seeds=1,solver_seconds=.1,limit=0,arms=list(e.ARMS))
    e.prepare(args)
    return case,args


def test_all_arms_real_cpu_search_reports_resume_and_gold_isolation(tmp_path,monkeypatch):
    from tools.runtime.modeling_v6 import run as runner
    monkeypatch.setattr(g,'generate',mock_generate)
    case,args=setup_experiment(tmp_path);run=args.run;cfg=e.validate(run)['settings']
    original=copy.deepcopy(case);changed=copy.deepcopy(case)
    changed['gold']['constraints']=[];changed['reference_answer']='SECRET'
    for phase in e.initial_phases(e.ARMS):assert e.initial_prompt(original,phase)==e.initial_prompt(changed,phase)
    for step in range(100):
        items=[(p,c.load(p)) for p in (run/'requests').glob('*.json') if not (run/'responses'/p.name).exists()]
        if items:
            assert all('SECRET' not in b.dump(r) and 'gold' not in r for _,r in items)
            w.execute(fake_generator(),items,run,0,f'test-{step}')
        if runner.process_case(run,case,cfg)=='complete':break
    result=e.summarize(run,[case],cfg)
    assert result['status']=='complete' and len(result['datasets']['fixture']['arms'])==11
    snapshot=report.load_snapshot(run)
    assert len(snapshot['responses'])==13 and len(snapshot['complete_stages'])==22
    stats=statistics.compare(snapshot,bootstrap=200)
    assert any(r['comparison']=='local_patch_vs_full_rewrite' for r in stats['comparisons'])
    estimation=eta.estimate(snapshot,now=10**10)
    assert estimation['state']=='complete' and estimation['gpu']['requests_expected']==13
    report.generate(run,tmp_path/'reports',bootstrap=200)
    assert (tmp_path/'reports/report.md').exists()
    e.prepare(args);assert runner.process_case(run,case,cfg)=='complete'
    assert all(a['attributed_input_tokens']>0 for a in result['datasets']['fixture']['arms'].values())
    assert (args.from_run/'cases.CPU_ONLY.json').read_bytes()==(run/'cases.CPU_ONLY.json').read_bytes()
