import copy
import json
from pathlib import Path
import time
from types import SimpleNamespace
import pytest
from jsonschema import Draft202012Validator
from qihc.experiment_runtime import resilient_storage as s
from experiments.feedback_witness_v1 import protocol
from experiments.feedback_witness_v1.common import HERE, ROOT, cpu_job, immutable, job_paths


def problem():
    quote='For each listed pair, at most one endpoint may be selected.'
    question=quote+' Maximize the number selected. All variables are binary. '+json.dumps({'edges':[{'u':0,'v':1},{'u':1,'v':2},{'u':0,'v':2}]})
    case={'id':'fixture','question':question,'variable_names':['x0','x1','x2']}
    model={'schema':'bounded-integer-v2','variables':[{'name':f'x{i}','lower':0,'upper':1} for i in range(3)],
           'objective':{'sense':'max','linear':[1,1,1],'quadratic':[]},
           'constraints':[{'id':f'e{k}','a':[int(i in pair) for i in range(3)],'op':'le','rhs':1}
                          for k,pair in enumerate([(0,1),(1,2),(0,2)])],'logic':[]}
    return case,model,quote


def response(model,quote,stage='patch'):
    obj={'model':model,'quotes':[quote]}
    if stage=='patch':obj.update(action='replace_families',families=['objective','constraints'],reason='Restore all source relations.')
    return {'status':'generated','response':json.dumps(obj),'input_tokens':10,'output_tokens':20,'generation_attempts':1}


def test_atomic_rule_family_and_invalid_recovery():
    case,model,quote=problem();old=copy.deepcopy(model);old['constraints']=[]
    short=copy.deepcopy(model);short['constraints']=short['constraints'][:1]
    row=protocol.guard(case,old,response(short,quote),model,{})
    assert row['status']=='guard_rejected' and row['model']==old and row['missing_factor_count']==2
    assert protocol.guard(case,old,response(model,quote),model,{})['status']=='accepted'
    assert protocol.guard(case,None,response(model,quote),model,{})['status']=='accepted'
    changed=copy.deepcopy(model);changed['variables'][0]['upper']=0
    with pytest.raises(ValueError,match='domain'):protocol.guard(case,old,response(changed,quote),model,{})
    with pytest.raises(ValueError,match='quote'):protocol.guard(case,old,response(model,'not in original source'),model,{})
    from qihc.modeling.sources import catalog
    table=next(v for v in catalog(case['question']) if isinstance(v['value'],list) and len(v['value'])==3)
    macro=copy.deepcopy(model)
    macro['constraints']=[{'forall_rows':{'source':table['id'],'id_prefix':'all',
                          'terms':[{'index':{'column':'u'},'coefficient':1},{'index':{'column':'v'},'coefficient':1}],
                          'op':'le','rhs':1}}]
    result=protocol.guard(case,old,response(macro,quote),model,{})
    assert result['status']=='accepted' and len(result['model']['constraints'])==3


def test_bounded_schema_and_whole_card_packing():
    from qihc.modeling.formatting import parser
    case,model,quote=problem()
    for stage in ['contract','patch']:
        schema=protocol.syntax(case,stage)
        obj=json.loads(response(model,quote,stage)['response']);Draft202012Validator(schema).validate(obj)
        state=parser(schema_override=schema)
        for char in json.dumps(obj):
            assert char in state.get_allowed_characters()
            state=state.add_character(char)
        assert state.can_end()
    class Tokenizer:
        def encode(self,text,**kwargs):return list(text)
    packet={'cards':[{'states':[[0,1]],'text':'x'*120},{'states':[[1,0]],'text':'y'*120}]}
    compact=protocol.compact_evidence(packet,Tokenizer(),230)
    assert len(compact['cards'])==1 and compact['cards'][0]==packet['cards'][0]
    assert len(packet['cards'])==2


def test_contract_is_blind_and_patch_does_not_receive_executable_answer():
    case,model,quote=problem();cfg=s.load(HERE/'config_8h.json')
    req=protocol.request(dict(case,gold={'SECRET':123}),'contract',cfg,model=model,contract={'SECRET':456})
    assert 'SECRET' not in req['prompt'] and 'current_model' not in req['prompt']
    req=protocol.request(case,'patch',cfg,contract={'status':'complete','model':model,'quotes':[quote]})
    assert 'expected_factor_count' in req['prompt']
    public=json.loads(req['prompt'].split('\n')[-1])
    assert 'model' not in public['source_only_hypothesis']


def test_factor_contrast_separates_objective_and_penalty():
    import numpy as np
    from experiments.feedback_witness_v1.evidence import readout,quartets
    case,model,quote=problem();model['objective']={'sense':'max','linear':[1,1,0],'quadratic':[[0,1,-2]]}
    xs=quartets(np.zeros(3,dtype=int),0,1);row=readout(model,xs,1.)
    assert row['objective_contrast']==2
    assert row['total_contrast']!=row['objective_contrast']
    assert -.5*(row['conditional_log_odds_T_0_5'][1]-row['conditional_log_odds_T_0_5'][0])==pytest.approx(row['total_contrast'])


def test_independent_semantics_distinguishes_error_unknown_and_proof():
    from experiments.feedback_witness_v1.cpu import equivalent
    case,model,quote=problem()
    same=copy.deepcopy(model);same['objective']['linear']=[2,2,2]
    assert equivalent(same,model,.1)['status']=='verified_correct'
    broken=copy.deepcopy(model);broken['constraints']=[]
    assert equivalent(broken,model,1)['status']=='disproved'
    assert equivalent(broken,model,0)['status']=='unknown'
    assert equivalent(None,model,.1)['status']=='invalid'


def test_timed_generation_adapter(monkeypatch):
    from experiments.feedback_witness_v1.gpu import infer
    from qihc.modeling import generation
    calls=[]
    class Model:
        def generate(self,**kwargs):calls.append(kwargs);return None
    class Tokenizer:
        def encode(self,text,**kwargs):return list(text)
    gen=SimpleNamespace(model=Model(),tokenizer=Tokenizer(),_inputs=lambda text:{'input_ids':SimpleNamespace(shape=(1,10))})
    def fake_generate(timed,items):
        timed.model.generate(max_time=900)
        public=json.loads(items[0][1]['prompt'])
        assert public['evidence']['cards']==[]
        return [{'status':'generated','response':'{}','input_tokens':10,'output_tokens':2}]
    monkeypatch.setattr(generation,'generate',fake_generate)
    cfg=s.load(HERE/'config_8h.json')
    req={'prompt':'{"evidence":"__WITNESS_EVIDENCE__"}','evidence_payload':{'cards':[]},
         'input_remaining':100,'time_limit_seconds':180}
    row=infer(gen,req,cfg)
    assert calls[0]['max_time']==180 and row['time_limit_seconds']==180


def fixture_source(path):
    case,model,quote=problem();cfg=s.load(HERE/'config_8h.json')
    cfg.update(case_count=2,base_seconds=.01,probe_seconds=.02,final_seconds=.01,guard_seconds=5,
               cross_solver_seconds=.02,semantic_check_seconds=.2,cpu_workers=2,chains=4,cross_backends=['pbit','scip'])
    cases=[]
    for i in range(2):
        c=dict(case,id=f'fixture-{i}',dataset='fixture',split='smoke',size=3,generation_seed=i,gold=model)
        cases.append(c)
        initial=copy.deepcopy(model) if i==0 else None
        if initial is not None:initial['constraints']=[]
        s.save(path/'initial'/(s.key(c['id'])+'.json'),{'model':initial,'error':'fixture invalid' if initial is None else None})
        s.save(path/'evaluation/references'/(s.key(c['id'])+'.json'),{'gold_sha256':s.value_digest(model),
               'result':{'feasible':True,'optimality_proven':True,'objective':1}})
    public=[{k:v for k,v in c.items() if k!='gold'} for c in cases]
    s.save(path/'cases.public.json',public);s.save(path/'evaluation/cases.CPU_ONLY.json',cases)
    s.save(path/'evaluation/e003_baselines.json',[])
    s.save(path/'manifest.json',{'experiment_id':'E004-1.2','public_digest':s.digest(path/'cases.public.json'),
         'evaluation_digest':s.digest(path/'evaluation/cases.CPU_ONLY.json'),'baselines_digest':s.digest(path/'evaluation/e003_baselines.json')})
    return cfg,model,quote


def test_two_round_six_arm_pipeline_resume_and_full_denominators(tmp_path):
    from experiments.feedback_witness_v1.prepare import prepare
    from experiments.feedback_witness_v1.runtime import run_pipeline
    from experiments.feedback_witness_v1.report import report,pack
    source=tmp_path/'source';root=tmp_path/'run';cfg,model,quote=fixture_source(source)
    m=prepare(root,source,tmp_path/'weights',smoke=True,config=cfg)
    class MockGPU:
        calls=0
        def batch(self,requests):
            rows={}
            for key,req in requests:
                p=root/'requests'/(key+'.json');immutable(p,req)
                out=root/'responses'/(key+'.json')
                if out.exists():rows[key]=s.load(out);continue
                self.calls+=1;row=response(model,quote,req['stage']);row['request_sha256']=s.digest(p)
                s.save(out,row);rows[key]=row
            return rows
    gpu=MockGPU();now=time.time();s.save(root/'clock.json',{'started_at':now,'work_deadline':now+300,'hard_deadline':now+600})
    states=run_pipeline(root,m,gpu,now+300,inline=True)
    assert len(states)==12 and gpu.calls==26
    assert all(st['round']==2 for st in states.values())
    result=report(root)
    assert result['rounds_processed_trajectories']==12 and result['cross_rows_terminal']==24
    for g in result['groups']:
        assert g['expected']==2 and g['counts']['final_compiled']==2
        assert g['counts']['invalid_initial_recovered_compilable']==1
        assert g['initial_semantic']['invalid']==1
        assert g['final_semantic']['verified_correct']==2
    run_pipeline(root,m,gpu,now+300,inline=True)
    assert gpu.calls==26
    assert pack(root).exists()


def test_real_isolated_cpu_backends_and_no_retry(tmp_path):
    case,model,quote=problem();cfg=s.load(HERE/'config_8h.json');cfg['cross_solver_seconds']=.2
    for backend in cfg['cross_backends']:
        job={'stage':'cross','config':cfg,'model':model,'backend':backend,'seed':0}
        row=cpu_job(tmp_path,job,25,time.time()+25)
        assert row.get('feasible'),(backend,row)
        assert row.get('objective')==1,(backend,row)
        assert cpu_job(tmp_path,job,25,time.time()+25)==row
    job={'stage':'final','config':cfg,'model':model,'seed':999}
    inp,out=job_paths(tmp_path,job);immutable(inp,job);s.save(inp.with_suffix('.started.json'),{'started_at':time.time()})
    assert cpu_job(tmp_path,job,5,time.time()+5)['status']=='interrupted_no_retry'
    timed=dict(job,seed=1000)
    assert cpu_job(tmp_path,timed,.001,time.time()+5)['status']=='budget_timeout'
    assert cpu_job(tmp_path,timed,5,time.time()+5)['status']=='budget_timeout'
    wrong=copy.deepcopy(model);wrong['constraints']=[]
    cfg.update(probe_seconds=.4,chains=4)
    for arm in ['R','P','X','A','M']:
        job={'stage':'probe','config':cfg,'model':wrong,'contract':model,'quotes':[quote],
             'arm':arm,'base':{},'seed':0}
        row=cpu_job(tmp_path,job,12,time.time()+12)
        assert row.get('status')=='complete',(arm,row)
        assert row.get('cards'),(arm,row)
        if arm!='A':assert row.get('windows'),(arm,row)
