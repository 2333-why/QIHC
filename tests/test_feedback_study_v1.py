import copy
import json
from pathlib import Path
import numpy as np
import pytest
from qihc.generic import feedback
from qihc.generic.bounded import objective, violations
from experiments.feedback_study_v1 import protocol, runtime, cpu
from experiments.feedback_study_v1.common import HERE, s, configurations
from experiments.feedback_study_v1.data import make_case, extension


def config():
    c=s.load(HERE/'config.json');c.update(base_seconds=.015,probe_seconds=.04,final_seconds=.015,chains=4,rounds=2)
    return c


def case():
    return make_case('MAXCUT',5,1234,'public')


def test_energy_matches_original_residuals():
    for family in ('KP','MIS','MVC','MAXCUT','MCP','MDS'):
        c=make_case(family,8,1234,'test'); spec=c['gold'];x=np.random.default_rng(1).integers(0,2,(12,8))
        e=feedback.Energy(spec);f,r,value=e.values(x)
        np.testing.assert_allclose(f,objective(spec,x))
        np.testing.assert_allclose(r,violations(spec,x)[:,1:])
        np.testing.assert_allclose(value,e.sign*f/e.scale+((r/e.div)**2)@e.weights)


def test_conditional_probability_and_clamp_release():
    spec=case()['gold']; before=copy.deepcopy(spec)
    np.testing.assert_allclose(feedback.conditional_probability(0,1,1),1/(1+np.exp(1)))
    assert feedback.conditional_probability(0,-1,1)>.5
    for method in ('pbit','metropolis'):
        row=feedback.sample(spec,.04,1,chains=16,method=method,clamps={0:1},max_steps=50)
        assert all(x[0]==1 for x in row['states'])
        assert all(x[0]==1 for x in row['final'])
    assert spec==before
    fresh=feedback.sample(spec,0,1,chains=64)
    assert {x[0] for x in fresh['final']}=={0,1}


def test_bounded_integer_encoding_and_logic():
    spec={'schema':'bounded-integer-v2','variables':[{'name':'x0','lower':-2,'upper':3}],
          'objective':{'sense':'max','linear':[1],'quadratic':[]},'constraints':[],'logic':[]}
    row=feedback.sample(spec,.01,0,chains=8,clamps={0:2})
    assert row['final']==[[2]]*8
    assert feedback.encode_states(spec,[[-2],[-1],[0],[1],[2],[3]]).shape[0]==6


def test_patch_does_not_mutate_and_ids_are_checked():
    spec=make_case('MIS',5,22,'test')['gold'];old=copy.deepcopy(spec)
    patch={'edits':[{'op':'sense','value':'min'}],'quote':'anything','reason':'test'}
    result=protocol.apply_patch(spec,patch,'')
    assert result['objective']['sense']=='min' and spec==old
    cid=spec['constraints'][0]['id']
    bad={'edits':[{'op':'logic','value':{'id':cid,'expr':{'var':0}}}],'quote':'','reason':''}
    with pytest.raises(ValueError,match='Unique'):
        protocol.apply_patch(spec,bad,'')
    with pytest.raises(ValueError):
        protocol.apply_patch(spec,{'edits':[{'op':'remove_constraint','id':'missing'}]},'')


def test_schemas_accept_empty_and_source_quote_is_not_execution_gate():
    import jsonschema
    c=case();_,review,patch=protocol.schemas(c['question'],c['variable_names'])
    jsonschema.validate({'hypotheses':[],'notes':'ok'},review)
    jsonschema.validate({'edits':[],'quote':'','reason':'retain'},patch)
    assert protocol.apply_patch(c['gold'],{'edits':[],'quote':'','reason':'retain'},c['question'])==c['gold']


@pytest.mark.parametrize('arm',['C0','C1','C2','C3','M1','M2','M3','M4','U1','U2','no_memory','no_q','event'])
def test_durable_trajectory_end_to_end(tmp_path,arm):
    c=case();cfg=config();state={'id':arm,'case_id':c['id'],'arm':arm,'seed':0,'round':0,'stage':'base',
        'model':c['gold'],'split':'public'}
    def schedule(key,path,out,seconds):
        row=cpu.execute(s.load(path));row['job_sha256']=s.digest(path);s.save(out,row)
    runtime.save_state(tmp_path,state)
    for _ in range(60):
        state=runtime.load_state(tmp_path,arm)  # Resume every single transition.
        if state.get('request_id'):
            rid=state['request_id'];path=tmp_path/'requests'/(rid+'.json');req=s.load(path)
            payload={'hypotheses':[],'notes':'retain'} if req['stage']=='review' else {'edits':[],'quote':'','reason':'retain'}
            s.save(tmp_path/'responses'/path.name,{'status':'generated','response':json.dumps(payload),
                'request_sha256':s.digest(path),'generation_attempts':1,'input_tokens':10,'output_tokens':10})
        if runtime.advance(tmp_path,c,state,cfg,schedule):
            break
    assert state['stage']=='done'
    assert state['x'] is not None
    assert state.get('generation_attempts',0)<=(4 if arm!='C0' else 0)
    assert len(list((tmp_path/'rounds'/arm).glob('*.json')))==2
    if arm=='C1':
        assert all('witnesses' not in v for v in state['memory'])
    if arm=='no_memory':
        assert not state['memory']
    if arm=='no_q':
        snapshot=s.load(tmp_path/'rounds'/arm/'0.json')
        assert not snapshot['base']['q']


def test_response_accounting_idempotent(tmp_path):
    s.save(tmp_path/'requests/r.json',{'prompt':'test'})
    s.save(tmp_path/'responses/r.json',{'request_sha256':s.digest(tmp_path/'requests/r.json'),'generation_attempts':1,'output_tokens':7})
    state={};runtime.consume(tmp_path,'r',state);runtime.consume(tmp_path,'r',state)
    assert state['generation_attempts']==1 and state['output_tokens']==7


def test_formal_count_and_data_size():
    cfg=s.load(HERE/'config.json');cases=extension(cfg)
    assert len(cases)==1200
    assert sum(c.get('diagnostic_subset',False) for c in cases)==240
    assert sum(len(configurations(c,cfg)) for c in cases)==23760
    assert 900*len(cfg['public_arms'])*len(cfg['seeds'])+23760==56160
    assert max(len(c['gold']['constraints']) for c in cases)<=512


def test_no_gold_in_prompt_or_probe_job(tmp_path):
    c=case();c['gold']['secret']='NEVER_IN_PROMPT'
    spec=copy.deepcopy(c['gold']);spec.pop('secret')
    state={'id':'t','round':0,'arm':'U2','seed':0,'model':spec,'review':{},'base':{}}
    req=protocol.build_request(c,state,'review')
    assert 'NEVER_IN_PROMPT' not in json.dumps(req)
    _,path,_=runtime.cpu_job(tmp_path,state,'probe',config(),c)
    assert 'NEVER_IN_PROMPT' not in path.read_text()


def test_final_archive_revalidated_after_patch():
    c=case();spec=c['gold'];x=[0]*5
    spec['objective']={'sense':'min','linear':[1]*5,'quadratic':[]}
    job={'stage':'final','model':spec,'arm':'C0','config':config(),'seed':1,'archive':[x]}
    row=cpu.execute(job)
    assert row['x']==x and row['archive_revalidated_states']==1


def test_request_build_failure_is_terminal_not_global(tmp_path,monkeypatch):
    c=case();state={'id':'broken','round':0,'arm':'U2','model':c['gold']}
    def broken(*args,**kwargs):
        raise ValueError('bad source binding')
    monkeypatch.setattr(runtime,'build_request',broken)
    key=runtime.enqueue(tmp_path,c,state,'review',config())
    response=runtime.consume(tmp_path,key,state)
    assert response['status']=='generation_error'
    assert state['generation_attempts']==0
    assert not list((tmp_path/'pending').rglob('*.json'))


def test_domain_repair_and_out_of_domain_archive():
    c=case();spec=c['gold']
    changed=protocol.apply_patch(spec,{'edits':[{'op':'domain','index':0,'lower':0,'upper':0}]},c['question'])
    row=cpu.execute({'stage':'final','model':changed,'arm':'C0','config':config(),'seed':1,
                    'initial':[[1]*5]*4,'archive':[[1]*5]})
    assert row['archive_revalidated_states']==0
    assert row['x'][0]==0


def test_pending_marker_removed_concurrently(tmp_path,monkeypatch):
    from experiments.feedback_study_v1.gpu import pending
    path=tmp_path/'pending/0/a.json';s.save(path,{'request_sha256':'x'})
    original=Path.stat
    def disappear(self,*args,**kwargs):
        if self==path:
            self.unlink(missing_ok=True)
            raise FileNotFoundError(str(self))
        return original(self,*args,**kwargs)
    monkeypatch.setattr(Path,'stat',disappear)
    assert pending(tmp_path,0,1)==[]


def test_gpu_oom_is_one_terminal_request_then_queue_continues(tmp_path,monkeypatch):
    from types import SimpleNamespace
    from experiments.feedback_study_v1 import gpu
    from qihc.modeling import backend,generation
    from qihc.experiment_runtime import identity as model_identity
    class OOM(RuntimeError):
        pass
    fake=SimpleNamespace(model=SimpleNamespace(hf_device_map={'':'cuda:0'}),
        torch=SimpleNamespace(cuda=SimpleNamespace(OutOfMemoryError=OOM,empty_cache=lambda:None,
                 memory_allocated=lambda i:100,device_count=lambda:1)),
        _inputs=lambda p:{'input_ids':SimpleNamespace(shape=(1,5))})
    monkeypatch.setattr(backend,'LocalModel',lambda *a,**k:fake)
    monkeypatch.setattr(model_identity,'model_identity',lambda p:[])
    cfg=config();cfg['batch_size']=1
    monkeypatch.setattr(gpu,'validate',lambda r:{'model':'test','model_files':[],'config':cfg})
    calls=[]
    def generate(model,items):
        calls.append(1)
        if len(calls)==1:
            raise OOM('simulated one-request OOM')
        s.save(tmp_path/'control.json',{'status':'stopped'})
        return [{'status':'generated','response':'{}','input_tokens':5,'output_tokens':2}]
    monkeypatch.setattr(generation,'generate',generate)
    for name in ('a','b'):
        path=tmp_path/'requests'/(name+'.json');s.save(path,{'prompt':name,'stage':'review'})
        s.save(tmp_path/'pending/0'/path.name,{'request_sha256':s.digest(path)})
    s.save(tmp_path/'control.json',{'status':'running'})
    gpu.STOP=False;gpu.worker(tmp_path,0)
    rows=[s.load(p) for p in (tmp_path/'responses').glob('*.json')]
    assert sorted(r['status'] for r in rows)==['generated','generation_error']
    assert len(calls)==2
    assert next(r for r in rows if r['status']=='generation_error')['cost_incomplete']


def test_unproven_reference_is_not_reported_as_optimal(tmp_path):
    from experiments.feedback_study_v1.common import identity
    from experiments.feedback_study_v1.report import report
    c={'id':'test-reference','dataset':'test','size':2,'split':'extension'}
    cfg=config();cfg.update(extension_arms=['U2','C1'],seeds=[0])
    s.save(tmp_path/'manifest.json',{'config':cfg});s.save(tmp_path/'cases.public.json',[c])
    for arm in cfg['extension_arms']:
        key=identity(c['id'],arm,0)
        s.save(tmp_path/'states'/(key+'.json'),{'stage':'done'})
        s.save(tmp_path/'evaluation/scores'/(key+'.json'),{'original_feasible':True,
               'reference_proven':False,'gap_percent':2.,'optimal_hit':False})
    summary=report(tmp_path,pairs=True)
    assert summary['status']=='complete'
    assert all(r['quality5_all_assigned']==0 and r['quality5_reference_all_assigned']==1 for r in summary['rows'])
    metrics={r['metric'] for r in s.load(tmp_path/'paired_extension.json')['comparisons']}
    assert metrics=={'original_feasible','quality5_reference'}


def test_launcher_accepts_scheduler_rank_metadata(tmp_path):
    import os,shutil,subprocess
    bash=shutil.which('bash')
    if os.name=='nt':
        bash='C:/Program Files/Git/usr/bin/bash.exe'
    if not bash or not Path(bash).exists():
        pytest.skip('Bash not installed')
    prefix=tmp_path/'mock-env';p=prefix/'bin/python';p.parent.mkdir(parents=True)
    p.write_text('#!/usr/bin/env bash\nprintf "MOCK_INTERPRETER %s\\n" "$*"\n',encoding='utf-8',newline='\n')
    p.chmod(0o755)
    env=dict(os.environ,QIHC_ENV_PREFIX=prefix.as_posix(),QIHC_BASE=tmp_path.as_posix(),
             QIHC_E004_RUN=(tmp_path/'run').as_posix(),WORLD_SIZE='4',RANK='1',LOCAL_RANK='0')
    command=[bash,(HERE/'01_run_4gpu.sh').as_posix()]
    if os.name=='nt':
        # The direct MSYS executable lacks Git's wrapper PATH initialization.
        command=[bash,'-c','export PATH=/usr/bin:/bin:$PATH; exec bash "$1"',
                 'launcher-test',(HERE/'01_run_4gpu.sh').as_posix()]
    output=tmp_path/'launcher-output.txt'
    # Git Bash process substitution may inherit PIPE handles on Windows. A real
    # file tests the launcher without mistaking tee's handle lifetime for a hang.
    with output.open('w',encoding='utf-8') as f:
        result=subprocess.run(command,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=180 if os.name=='nt' else 30)
    text=output.read_text(encoding='utf-8')
    assert result.returncode==0,text
    assert 'MOCK_INTERPRETER' in text
