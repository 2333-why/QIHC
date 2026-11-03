import copy
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import time
import pytest
from jsonschema import ValidationError
from qihc.experiment_runtime import resilient_storage as s
from experiments.feedback_witness_v2 import protocol
from experiments.feedback_witness_v2.common import HERE, ROOT, cpu_job, immutable, job_paths
from test_feedback_witness_v1 import problem, fixture_source


def compact_model():
    return {'domain': {'lower': 0, 'upper': 1},
            'objective': {'sense': 'max', 'linear': {'fill': 1}, 'quadratic': []},
            'constraints': [{'rows': [[0,1],[1,2],[0,2]], 'coefficient': 1, 'op': 'le', 'rhs': 1}]}


def response(stage='patch', model=None):
    obj={'model':model or compact_model(),'source_refs':['s0']}
    if stage=='patch':obj.update(action='replace_families',families=['objective','constraints'],reason='Include the full rule family.')
    return {'status':'generated','response':json.dumps(obj),'input_tokens':10,'output_tokens':20,'generation_attempts':1}


def test_compact_full_family_stable_spans_and_fallible_contract():
    case,gold,_=problem();old=copy.deepcopy(gold);old['constraints']=[]
    value,model=protocol.parse_model(response('contract'),case)
    assert model['objective']['linear']==[1,1,1] and len(model['constraints'])==3
    assert len({c['id'] for c in model['constraints']})==3
    for span in value['source_ledger']:
        assert case['question'][span['start']:span['end']]==span['text']
    # Wrong independent hypothesis is diagnosed but cannot veto a correct repair.
    row=protocol.guard(case,old,response(),old,{})
    assert row['status']=='accepted' and row['hypothesis_diagnostics']['mismatch_counts'][0]>0
    assert protocol.guard(case,None,response(),gold,{})['status']=='accepted'
    changed=compact_model();changed['domain']['upper']=0
    with pytest.raises(ValueError,match='domain'):protocol.guard(case,gold,response(model=changed),gold,{})
    bad=json.loads(response()['response']);bad['source_refs']=['s99999']
    with pytest.raises(ValidationError):protocol.parse_model({'status':'generated','response':json.dumps(bad)},case,'patch')
    bad=json.loads(response()['response']);bad['families']=['constraints'];bad['model']['objective']['linear']={'fill':3}
    with pytest.raises(ValueError,match='undeclared objective'):protocol.guard(case,gold,dict(response(),response=json.dumps(bad)),gold,{})


def test_source_blind_schema_enforcer_and_no_affine():
    from qihc.modeling.formatting import parser
    cfg=s.load(HERE/'config_6h.json');case,model,_=problem()
    for stage in ['contract','patch']:
        schema=protocol.syntax(case,stage);state=parser(schema_override=schema)
        assert 'forall_affine' not in json.dumps(schema)
        for char in response(stage)['response']:
            assert char in state.get_allowed_characters(),char
            state=state.add_character(char)
        assert state.can_end()
    req=protocol.request(dict(case,gold={'SECRET_GOLD':123}),'contract',cfg,model=model,contract={'SECRET_CONTRACT':456})
    assert 'SECRET' not in req['prompt'] and 'current_model' not in req['prompt']
    req=protocol.request(case,'patch',cfg,contract={'model':{'SECRET_MODEL':7},'source_refs':['s0'],'status':'complete'})
    assert 'SECRET_MODEL' not in req['prompt']


def test_same_key_concurrent_cpu_job_has_one_canonical_result(tmp_path):
    # Real subprocess, not a mocked lock. Reproduces the 1.4 .started race.
    case,model,_=problem();cfg=s.load(HERE/'config_6h.json');cfg['cross_solver_seconds']=.2
    job={'stage':'cross','config':cfg,'model':model,'backend':'pbit','seed':0}
    deadline=time.time()+30
    with ThreadPoolExecutor(max_workers=12) as pool:
        rows=list(pool.map(lambda _:cpu_job(tmp_path,job,25,deadline),range(24)))
    assert all(row==rows[0] for row in rows)
    assert rows[0].get('feasible') and rows[0]['status']!='interrupted_no_retry'
    assert len(list((tmp_path/'jobs').glob('*.started.json')))==1
    assert len(list((tmp_path/'outputs').glob('*.json')))==1
    assert cpu_job(tmp_path,job,25,deadline)==rows[0]


def test_waiter_cannot_overwrite_live_owner_or_timeout(tmp_path):
    case,model,_=problem();cfg=s.load(HERE/'config_6h.json')
    job={'stage':'final','config':cfg,'model':model,'seed':30};ip,op=job_paths(tmp_path,job)
    with s.worker_lock(ip.with_suffix('.lock')):
        row=cpu_job(tmp_path,job,1,time.time()+.08)
        assert row['status']=='deadline_waiting_for_job' and not op.exists()
    immutable(ip,job);s.save(ip.with_suffix('.started.json'),{'started_at':0})
    assert cpu_job(tmp_path,job,1,time.time()+1)['status']=='interrupted_no_retry'
    timed=dict(job,seed=31)
    first=cpu_job(tmp_path,timed,.001,time.time()+3)
    assert first['status']=='budget_timeout'
    assert cpu_job(tmp_path,timed,10,time.time()+10)==first


def test_scheduler_deduplicates_and_copies_aliases(tmp_path,monkeypatch):
    from experiments.feedback_witness_v2 import runtime
    calls=[]
    def fake(root,job,timeout,deadline):calls.append(job);return {'status':'ok','x':[1]}
    monkeypatch.setattr(runtime,'cpu_job',fake)
    result=runtime.process_jobs(tmp_path,[(str(i),{'stage':'fixture'}) for i in range(100)],1,time.time()+5,24)
    assert len(calls)==1 and len(result)==100
    result['0']['x'][0]=0
    assert result['1']['x']==[1]


def test_pipeline_full_denominators_cross_controls_resume_and_fixed_duration(tmp_path):
    from experiments.feedback_witness_v2.prepare import prepare
    from experiments.feedback_witness_v2.runtime import run_pipeline
    from experiments.feedback_witness_v2.report import report,pack
    source=tmp_path/'source';root=tmp_path/'run';fixture_cfg,model,_=fixture_source(source)
    cfg=s.load(HERE/'config_6h.json')
    for key in ('case_count','base_seconds','probe_seconds','final_seconds','cross_solver_seconds',
                'semantic_check_seconds','cpu_workers','chains','cross_backends'):cfg[key]=fixture_cfg[key]
    cfg['stage_reservations_seconds']['offline_evaluation']=10
    m=prepare(root,source,tmp_path/'weights',smoke=True,config=cfg)
    class MockGPU:
        calls=0
        def batch(self,requests):
            rows={}
            for key,req in requests:
                p=root/'requests'/(key+'.json');immutable(p,req);out=root/'responses'/(key+'.json')
                if out.exists():rows[key]=s.load(out);continue
                self.calls+=1;row=response(req['stage']);row['request_sha256']=s.digest(p)
                s.save(out,row);rows[key]=row
            return rows
    gpu=MockGPU();now=time.time();s.save(root/'clock.json',{'started_at':now,'work_deadline':now+300,'hard_deadline':now+600})
    states=run_pipeline(root,m,gpu,now+300,inline=True)
    assert len(states)==12 and gpu.calls==26
    result=report(root)
    assert result['status']=='all_stages_processed' and result['cross_rows_terminal']==32
    for group in result['groups']:
        assert group['expected']==2 and group['initial_semantic']['invalid']==1
        assert group['final_semantic']['verified_correct']==2
        assert group['counts']['invalid_initial_recovered_correct']==1
    assert result['contract_semantic_counts']['verified_correct']==2
    for row in result['cross_groups']:
        if row['arm']=='C0':assert row['full_denominator']==2 and row['valid_model_denominator']==1
    run_pipeline(root,m,gpu,now+300,inline=True)
    assert gpu.calls==26
    s.save(root/'status.json',{'stage':'stopped','time':now+100})
    assert report(root)['run_wall_seconds']==100
    assert report(root)['run_wall_seconds']==100
    assert pack(root).exists()


def test_six_hour_capacity_and_real_seven_backend_dispatch(tmp_path):
    from experiments.feedback_witness_v2.budget import assess
    cfg=s.load(HERE/'config_6h.json');audit=assess(cfg,s.load(ROOT/cfg['selection_file']))
    assert audit['status']=='capacity_check_passed' and audit['seconds']['planned_total']<21600
    assert audit['counts']['cross_solver_jobs_max']==1008 and audit['counts']['llm_calls']==234
    case,model,_=problem();cfg['cross_solver_seconds']=.15
    for backend in cfg['cross_backends']:
        row=cpu_job(tmp_path,{'stage':'cross','config':cfg,'model':model,'backend':backend,'seed':0},25,time.time()+25)
        assert row.get('feasible') and row.get('objective')==1,(backend,row)


def test_supervisor_uses_original_deadline_and_terminates_child(tmp_path,monkeypatch):
    import signal
    import subprocess
    import sys
    from experiments.feedback_witness_v2 import supervise
    now=time.time();clock={'started_at':now-100,'work_deadline':now+.1,'hard_deadline':now+30.5}
    s.save(tmp_path/'clock.json',clock)
    original=subprocess.Popen;children=[]
    def launch(cmd,**kwargs):
        child=original([sys.executable,'-c','import time; time.sleep(60)'],**kwargs)
        children.append(child);return child
    monkeypatch.setattr(supervise.subprocess,'Popen',launch)
    monkeypatch.setattr(sys,'argv',['supervise','--run',str(tmp_path)])
    handlers={sig:signal.getsignal(sig) for sig in (signal.SIGINT,signal.SIGTERM)}
    try:
        with pytest.raises(SystemExit) as exc:supervise.main()
        assert exc.value.code!=0 and children[0].poll() is not None
        assert s.load(tmp_path/'clock.json')==clock
        assert s.load(tmp_path/'hard_deadline.json')['status']=='hard_deadline_reached'
        assert time.time()-now<10
    finally:
        for sig,handler in handlers.items():signal.signal(sig,handler)
        for child in children:
            if child.poll() is None:child.kill();child.wait(timeout=5)
