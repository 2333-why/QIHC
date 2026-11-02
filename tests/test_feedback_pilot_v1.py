"""Isolated pilot adapter tests; never install its overrides in shared pytest."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import pytest
from jsonschema import Draft202012Validator, ValidationError
from experiments.feedback_pilot_v1.prepare import select_cases, configuration
from experiments.feedback_pilot_v1.adapter import bounded_schema
from experiments.feedback_study_v1.protocol import schemas


def population():
    return [{'id':f'{task}-{tier}-{i}', 'dataset':f'nlco_{task}_{tier}', 'score':i}
            for task in ('KP','MIS','MVC','MAXCUT','MCP','MDS') for tier in ('S','M','L') for i in range(3)]


def test_selection_is_stratified_order_and_outcome_independent():
    rows=population(); first=[c['id'] for c in select_cases(rows)]
    changed=[dict(c,score=100-c['score'],model=None) for c in reversed(rows)]
    assert first == [c['id'] for c in select_cases(changed)]
    assert len(first)==18
    with pytest.raises(ValueError):select_cases(rows[:-3])


def test_budget_and_finite_schema_without_changing_original():
    cfg=configuration()
    assert cfg['rounds']*len(cfg['public_arms'])*18==144  # round trajectories
    assert cfg['rounds']*2*3*18==216
    raw=schemas('Choose x0 and x1.', ['x0','x1'])[2]; old=copy.deepcopy(raw)
    syntax=bounded_schema(raw); assert raw==old
    v=Draft202012Validator(syntax)
    leaf={'var':0}; expr=leaf
    for _ in range(4):expr={'op':'not','args':[expr]}
    payload={'edits':[{'op':'logic','value':{'id':'l0','expr':expr}}],'quote':'','reason':''}
    v.validate(payload)
    payload['edits'][0]['value']['expr']={'op':'not','args':[expr]}
    with pytest.raises(ValidationError):v.validate(payload)
    payload['edits'][0]['value']['expr']={'op':'not','args':[leaf,leaf]}
    with pytest.raises(ValidationError):v.validate(payload)


def test_isolated_prepare_replay_reporting_and_parser(tmp_path):
    root=Path(__file__).resolve().parents[1]
    code="import runpy; ns=runpy.run_path('tests/test_feedback_pilot_v1.py'); ns['isolated_checks'](r'"+str(tmp_path)+"')"
    result=subprocess.run([sys.executable,'-c',code],cwd=root,env=os.environ.copy(),text=True,capture_output=True,timeout=180)
    assert result.returncode==0,result.stdout+'\n'+result.stderr


def test_gpu_recursion_fails_one_request_then_continues(tmp_path, monkeypatch):
    from types import SimpleNamespace as NS
    from qihc.experiment_runtime import resilient_storage as s
    from experiments.feedback_pilot_v1 import gpu
    from experiments.feedback_study_v1 import common, placement
    from qihc.experiment_runtime import identity
    from qihc.modeling import backend, generation
    cfg=configuration();m={'config':cfg,'model':str(tmp_path/'model'),'model_files':[]}
    monkeypatch.setattr(common,'validate',lambda root:m)
    monkeypatch.setattr(identity,'model_identity',lambda path:[])
    monkeypatch.setattr(placement,'verify_placement',lambda *args:{'status':'passed'})
    monkeypatch.setattr(gpu,'STOP',False)
    loaded=[]
    cuda=NS(device_count=lambda:1,empty_cache=lambda:None,memory_allocated=lambda i:0,
            OutOfMemoryError=type('TestOOM',(RuntimeError,),{}))
    def model(*args,**kwargs):
        loaded.append(1)
        return NS(model=object(),torch=NS(cuda=cuda),tokenizer=None,
                  _inputs=lambda prompt:{'input_ids':NS(shape=(1,10))})
    monkeypatch.setattr(backend,'LocalModel',model)
    calls=[]
    def generate(gen,items):
        calls.append(items[0][0].stem)
        if len(calls)==1:raise RecursionError('test malformed nested JSON')
        s.save(tmp_path/'control.json',{'status':'stopped'})
        return [{'status':'generated','response':'{}','input_tokens':10,'output_tokens':2}]
    monkeypatch.setattr(generation,'generate',generate)
    s.save(tmp_path/'control.json',{'status':'running'})
    for key in ('a','b'):
        path=tmp_path/'requests'/(key+'.json')
        s.save(path,{'stage':'review','prompt':'TEST MOCK','input_remaining':32768})
        s.save(tmp_path/'pending/0'/path.name,{'request_sha256':s.digest(path)})
    gpu.worker(tmp_path,0)
    assert len(loaded)==1 and len(calls)==2
    first=s.load(tmp_path/'responses'/(calls[0]+'.json'))
    second=s.load(tmp_path/'responses'/(calls[1]+'.json'))
    assert first['status']=='generation_error' and first['generation_attempts']==1 and first['cost_incomplete']
    assert second['status']=='generated' and second['generation_attempts']==1
    assert {s.load(p)['status'] for p in (tmp_path/'inference_events').glob('*.json')}=={'failed','complete'}


def isolated_checks(path):
    from qihc.experiment_runtime import resilient_storage as s
    from experiments.feedback_study_v1 import common
    oldhash=common.fingerprint()
    from experiments.feedback_pilot_v1.prepare import prepare
    from experiments.feedback_pilot_v1.report import report
    source=Path(path)/'source'; root=Path(path)/'pilot'
    gold={'schema':'bounded-integer-v2','variables':[{'name':'x0','lower':0,'upper':1}],
          'objective':{'sense':'max','linear':[1],'quadratic':[]},'constraints':[],'logic':[]}
    cases=[dict(c,question='Maximize x0; x0 is binary.',gold=gold) for c in population()]
    s.save(source/'cases.CPU_ONLY.json',cases)
    s.save(source/'manifest.json',{'cases_digest':s.value_digest(cases),'config':{'seeds':[0], 'seconds':[20],'backends':['scip','pbit']}})
    selected=select_cases(cases)
    # Preserve a deliberately invalid initial; selection must not drop it.
    invalid=selected[0]['id']
    for c in selected:
        cid=c['id']; ref={'gold_sha256':s.value_digest(gold),'result':{'feasible':True,'optimality_proven':True,'objective':1}}
        s.save(source/'references'/(s.key(cid)+'.json'),ref)
        for track in ('canonical','llm_typed_flat'):
            spec=None if track=='llm_typed_flat' and cid==invalid else gold
            s.save(source/'models'/(s.key(cid,track)+'.json'),{'model':spec,'model_sha256':s.value_digest(spec) if spec else None,'response_sha256':'response','error':'test-invalid' if spec is None else None})
            for backend in ('scip','pbit'):
                s.save(source/'scores'/s.key(cid,track)/f'{backend}-20-0.json',
                       {'case_id':cid,'source_arm':track,'backend':backend,'budget_seconds':20,'seed':0,
                        'response_sha256':'response','model_sha256':s.value_digest(spec) if spec else None,
                        'verified_feasible':spec is not None,'verified_objective':1 if spec else None})
    before={str(p):s.digest(p) for p in source.rglob('*.json')}
    manifest=prepare(root,source,Path(path)/'mock-model',smoke=True)
    assert len(s.load(root/'cases.public.json'))==18
    assert all('gold' not in c for c in s.load(root/'cases.public.json'))
    assert manifest==prepare(root,source,Path(path)/'mock-model',smoke=True)
    assert report(root)['completed']==0
    from experiments.feedback_study_v1 import protocol,runtime
    from experiments.feedback_pilot_v1 import cpu
    from experiments.feedback_pilot_v1.gpu import requires_process_restart
    from experiments.feedback_pilot_v1.adapter import _fingerprint
    assert oldhash==_fingerprint()
    assert runtime.ENTRY=='experiments.feedback_pilot_v1'
    assert not requires_process_restart(RecursionError('nested JSON'))
    assert requires_process_restart(RuntimeError('CUDA device failure'))
    # Real LMFE parser validates the finite reference DAG when locally available.
    import importlib.util
    if importlib.util.find_spec('lmformatenforcer'):
        from qihc.modeling.formatting import parser
        syntax=protocol.schemas('Binary x0',['x0'])[2]
        payload=json.dumps({'edits':[{'op':'logic','value':{'id':'l0','expr':{'op':'not','args':[{'var':0}]}}}],'quote':'','reason':'retain'})
        p=parser('flat',syntax)
        for ch in payload:
            assert ch in p.get_allowed_characters(),ch
            p=p.add_character(ch)
        assert p.can_end()
    # Synthetic response + real mathematical CPU stages and on-disk checkpoints.
    cfg=manifest['config']; cfg.update(base_seconds=.002,probe_seconds=.006,final_seconds=.002,diagnostic_seconds=.02,chains=2)
    manifest['config_digest']=s.value_digest(cfg);s.save(root/'manifest.json',manifest)
    case=next(c for c in s.load(root/'cases.public.json') if c['id']!=invalid)
    goldcase=next(c for c in s.load(root/'evaluation/cases.CPU_ONLY.json') if c['id']==case['id'])
    for arm in cfg['public_arms']:
        key=s.key(case['id'],arm,'0')
        st={'id':key,'case_id':case['id'],'arm':arm,'seed':0,'round':0,'stage':'base',
            'model':copy.deepcopy(gold),'initial_valid':True,'split':'pilot','dataset':case['dataset'],'size':1}
        runtime.save_state(root,st)
        for _ in range(40):
            def schedule(k,job,out,seconds):
                result=cpu.execute(s.load(job));result['job_sha256']=s.digest(job);s.save(out,result)
            done=runtime.advance(root,case,st,cfg,schedule)
            if st.get('request_id'):
                request=root/'requests'/(st['request_id']+'.json')
                req=s.load(request)
                assert 'gold' not in req['prompt'] and 'verified_objective' not in req['prompt']
                payload={'hypotheses':[],'notes':'TEST MOCK'} if req['stage']=='review' else {'edits':[],'quote':'','reason':'TEST MOCK retain'}
                s.save(root/'responses'/request.name,{'status':'generated','response':json.dumps(payload),'request_sha256':s.digest(request),'generation_attempts':1,'input_tokens':10,'output_tokens':10})
            # Reopen every transition to exercise saved model identity/state.
            st=runtime.load_state(root,key)
            if done:break
        assert st['stage']=='done',st
        assert st.get('generation_attempts',0)==(0 if arm=='C0' else 4)
        assert not st.get('errors'),st.get('errors')
        from types import SimpleNamespace
        inline=SimpleNamespace(schedule=schedule)
        for _ in range(3):
            if runtime.evaluate_tasks(root,st,goldcase,cfg,inline):break
        score=root/'evaluation/scores'/(key+'.json');h=s.digest(score)
        assert runtime.evaluate_tasks(root,st,goldcase,cfg,inline) and s.digest(score)==h
    result=report(root)
    assert result['completed']==4 and result['expected']==72 and result['status']=='partial'
    assert all(p['paired_instances']==1 for p in result['paired'])
    # Exercise the actual new CPU module dispatch once, in a separate process.
    jobpath=root/'test-native-job.json';outpath=root/'test-native-output.json'
    s.save(jobpath,{'stage':'base','config':cfg,'model':gold,'arm':'B2','seed':0,'initial':[],'archive':[[2]]})
    proc=subprocess.run([sys.executable,'-m','experiments.feedback_pilot_v1.cpu',str(jobpath),str(outpath)],
                        text=True,capture_output=True,timeout=60)
    assert proc.returncode==0,proc.stderr
    assert s.load(outpath)['status']=='complete',s.load(outpath)
    assert s.load(outpath)['rejected_stale_candidate_records']==2
    assert before=={str(p):s.digest(p) for p in source.rglob('*.json')}
    # Fail closed on wrong experiment and tampered baseline identities.
    bad=Path(path)/'old';s.save(bad/'manifest.json',{'experiment_id':'E004-1.1'})
    try:common.validate(bad)
    except ValueError:pass
    else:raise AssertionError('old experiment accepted')
    s.save(root/'evaluation/e003_baselines.json',[])
    try:report(root)
    except ValueError:pass
    else:raise AssertionError('tampered baselines accepted')
