"""Real CPU subprocess lifecycle with explicitly MOCKED LLM responses.

This is an engineering test, not model quality or hardware acceptance evidence.
"""
import importlib.util
import json
import threading
import time
from pathlib import Path
import pytest
from experiments.feedback_study_v1.common import s, HERE, identity
from experiments.feedback_study_v1.data import make_case
from experiments.feedback_study_v1.prepare import prepare
from experiments.feedback_study_v1 import runtime
from experiments.feedback_study_v1.report import report


@pytest.mark.skipif(not importlib.util.find_spec('pyscipopt') or not importlib.util.find_spec('ortools'),reason='native optional test dependencies')
def test_native_cpu_mock_llm_resume(tmp_path, monkeypatch):
    monkeypatch.setenv('OMP_NUM_THREADS','1');monkeypatch.setenv('OPENBLAS_NUM_THREADS','1');monkeypatch.setenv('MKL_NUM_THREADS','1')
    source=tmp_path/'e003';root=tmp_path/'e004';cfg=s.load(HERE/'config.json')
    cfg.update(rounds=1,seeds=[0],chains=4,base_seconds=.03,probe_seconds=.08,final_seconds=.03,
               diagnostic_seconds=.3,reference_seconds=2.,cpu_workers=4,cpu_watchdog_grace=30,
               development_cases=2,extension_sizes=[8],extension_per_cell=0,diagnostic_per_cell=0,
               public_arms=['C0','U2','B1','B2'],
               extension_arms=['C0','U2','B1','B2'],report_interval_seconds=600,inflight_trajectories=24)
    case=make_case('MAXCUT',7,678,'public')
    s.save(source/'cases.CPU_ONLY.json',[case]);s.save(source/'manifest.json',{'cases_digest':s.value_digest([case])})
    s.save(source/'models'/(s.key(case['id'],'llm_typed_flat')+'.json'),
           {'model':case['gold'],'model_sha256':s.value_digest(case['gold']),'generation_cost':{}})
    prepare(root,source,tmp_path/'mock-weights',cfg,smoke=True)
    cases={c['id']:c for c in s.load(root/'evaluation'/'cases.CPU_ONLY.json')}
    stop=threading.Event();errors=[]
    def mocked_llm():
        try:
            while not stop.is_set():
                for path in (root/'requests').glob('*.json'):
                    output=root/'responses'/path.name
                    if output.exists():
                        continue
                    req=s.load(path)
                    payload=({'model':cases[req['case_id']]['gold']} if req['stage']=='initial' else
                             {'hypotheses':[],'notes':'TEST MOCK'} if req['stage']=='review' else
                             {'edits':[],'quote':'','reason':'TEST MOCK retain'})
                    s.save(output,{'status':'generated','response':json.dumps(payload),'request_sha256':s.digest(path),
                                   'generation_attempts':1,'input_tokens':10,'output_tokens':10,'test_mock':True})
                time.sleep(.01)
        except BaseException as exc:
            errors.append(exc)
    thread=threading.Thread(target=mocked_llm,daemon=True);thread.start()
    # Force an early interruption, then replay all durable CPU and GPU stages.
    timer=threading.Timer(2.,lambda:setattr(runtime,'STOP',True));timer.start()
    try:
        runtime.STOP=False;runtime.launch(root,cpu_only=True)
        timer.cancel()
        for _ in range(15):
            runtime.STOP=False;runtime.launch(root,cpu_only=True)
            if s.load(root/'status.json')['status']=='complete':
                break
            time.sleep(.1)
        result=report(root,pairs=True)
        assert result['status']=='complete',s.load(root/'status.json')
        assert s.load(root/'acceptance.json')['passed']
        for path in (root/'states').glob('*.json'):
            row=s.load(path)
            assert row['stage']=='done'
            assert row.get('generation_attempts',0)<=2
        assert not errors
        # Replaying a complete run must not add inference calls or change scores.
        before={p.name:s.digest(p) for p in (root/'evaluation'/'scores').glob('*.json')}
        runtime.STOP=False;runtime.launch(root,cpu_only=True)
        assert before=={p.name:s.digest(p) for p in (root/'evaluation'/'scores').glob('*.json')}
    finally:
        stop.set();timer.cancel();thread.join(timeout=5);runtime.STOP=False
