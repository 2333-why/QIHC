import copy
from pathlib import Path
import numpy as np
import pytest
from qihc.experiment_runtime.storage import save,load
from experiments.pbit_joint_v1 import search as s,modeling,run as r
from test_pvcd_cycle import tiny_engine,problem


def binary_spec():
    _,gold,_=problem();gold=copy.deepcopy(gold)
    gold['variables']=[dict(name='x0',lower=0,upper=1),dict(name='x1',lower=0,upper=1)]
    gold['objective']['linear']=[1,1];gold['constraints']=[dict(id='cap',a=[1,1],op='le',rhs=1)]
    return gold


def test_conditional_clamp_preserves_protected_half_model_archive_and_budget():
    spec=binary_spec();original=copy.deepcopy(spec)
    state=s.initial(spec,3,8);state=s.advance(spec,state,'cool',11,8)
    before=copy.deepcopy(state);clamps=s.intervention(spec,state)
    assert len(clamps)==1
    cool=s.advance(spec,state,'cool',20,16);clamped=s.advance(spec,state,'clamp',20,16)
    assert clamped['x'][:4]==cool['x'][:4]
    for i,value in clamps.items():assert all(x[i]==value for x in clamped['x'][4:])
    assert state==before and spec==original
    assert s.key(clamped['best'],spec)<=s.key(state['best'],spec)
    cfg=dict(r.CFG,chains=8,window_steps=8,decisions=2)
    for a in s.ACTIONS:
        result=s.episode(spec,'fixture',cfg,lambda *args:a)
        assert result['heat_bath_attempts']==192
    assert s.advance(spec,state,'heat',20,16)==s.advance(spec,state,'heat',20,16)


def test_no_full_clamp_or_gold_in_policy_prompt():
    _,single,_=problem();state=s.initial(single,1,4)
    state=s.advance(single,state,'cool',2,4)
    assert s.intervention(single,state)=={}
    text=s.prompt(single,state,4)
    assert 'reference_proven' not in text and 'case_id' not in text and 'gold' not in text
    fixed=copy.deepcopy(single);fixed['variables'][0]['upper']=0
    result=s.episode(fixed,'fixed',dict(r.CFG,chains=4),lambda *a:'clamp')
    assert result['x']==[0] and result['heat_bath_attempts']==0


def test_collapsed_training_state_is_legal_recomputes_conditionals_and_preserves_input():
    spec=binary_spec();state=s.advance(spec,s.initial(spec,3,8),'cool',11,8);original=copy.deepcopy(state)
    collapsed=s.collapse_training_state(spec,state)
    assert state==original and collapsed['best']==state['best']
    assert all(x==state['best']['x'] for x in collapsed['x'])
    assert collapsed['history'][-1]['diversity']==1/8
    assert all(0<=row['conditional_q_mean']<=1 for row in collapsed['q'])
    for action in s.ACTIONS:assert s.advance(spec,collapsed,action,12,8)['steps']==16


def test_terminal_labels_require_improvement_over_cooling_not_population(monkeypatch):
    calls=[]
    def outcome(spec,state,action,seed,remaining,steps):
        calls.append((action,seed,remaining));return .5
    monkeypatch.setattr(s,'outcome',outcome)
    label=s.label({}, {},3,3,r.CFG)
    assert label['action']=='cool' and not label['confirmed_gain']
    assert all(x[2]==3 for x in calls)
    assert [x[1] for x in calls[:2]]==[x[1] for x in calls[2:4]]
    monkeypatch.setattr(s,'outcome',lambda spec,state,action,*a:.6 if action=='clamp' else .5)
    assert s.label({}, {},3,3,r.CFG)['action']=='clamp'


def test_weighted_supervision_uses_verified_gold_span_not_generated_repair():
    from experiments.posttraining_v1.data import target
    _,gold,wrong=problem();answer=target(gold)
    class Tok:
        def __call__(self,text,**kw):return {'offset_mapping':[(i,i+1) for i in range(len(text))]}
        def encode(self,text,**kw):return list(range(len(text)))
    card=dict(kind='feasibility',x=[1],correct_feasible=False,candidate_feasible=True)
    weights=modeling.token_weights(Tok(),answer,wrong,gold,[card])
    assert len(weights)==len(answer) and 2 in weights and 1 in weights
    assert modeling.token_weights(Tok(),answer,None,gold,[])==[1.]*len(answer)
    evidence=modeling.collect(wrong,gold,'train',dict(r.CFG,chains=4,probe_steps=4))
    assert evidence['cards'] and {x['side'] for x in evidence['trace']}=={'candidate','gold'}


def test_optimizer_real_lora_resume_preserves_result(tiny_engine,tmp_path,monkeypatch):
    import torch
    monkeypatch.setattr(r,'CFG',dict(r.CFG,checkpoint_steps=1))
    e=tiny_engine;original=e.snapshot();calls=[0]
    def backward(i):
        calls[0]+=1
        if calls[0]==2:raise RuntimeError('interrupted')
        return dict(loss=e.backward('sample',[1,2,3],'nll'))
    with pytest.raises(RuntimeError,match='interrupted'):
        r.optimize(e,tmp_path/'resumed',[0,1],1e-4,backward,dict(test=True))
    def normal(i):return dict(loss=e.backward('sample',[1,2,3],'nll'))
    r.optimize(e,tmp_path/'resumed',[0,1],1e-4,normal,dict(test=True));resumed=e.snapshot()
    e.restore(original);r.optimize(e,tmp_path/'straight',[0,1],1e-4,normal,dict(test=True))
    assert all(torch.equal(v,e.snapshot()[k]) for k,v in resumed.items())
    assert any(not torch.equal(v,original[k]) for k,v in resumed.items())


def test_control_training_gate_counts_distinct_cases(tmp_path,monkeypatch):
    monkeypatch.setattr(r,'RUN',tmp_path)
    rows=[('one',{},dict(label=dict(action='clamp',confirmed_gain=True)))]*50
    rows.append(('one',{},dict(label=dict(action='cool',confirmed_gain=False))))
    monkeypatch.setattr(r,'control_examples',lambda:rows)
    assert not r.control_gate()['passed']
    rows.extend((cid,{},dict(label=dict(action='heat',confirmed_gain=True))) for cid in ('two','three'))
    assert r.control_gate()['passed']


def test_factorial_report_interaction_and_partial_coverage(tmp_path,monkeypatch):
    monkeypatch.setattr(r,'RUN',tmp_path);monkeypatch.setattr(r,'CFG',dict(r.CFG,bootstrap_repeats=10))
    monkeypatch.setattr(r,'test_cases',lambda:[dict(id='a',stage='main'),dict(id='b',stage='scale')])
    monkeypatch.setattr(r,'train_rows',lambda:[{}])
    monkeypatch.setattr(r,'training_status',lambda *a:dict(passed=True))
    save(tmp_path/'control_gate.json',dict(passed=True,examples=1))
    # Joint gain is larger than the sum of isolated gains in this fixture.
    for arm in r.CFG['arms']:
        for control in r.CFG['controls']:
            for cid,stage in [('a','main'),('b','scale')]:
                hit=arm=='pbit' and control=='learned'
                save(tmp_path/f'evaluations/{arm}__{control}/{cid}.json',dict(stage=stage,
                    score=dict(original_feasible=True,optimal_hit=hit,reference_proven=True,gap=0 if hit else .5),
                    semantic=dict(status='verified_correct'),invalid_model=False,episode_seconds=1,decision_seconds=.1,
                    search_seconds=.9,heat_bath_attempts=20480,intervention_writes=0,actions=['cool']))
    result=r.report()
    assert result['complete'] and result['comparisons']['interaction']['delta']==1
    assert result['summary']['pbit__learned']['all']['proven_feasible_gap']==0
    assert result['summary']['base__schedule']['all']['proven_feasible_gap_cases']==2
    (tmp_path/'evaluations/base__schedule/a.json').unlink()
    result=r.report();assert not result['complete'] and not result['comparisons']


def test_evaluation_invalid_models_count_as_failures_and_reuse_predictions(tmp_path,monkeypatch):
    from experiments.feedback_active_v3 import cpu
    from qihc.generic.feedback import verification
    monkeypatch.setattr(r,'RUN',tmp_path)
    _,spec,_=problem();goldpath=tmp_path/'gold.json';save(goldpath,[dict(id='a',gold=spec)])
    monkeypatch.setattr(r,'validate',lambda:dict(gold_paths=[str(goldpath)]))
    monkeypatch.setattr(r,'test_cases',lambda:[dict(id='a',stage='main')])
    save(r.prediction('base','a'),dict(model=None))
    monkeypatch.setattr(cpu,'execute',lambda args:dict(status='optimal',objective=0,optimality_proven=True))
    monkeypatch.setattr(verification,'equivalent',lambda model,gold:dict(status='invalid'))
    r.evaluate('base','schedule');r.evaluate('base','rule')
    first=load(tmp_path/'evaluations/base__schedule/a.json');second=load(tmp_path/'evaluations/base__rule/a.json')
    assert first['invalid_model'] and not first['score']['original_feasible']
    assert first['model_hash']==second['model_hash'] and first['actions']==[]


def test_controller_uses_real_categorical_lora_gradient(tiny_engine,tmp_path,monkeypatch):
    import torch
    from experiments.posttraining_v1 import engine
    e=tiny_engine;original=e.snapshot()
    monkeypatch.setattr(r,'RUN',tmp_path);monkeypatch.setattr(r,'validate',lambda:dict(model='fixture'))
    monkeypatch.setattr(engine,'Engine',lambda path:e)
    monkeypatch.setattr(r,'action_ids',lambda e:[1,2,3])
    monkeypatch.setattr(s,'prompt',lambda *a:'control')
    monkeypatch.setattr(r,'CFG',dict(r.CFG,min_control_gain_cases=1))
    examples=[('one',{},dict(state={},remaining=2,label=dict(action=a,confirmed_gain=a!='cool'))) for a in ('cool','clamp')]
    monkeypatch.setattr(r,'control_examples',lambda:examples)
    r.controller_worker()
    assert r.training_status(tmp_path/'controller',4)['passed']
    assert any(not torch.equal(v,original[k]) for k,v in e.snapshot().items())


def test_model_worker_trains_weighted_gold_and_freezes_predictions(tiny_engine,tmp_path,monkeypatch):
    import torch
    from experiments.posttraining_v1 import engine
    from experiments.posttraining_v1.data import target
    e=tiny_engine;case,gold,wrong=problem();held=dict(case,id='held')
    adapter=tmp_path/'parent.pt';torch.save(e.snapshot(),adapter)
    monkeypatch.setattr(r,'RUN',tmp_path)
    monkeypatch.setattr(r,'validate',lambda:dict(model='fixture',adapter=str(adapter)))
    monkeypatch.setattr(engine,'Engine',lambda path:e)
    monkeypatch.setattr(r,'train_rows',lambda:[dict(case=case,gold=gold)])
    monkeypatch.setattr(r,'test_cases',lambda:[held])
    monkeypatch.setattr(e,'evaluate',lambda *a:dict(choices=[dict(finish_reason='stop',message=dict(content=target(gold)))],usage={}))
    save(tmp_path/'cards/fixture.json',dict(model=wrong,cards=[dict(kind='feasibility',x=[1],correct_feasible=False,candidate_feasible=True)]))
    r.model_worker('pbit')
    assert r.training_status(tmp_path/'models/pbit',2)['passed']
    assert load(r.prediction('pbit','held'))['model'] is not None
    assert load(tmp_path/'models/pbit/done.json')['predictions']==2


def test_joint_gpu_sequence_and_device_assignment(tmp_path,monkeypatch):
    import torch
    from contextlib import nullcontext
    from types import SimpleNamespace
    monkeypatch.setattr(r,'RUN',tmp_path);monkeypatch.setattr(r,'validate',lambda:None)
    monkeypatch.setattr(torch.cuda,'device_count',lambda:2)
    monkeypatch.setattr(torch.cuda,'device',lambda i:nullcontext())
    monkeypatch.setattr(torch.cuda,'get_device_properties',lambda i:SimpleNamespace(name='mock',total_memory=80*1024**3))
    monkeypatch.setattr(torch.cuda,'is_bf16_supported',lambda:True)
    monkeypatch.setattr(torch.cuda,'synchronize',lambda i:None)
    ones=torch.ones;monkeypatch.setattr(torch,'ones',lambda *a,**kw:ones(*a,**dict(kw,device='cpu')))
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','2,3')
    events=[]
    monkeypatch.setattr(r,'model_worker',lambda a:events.append(('base',a)))
    monkeypatch.setattr(r,'batch',lambda tasks,devices:events.append(('batch',tasks,devices)))
    def labels():events.append(('labels',));return dict(passed=True)
    monkeypatch.setattr(r,'make_control_labels',labels)
    monkeypatch.setattr(r,'report',lambda:dict(passed=True))
    r.launch()
    assert events[1][1]==[['model','plain'],['model','pbit']] and events[1][2]==['2','3']
    assert events[2]==('labels',) and events[4][1]==[['controller']]
    assert len(events[3][1])==6 and events[3][2]==['','']
    assert len(events[5][1])==3 and all(t[-1]=='learned' for t in events[5][1])
