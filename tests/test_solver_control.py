import copy
import json
import numpy as np
import pytest
from qihc.experiment_runtime.storage import save,load
from qihc.experiment_runtime import pbit_control as c
from test_pvcd_cycle import tiny_engine,problem


def test_heat_bath_continuation_archive_and_domains():
    _,spec,_=problem();original=copy.deepcopy(spec)
    s=c.initial(spec,17,8);old=copy.deepcopy(s)
    for a in c.ACTIONS:
        out=c.advance(spec,s,a,29,16)
        assert s==old and spec==original
        assert out['steps']==16 and len(out['history'])==1
        assert np.asarray(out['x']).min()>=0 and np.asarray(out['x']).max()<=1
        assert c.key(out['best'],spec)<=c.key(s['best'],spec)
        assert c.advance(spec,json.loads(json.dumps(s)),a,29,16)==out
    out=c.advance(spec,out,'heat',43,16)
    assert out['steps']==32 and len(out['history'])==2
    # Fixed weights/one bit: empirical heat-bath frequency matches its conditional.
    _,_,free=problem();state=c.initial(free,1,10000)
    result=c.advance(free,state,'hold',2,1)
    expected=float(1/(1+np.exp(-1/.5)))
    assert abs(np.mean(result['x'])-expected)<.025


def test_preferences_share_state_and_rng_and_reject_ties(monkeypatch):
    _,spec,_=problem();s=c.initial(spec,1,4);calls=[]
    def fake(spec,state,action,random_seed,steps):
        calls.append((action,random_seed,copy.deepcopy(state)))
        return state
    monkeypatch.setattr(c,'advance',fake)
    result=c.preference(spec,s,7,4,2,4)
    assert not result['accepted']
    assert all(row[2]==s for row in calls)
    assert [r[1] for r in calls[:2]]==[r[1] for r in calls[2:4]]
    assert set(r[1] for r in calls[:10]).isdisjoint(r[1] for r in calls[10:])


def test_utility_and_observations_ignore_energy_and_reference():
    _,spec,_=problem();s=c.advance(spec,c.initial(spec,1,4),'hold',2,4)
    text=c.prompt(spec,s,False);full=c.prompt(spec,s,True)
    assert 'history' not in text and 'history' in full
    altered=copy.deepcopy(s);altered['history'][-1]['energy_mean']=-1e9
    assert c.utility(spec,s)==c.utility(spec,altered)
    assert 'reference_proven' not in full and 'case_id' not in full


def test_episode_all_controls_equal_budget_and_never_mutate_model(monkeypatch):
    from experiments.solver_control_v1 import run as r
    monkeypatch.setattr(r,'CFG',dict(r.CFG,chains=4,window_steps=4,decisions=2))
    _,spec,_=problem();old=copy.deepcopy(spec)
    for arm in ('fixed','schedule','rule','random'):
        result=r.episode(spec,'fixture',arm)
        assert result['bit_updates']==48 and len(result['history'])==3
        assert spec==old


def test_real_lora_training_resume_and_dpo(tiny_engine,tmp_path,monkeypatch):
    import torch
    from experiments.solver_control_v1 import run as r
    e=tiny_engine;_,spec,_=problem();s=c.initial(spec,1,4)
    examples=[('fixture',i,spec,dict(state=s,pair=dict(chosen='heat',rejected='hold'))) for i in range(2)]
    monkeypatch.setattr(r,'RUN',tmp_path)
    monkeypatch.setattr(r,'CFG',dict(r.CFG,sft_epochs=1,dpo_epochs=1,checkpoint_steps=1))
    monkeypatch.setattr(r,'training_examples',lambda mode:examples)
    # Tiny fixture tokenizer need not support letter tokens; test same categorical head.
    monkeypatch.setattr(r,'action_ids',lambda engine:[1,2,3,4,5])
    monkeypatch.setattr(r,'prompt',lambda *a:'control')
    initial=e.snapshot();real=r.policy_logits;calls=[0]
    def interrupted(*a):
        calls[0]+=1
        if calls[0]==2:raise RuntimeError('interrupted')
        return real(*a)
    monkeypatch.setattr(r,'policy_logits',interrupted)
    with pytest.raises(RuntimeError,match='interrupted'):r.train(e,'pilot','trajectory_sft')
    assert torch.load(tmp_path/'pilot/models/trajectory_sft/checkpoint.pt',weights_only=True)['step']==1
    monkeypatch.setattr(r,'policy_logits',real)
    r.train(e,'pilot','trajectory_sft');resumed=e.snapshot()
    e.restore(initial);r.train(e,'pilot','best_sft')
    assert all(torch.equal(v,e.snapshot()[k]) for k,v in resumed.items())
    assert any(not torch.equal(v,initial[k]) for k,v in resumed.items())
    r.train(e,'pilot','trajectory_dpo')
    assert any(not torch.equal(v,resumed[k]) for k,v in e.snapshot().items())


def test_gate_uses_distinct_training_cases_not_number_of_states(tmp_path,monkeypatch):
    from experiments.solver_control_v1 import run as r
    monkeypatch.setattr(r,'RUN',tmp_path)
    monkeypatch.setattr(r,'rows',lambda *a:[dict(case=dict(id='one'))])
    save(tmp_path/'pilot/labels/one.json',dict(examples=[dict(pair=dict(accepted=True,chosen=a)) for a in ('cool','heat')]*20))
    gate=r.label_gate('pilot')
    assert gate['pairs']==40 and gate['cases']==1 and not gate['passed']


def test_best_quality_has_priority_over_population_progress():
    assert c.compare((1.,.5),(.99,1.2))==1
    assert c.compare((1.,.5),(1.,.6))==-1
    assert c.compare((1.,.5),(1.,.5))==0


def test_reference_only_evaluated_after_control_and_cached(tmp_path,monkeypatch):
    from experiments.solver_control_v1 import run as r
    from experiments.feedback_active_v3 import cpu
    _,spec,_=problem();events=[]
    monkeypatch.setattr(r,'RUN',tmp_path)
    monkeypatch.setattr(r,'rows',lambda *a:[dict(case=dict(id='one',stage='main'),gold=spec)])
    def ep(*a):
        events.append('decision')
        return dict(best=dict(x=[0]),history=[],actions=['hold'],bit_updates=4,episode_seconds=1,decision_seconds=0)
    def reference(job):
        events.append('reference');return dict(status='optimal',optimality_proven=True,objective=0)
    monkeypatch.setattr(r,'episode',ep);monkeypatch.setattr(cpu,'execute',reference)
    r.evaluate('pilot','fixed');r.evaluate('pilot','fixed')
    assert events==['decision','reference']
    result=load(tmp_path/'pilot/evaluations/fixed/one.json')
    assert result['score']['optimal_hit'] and result['stage']=='pilot'


def test_report_requires_full_coverage_and_weights_cases_equally(tmp_path,monkeypatch):
    from experiments.solver_control_v1 import run as r
    monkeypatch.setattr(r,'RUN',tmp_path)
    monkeypatch.setattr(r,'CFG',dict(r.CFG,bootstrap_repeats=10))
    monkeypatch.setattr(r,'rows',lambda *a:[dict(case=dict(id=i)) for i in ('a','b','c')])
    monkeypatch.setattr(r,'training_examples',lambda mode:[1])
    monkeypatch.setattr(r,'label_gate',lambda mode:dict(passed=True))
    save(tmp_path/'screen/label_gate.json',dict(passed=True))
    for arm in r.CFG['arms']:
        for i,cid in enumerate(('a','b','c')):
            save(tmp_path/f'screen/evaluations/{arm}/{cid}.json',dict(stage='main' if i<2 else 'scale',
                score=dict(original_feasible=True,optimal_hit=i<2,reference_proven=True),
                actions=['hold'],episode_seconds=1,decision_seconds=0))
        if arm in r.CFG['trained_arms']:
            for j in range(1 if arm=='trajectory_dpo' else 2):
                save(tmp_path/f'screen/models/{arm}/steps/{j}.json',dict(gradient_norm=1))
    result=r.report('screen')
    assert result['gate']['passed']
    assert result['summary']['fixed']['all']['optimal']==pytest.approx(2/3)
    assert result['comparisons']['trajectory_sft-best_sft']['delta']==0
    (tmp_path/'screen/evaluations/fixed/a.json').unlink()
    result=r.report('screen')
    assert not result['gate']['passed'] and not result['comparisons']
