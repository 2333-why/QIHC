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
    monkeypatch.setattr(r,'rows',lambda *a:[dict(case=dict(id=i,stage='scale' if i=='c' else 'main')) for i in ('a','b','c')])
    monkeypatch.setattr(r,'training_examples',lambda mode:[1])
    monkeypatch.setattr(r,'label_gate',lambda mode:dict(passed=True,complete=True))
    save(tmp_path/'screen/label_gate.json',dict(passed=True))
    for arm in r.CFG['arms']:
        for i,cid in enumerate(('a','b','c')):
            save(tmp_path/f'screen/evaluations/{arm}/{cid}.json',dict(stage='main' if i<2 else 'scale',
                score=dict(original_feasible=True,optimal_hit=i<2,reference_proven=True),
                actions=['hold'],episode_seconds=1,decision_seconds=0,search_seconds=1))
        if arm in r.CFG['trained_arms']:
            save(tmp_path/f'screen/models/{arm}/adapter.pt',{})
            for j in range(1 if arm=='trajectory_dpo' else 2):
                save(tmp_path/f'screen/models/{arm}/steps/{j:05}.json',dict(gradient_norm=1,loss=1))
    result=r.report('screen')
    assert result['gate']['passed']
    assert result['summary']['fixed']['all']['optimal']==pytest.approx(2/3)
    assert result['comparisons']['trajectory_sft-best_sft']['delta']==0
    (tmp_path/'screen/evaluations/fixed/a.json').unlink()
    result=r.report('screen')
    assert not result['gate']['passed'] and not result['comparisons']


def test_full_preparation_uses_all_training_and_780_test_without_pilot(tmp_path,monkeypatch):
    from experiments.solver_control_v1 import run as r
    monkeypatch.setattr(r,'RUN',tmp_path)
    save(tmp_path/'inputs/train.json',[dict(case=dict(id=f't{i}')) for i in range(90)])
    save(tmp_path/'inputs/test.json',[dict(case=dict(id=f'e{i}',stage='main' if i<480 else 'scale')) for i in range(780)])
    assert len(r.rows('full',True))==90
    assert len(r.rows('full'))==780
    assert sum(x['case']['stage']=='main' for x in r.rows('full'))==480
    assert not (tmp_path/'inputs/validation.json').exists()
    monkeypatch.setattr(r,'validate',lambda:None)
    monkeypatch.setattr(r,'report',lambda mode:pytest.fail('Full must not request pilot report'))
    monkeypatch.setattr(r,'label_gate',lambda mode:dict(passed=True))
    seen=[]
    class Pool:
        def __init__(self,**kw):pass
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def map(self,fn,jobs):
            seen.extend(jobs);return []
    monkeypatch.setattr(r,'ProcessPoolExecutor',Pool)
    r.make_labels('full')
    assert len(seen)==90 and all(mode=='full' for _,mode in seen)
    with pytest.raises(BaseException,match='pilot report'):r.make_labels('screen')


def test_partial_full_report_is_repeatable_and_cannot_claim_complete(tmp_path,monkeypatch):
    from experiments.solver_control_v1 import run as r
    monkeypatch.setattr(r,'RUN',tmp_path)
    save(tmp_path/'inputs/train.json',[dict(case=dict(id=f't{i}')) for i in range(90)])
    save(tmp_path/'inputs/test.json',[dict(case=dict(id=f'e{i}',stage='main' if i<480 else 'scale')) for i in range(780)])
    for _ in range(2):
        result=r.report('full')
        assert result['expected_evaluations']==5460 and result['completed_evaluations']==0
        assert not result['gate']['passed'] and not result['comparisons']


def test_training_gate_requires_artifact_exact_steps_and_finite_loss(tmp_path,monkeypatch):
    from experiments.solver_control_v1 import run as r
    monkeypatch.setattr(r,'RUN',tmp_path)
    monkeypatch.setattr(r,'training_examples',lambda mode:[1])
    out=tmp_path/'full/models/trajectory_sft'
    for i in range(2):save(out/f'steps/{i:05}.json',dict(gradient_norm=1,loss=1))
    assert not r.training_complete('full','trajectory_sft')
    save(out/'adapter.pt',{})
    assert r.training_complete('full','trajectory_sft')
    (out/'steps/00001.json').write_text('{"gradient_norm": 0, "loss": NaN}',encoding='utf8')
    assert not r.training_complete('full','trajectory_sft')


def test_full_gpu_dispatch_and_resume_do_not_depend_on_pilot(tmp_path,monkeypatch):
    import torch
    from types import SimpleNamespace
    from contextlib import nullcontext
    from experiments.solver_control_v1 import run as r
    from experiments.pvcd_cycle_v1 import launch
    monkeypatch.setattr(r,'RUN',tmp_path)
    monkeypatch.setattr(r,'validate',lambda:None)
    monkeypatch.setattr(r,'label_gate',lambda mode:dict(passed=True))
    def report(mode):
        assert mode=='full';return dict(gate=dict(passed=True))
    monkeypatch.setattr(r,'report',report)
    monkeypatch.setattr(torch.cuda,'device_count',lambda:2)
    monkeypatch.setattr(torch.cuda,'device',lambda i:nullcontext())
    monkeypatch.setattr(torch.cuda,'get_device_properties',lambda i:SimpleNamespace(name='fake',total_memory=80*1024**3))
    monkeypatch.setattr(torch.cuda,'is_bf16_supported',lambda:True)
    monkeypatch.setattr(torch.cuda,'synchronize',lambda i:None)
    real_ones=torch.ones
    monkeypatch.setattr(torch,'ones',lambda *a,**kw:real_ones(*a,**dict(kw,device='cpu')))
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','2,3')
    calls=[]
    class Proc:
        returncode=0
        def __init__(self,args,**kw):calls.append((args,kw['env']['CUDA_VISIBLE_DEVICES']))
        def poll(self):return 0
    monkeypatch.setattr(r.subprocess,'Popen',Proc)
    monkeypatch.setattr(launch,'stop_process',lambda p:None)
    r.launch('full')
    assert len(calls)==7 and all(args[-2]=='full' for args,gpu in calls)
    assert [(args[-1],gpu) for args,gpu in calls if gpu]==[('best_sft','2'),('trajectory_sft','3'),('trajectory_dpo','2')]
    for arm in r.CFG['arms']:save(tmp_path/f'full/models/{arm}/done.json',dict(cases=780))
    calls.clear();r.launch('full');assert not calls
    (tmp_path/'full/models/best_sft/done.json').unlink()
    class Failed(Proc):
        returncode=1
        def __init__(self,args,**kw):
            super().__init__(args,**kw)
            kw['stdout'].write('OverflowError: value too large to convert to int\n')
        def poll(self):return 1
    monkeypatch.setattr(r.subprocess,'Popen',Failed)
    with pytest.raises(RuntimeError,match='(?s)best_sft: exit=1.*OverflowError'):
        r.launch('full')


def test_expanded_labels_have_16_independent_states_and_resume(tmp_path,monkeypatch):
    from experiments.solver_control_v1 import run as r
    _,spec,_=problem()
    monkeypatch.setattr(r,'RUN',tmp_path)
    monkeypatch.setattr(r,'CFG',dict(r.CFG,chains=4,window_steps=2,training_temperatures=[.5,.25,1.,2.]))
    starts=[]
    real_initial=r.initial
    def initial(spec,seed,chains):
        starts.append(seed);return real_initial(spec,seed,chains)
    monkeypatch.setattr(r,'initial',initial)
    seen=[]
    def preference(spec,state,seed,*args):
        seen.append(seed);return dict(accepted=True,chosen='cool',rejected='hold')
    monkeypatch.setattr(r,'preference',preference)
    job=(dict(case=dict(id='fixture'),gold=spec),'full')
    result=r.label_one(job)
    assert len(result['examples'])==16 and len(set(starts))==4 and len(set(seen))==16
    assert [e['state']['temperature'] for e in result['examples'][::4]]==[.5,.25,1.,2.]
    assert r.label_one(job)==result and len(seen)==16
    assert r.CFG['decisions']==4


@pytest.mark.parametrize('raw',[0,17,2147483647,4045664162,4294967295])
def test_reference_seed_is_signed32_without_changing_search(tmp_path,monkeypatch,raw):
    from experiments.solver_control_v1 import run as r
    from experiments.feedback_active_v3 import cpu
    _,spec,_=problem()
    monkeypatch.setattr(r,'RUN',tmp_path)
    monkeypatch.setattr(r,'rows',lambda *a:[dict(case=dict(id='one',stage='main'),gold=spec)])
    monkeypatch.setattr(r,'seed',lambda *a:raw)
    monkeypatch.setattr(r,'episode',lambda *a:dict(best=dict(x=[0]),actions=['hold']))
    def reference(job):
        assert job['seed']==(raw & 0x7fffffff)
        assert 0<=job['seed']<=2147483647
        return dict(status='optimal',optimality_proven=True,objective=0)
    monkeypatch.setattr(cpu,'execute',reference)
    r.evaluate('full','fixed')


def test_seed_fix_migration_is_narrow_atomic_and_preserves_artifacts(tmp_path,monkeypatch):
    import hashlib
    from experiments.solver_control_v1 import run as r
    key='experiments/solver_control_v1/run.py'
    h=lambda b:hashlib.sha256(b).hexdigest()
    cfg=dict(r.CFG,experiment_id='E005-1.12')
    old=dict(experiment_id='E005-1.12',config=cfg,code_sha='original',sources={key:h(b'old'),'other.py':'unchanged'})
    new={key:h(b'new'),'other.py':'unchanged'}
    monkeypatch.setattr(r,'RUN',tmp_path);monkeypatch.setattr(r,'CFG',cfg)
    monkeypatch.setattr(r,'sources',lambda:new)
    def command(args,**kw):
        if args[1]=='rev-parse':return 'newsha\n'
        return b'old' if args[2].startswith('7fb82f7f') else b'new'
    monkeypatch.setattr(r.subprocess,'check_output',command)
    save(tmp_path/'manifest.json',old)
    save(tmp_path/'full/models/best_sft/checkpoint.pt',{'sentinel':17})
    def invalid(m=None):raise ValueError('Environment changed')
    monkeypatch.setattr(r,'validate',invalid)
    with pytest.raises(ValueError,match='Environment'):r.migrate_reference_seed_fix()
    assert load(tmp_path/'manifest.json')==old
    assert not (tmp_path/'manifest.before_reference_seed_fix.json').exists()
    monkeypatch.setattr(r,'validate',lambda m=None:m)
    new['other.py']='changed'
    with pytest.raises(ValueError,match='Unrelated'):r.migrate_reference_seed_fix()
    new['other.py']='unchanged'
    r.migrate_reference_seed_fix();r.migrate_reference_seed_fix()
    assert load(tmp_path/'manifest.before_reference_seed_fix.json')==old
    assert load(tmp_path/'manifest.json')['code_sha']=='newsha'
    assert load(tmp_path/'full/models/best_sft/checkpoint.pt')=={'sentinel':17}
