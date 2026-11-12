import copy
from types import SimpleNamespace
import pytest
import torch
from test_pvcd_cycle import tiny_engine,problem
from qihc.experiment_runtime.storage import load,save
from experiments.posttraining_v1.data import target
from experiments.pvcd_verified_v1.feedback import explain,remaining,prompt
from experiments.pvcd_verified_v1.run import CFG

def test_explained_order_minimized_witness_and_index_contract():
    case,gold,wrong=problem()
    cards=[dict(kind='feasibility',x=[1],correct_feasible=False,candidate_feasible=True)]
    feedback=explain(wrong,gold,cards)
    assert feedback[0]['x']==[1] and 'hard constraint' in feedback[0]['meaning']
    assert remaining(gold,gold,feedback)==0 and remaining(wrong,gold,feedback)==1
    reverse=copy.deepcopy(wrong);reverse['objective']['sense']='min'
    obj=explain(reverse,wrong,[dict(kind='objective',x=[1],y=[0],gold_order=-1,model_order=1)])
    assert obj[0]['meaning']=='x must be preferred to y'
    assert obj[0]['raw_objective_values']['reference']=={'x':1.,'y':0.}
    assert 'zero-based' in prompt('base','attempt',wrong,obj)

def test_control_matches_original_gradients_and_positives_train(tiny_engine,tmp_path,monkeypatch):
    from experiments.pvcd_verified_v1 import training as t
    from experiments.posttraining_v1 import learn
    case,gold,wrong=problem();row=dict(case=case,gold=gold,answer=target(gold));e=tiny_engine;ref=e.snapshot()
    cards=[dict(kind='feasibility',x=[1],correct_feasible=False,candidate_feasible=True)]
    monkeypatch.setattr(t,'probe',lambda *a,**kw:(cards,[]));monkeypatch.setattr(learn,'probe',lambda *a,**kw:(cards,[]))
    def sample(p,s):
        text=target(gold if ('Feedback:' in p or 'TRAINING REPAIR' in p or s%2) else wrong)
        return e.encode(text),text
    monkeypatch.setattr(e,'sample',sample)
    for p in e.params.values():p.grad=None
    learn.train_one(e,'pvcd',row,17,ref,tmp_path,0)
    expected={k:p.grad.clone() for k,p in e.params.items()}
    for arm in ('pvcd','positive','localized'):
        e.restore(ref)
        for p in e.params.values():p.grad=None
        out=t.step(e,arm,row,17,ref,CFG)
        assert out['positive_candidates']==2 and out['verified_repairs']==2
        assert out['positive_used']==(arm!='pvcd')
        if arm=='pvcd':assert all(torch.allclose(p.grad,expected[k],atol=1e-6) for k,p in e.params.items())
        else:assert any(not torch.allclose(p.grad,expected[k]) for k,p in e.params.items())

def test_unknown_never_used_as_positive(tiny_engine,monkeypatch):
    from experiments.pvcd_verified_v1 import training as t
    case,gold,_=problem();e=tiny_engine;row=dict(case=case,gold=gold,answer=target(gold))
    monkeypatch.setattr(t,'equivalent',lambda *a:dict(status='unknown'))
    monkeypatch.setattr(t,'probe',lambda *a,**kw:([],[]))
    monkeypatch.setattr(e,'sample',lambda *a:(e.encode(target(gold)),target(gold)))
    result=t.step(e,'positive',row,17,e.snapshot(),CFG)
    assert not result['positive_used'] and result['positive_candidates']==0

def test_new_worker_checkpoint_resume(tiny_engine,tmp_path,monkeypatch):
    from qihc.experiment_runtime import posttrain_screen as r
    from experiments.posttraining_v1 import engine as em
    case,gold,_=problem();e=tiny_engine;initial=e.snapshot()
    adapter=tmp_path/'sft.pt';torch.save(initial,adapter)
    save(tmp_path/'manifest.json',dict(adapter=str(adapter),model='tiny'))
    row=dict(case=case,gold=gold,answer=target(gold));cfg=dict(CFG,checkpoint_steps=1)
    calls=[0];interrupt=[True]
    def step(eng,*a):
        calls[0]+=1
        if interrupt[0] and calls[0]==2:raise RuntimeError('interrupted')
        eng.backward('fixture',eng.encode(target(gold)),'nll')
        return dict(positive_used=True,verified_repairs=0)
    job=SimpleNamespace(RUN=tmp_path,CFG=cfg,step=step)
    monkeypatch.setattr(em,'Engine',lambda *a:e)
    monkeypatch.setattr(r,'train_list',lambda *a:[row,row]);monkeypatch.setattr(r,'case_list',lambda *a:[])
    with pytest.raises(RuntimeError,match='interrupted'):r.worker(job,'pilot','positive')
    out=tmp_path/'pilot/models/17/positive'
    assert torch.load(out/'checkpoint.pt',weights_only=True)['step']==1
    interrupt[0]=False;r.worker(job,'pilot','positive');resumed=e.snapshot()
    r.worker(job,'pilot','localized')
    assert all(torch.equal(v,resumed[k]) for k,v in e.snapshot().items())

def test_reporting_gate_does_not_claim_repairs_when_only_positives(tmp_path,monkeypatch):
    from experiments.pvcd_verified_v1 import report as r
    case,gold,_=problem();job=SimpleNamespace(RUN=tmp_path,CFG=dict(CFG,bootstrap_repeats=10))
    monkeypatch.setattr(r,'case_list',lambda *a:[dict(id='a',stage='pilot'),dict(id='b',stage='pilot')])
    monkeypatch.setattr(r,'train_list',lambda *a:[1,2])
    for arm in CFG['arms']:
        base=tmp_path/f'pilot/models/17/{arm}'
        for i,cid in enumerate(('a','b')):
            save(base/'evaluations'/(cid+'.json'),dict(semantic=dict(status='verified_correct'),cross={b:dict(score=dict(original_feasible=True,optimal_hit=True)) for b in ('scip','pbit')}))
            if arm!='sft':
                save(base/'trace'/f'{i}.json',dict(case_id=cid,positive_used=arm!='pvcd',positive_candidates=1,reward_update=False,verified_repairs=0,samples=[]))
                save(base/'steps'/f'{i}.json',dict(gradient_norm=1,seconds=1))
    out=r.build_report(job,'pilot')
    assert out['gate']['passed'] and not out['gate']['repair_active']
    (tmp_path/'pilot/models/17/pvcd/evaluations/a.json').unlink()
    out=r.build_report(job,'pilot')
    assert not out['complete'] and not out['gate']['passed']
    assert not out['stages']['pilot']['comparisons']

def test_cpu_prepare_from_realistic_previous_layout_and_score(tmp_path,monkeypatch):
    from qihc.experiment_runtime import posttrain_screen as r
    from qihc.experiment_runtime.storage import digest
    from experiments.pvcd_cycle_v1 import common
    from experiments.feedback_active_v3 import check_local_model
    case,gold,_=problem();old=tmp_path/'previous';new=tmp_path/'new';history=tmp_path/'history'
    adapter=history/'formal/models/17/sft/adapter.pt';adapter.parent.mkdir(parents=True);adapter.write_bytes(b'fixture')
    train=[dict(case=dict(case,id=f't{i}'),gold=gold,answer=target(gold)) for i in range(90)]
    val=[dict(case=dict(case,id=f'v{i}'),gold=gold,answer=target(gold)) for i in range(24)]
    cohort=[dict(row['case'],stage='development') for row in train+val]
    cohort += [dict(case,id=f'd{i}',stage='development') for i in range(6)]
    cohort += [dict(case,id=f'c{i}',stage='main') for i in range(480)]
    cohort += [dict(case,id=f's{i}',stage='scale') for i in range(300)]
    cohort += [dict(case,id=f'p{i}',stage='public') for i in range(10)]
    for name,obj in (('train',train),('validation',val),('cohort',cohort)):save(old/f'inputs/{name}.json',obj)
    frozen={str(adapter):digest(adapter)};predictions={}
    for c in cohort:
        path=adapter.parent/'predictions'/(c['id']+'.json');save(path,dict(model=gold))
        frozen[str(path)]=digest(path);predictions[str(path)]=True
    paths=[old/'data/evaluation/cases.gold.json',old/'data/evaluation/cases.public_external.gold.json']
    save(paths[0],[dict(id='c0',gold=gold,family='fixture')]);save(paths[1],[])
    save(old/'manifest.json',dict(experiment_id='E005-1.9',history=str(history),model=str(tmp_path/'model'),fingerprints=[],
        donor_files=frozen,donor_predictions=predictions,data_files={str(p):digest(p) for p in paths},
        input_files={f'inputs/{name}.json':digest(old/f'inputs/{name}.json') for name in ('train','validation','cohort')}))
    job=SimpleNamespace(RUN=new,PREVIOUS=old,HERE=tmp_path/'repo/experiments/new',CFG=CFG)
    monkeypatch.setattr(r,'source_hashes',lambda *a:{'fixture.py':'hash'})
    monkeypatch.setattr(r,'model_identity',lambda *a:[])
    monkeypatch.setattr(common,'environment_identity',lambda:{'python':'test'})
    monkeypatch.setattr(check_local_model,'check',lambda *a:None)
    monkeypatch.setattr(r.subprocess,'check_output',lambda args,**kw:'sha\n' if 'rev-parse' in args else '')
    r.prepare(job);first=load(new/'manifest.json');r.prepare(job)
    assert first==load(new/'manifest.json');r.validate(job)
    assert (new/'inputs/train.json').read_bytes()==(old/'inputs/train.json').read_bytes()
    p=new/'screen/models/17/positive/predictions/c0.json';save(p,dict(case_id='c0',stage='main',model=gold))
    r.score_one((str(new),dict(CFG,solver_seconds=.2),str(p),[str(p) for p in paths]))
    result=load(p.parent.parent/'evaluations/c0.json')
    assert all(result['cross'][b]['score']['optimal_hit'] for b in ('scip','pbit'))
    (new/'inputs/train.json').write_text('[]')
    with pytest.raises(ValueError,match='Input snapshot changed'):r.validate(job)
