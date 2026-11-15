import copy
import numpy as np
import torch
from experiments.pbit_joint_v3 import run as r, report, methods, audit
from experiments.pbit_joint_v2.signals import control_prompt
from qihc.experiment_runtime.storage import load,save
from test_pvcd_cycle import tiny_engine,problem


def sample(matrix,label):
    return dict(prompt='action?',returns=np.repeat(np.array(matrix)[None,:,None],4,axis=0).tolist(),
                labels={'trajectory':label})


def test_ties_are_not_preferences_and_negative_alternatives_are_retained():
    assert methods.confirmed_pairs(sample([.5,.5,.5],'cool'))==[]
    assert methods.confirmed_pairs(sample([.5,.4,.6],'clamp'))==[(0,1),(2,0)]
    evidence={'a':{'samples':[sample([.5,.5,.5],'cool')]*10},
              'b':{'samples':[sample([.5,.4,.6],'clamp')]}}
    rows,w=methods.control_data(evidence,'balanced')
    assert np.isclose(sum(w[:-1]),w[-1])
    replay=methods.balanced_replay(rows,20,17)
    assert len(replay)==20 and sum(rows[i][2]==2 for i in replay)==10


def test_four_arms_and_factorial(tiny_engine,tmp_path,monkeypatch):
    from experiments.posttraining_v1.data import target
    case,gold,wrong=problem()
    cid=case['id']
    monkeypatch.setattr(r,'RUN',tmp_path)
    monkeypatch.setattr(r,'CFG',dict(r.CFG,shards=1,epochs=1))
    monkeypatch.setattr(r.common,'action_ids',lambda engine:[1,2,3])
    evidence={cid:dict(candidate=wrong,cards=[],samples=[sample([.5,.5,.5],'cool'),sample([.5,.4,.6],'clamp')])}
    parent=tmp_path/'parent.pt'
    original=tiny_engine.snapshot()
    torch.save(original,parent)
    ready={'diagnostics':{a:dict(winners={'cool':1,'clamp':1}) for a in r.POLICIES[1:]}}
    for arm in r.ARMS:
        tiny_engine.restore(original)
        methods.train(tiny_engine,arm,tmp_path/'trained'/arm,{'adapter':str(parent)},ready,
                      [dict(case=case,gold=gold)],evidence,r.CFG)
        assert (tmp_path/'trained'/arm/'adapter.pt').exists()
    save(tmp_path/'predictions/weighted'/f'{cid}.json',dict(model=gold))
    save(tmp_path/'references'/f'{cid}.json',dict(optimality_proven=True,objective=0))
    monkeypatch.setattr(tiny_engine,'evaluate',lambda *a:dict(choices=[dict(message=dict(content=target(gold)),finish_reason='stop')]))
    for task in r.subtasks():
        assert r.eligible(task)
        tiny_engine.restore(torch.load(tmp_path/'trained'/task[2]/'adapter.pt',weights_only=True))
        r.process_shard(task,tiny_engine,ready,[case])
        save(r.marker(task),{})
    report.score_case((dict(case,stage='scale'),gold))
    scored=load(tmp_path/'scores'/f'{cid}.json')
    assert len(scored['cells'])==9 and len(report.contrasts([scored]))==4
    assert len(report.intervention_results([scored]))==6


def test_scale_audit_recovery_and_unresolved(monkeypatch):
    case,gold,wrong=problem()
    cells={m+'__schedule':dict(semantic='unknown',score=dict(optimal_hit=hit,original_feasible=True))
           for m,hit in [('base',True),('weighted',False)]}
    monkeypatch.setattr(audit,'solve',lambda *a:dict(x=[0],optimality_proven=True,status='optimal'))
    row=audit.examine('nlco_KP_L_001',{'cells':cells},{'model':gold},{'model':wrong},gold,
                      dict(optimality_proven=True,objective=0))
    assert audit.summarize([row])['lost_case_diagnostic']['weighted_SCIP_recovers_original_optimum']==1
    assert audit.summarize([row])['families']['KP']['net']==-1
