import numpy as np
import pytest
from test_pvcd_cycle import tiny_engine, problem
from experiments.pbit_joint_v2 import run as r, report as reporting
from experiments.pbit_joint_v2.signals import action_labels, pair_loss
from qihc.experiment_runtime.storage import load, save


def test_early_gain_and_tie():
    values = np.zeros((4,3,4))
    values[:,:,3]=1
    values[:,1,1:]=1
    assert action_labels(values)==dict(terminal='cool',trajectory='heat')
    values[0,1,:]=-1
    assert action_labels(values)['trajectory']=='cool'


def test_pair_gradient_and_common_prefix(tiny_engine):
    import torch
    loss=pair_loss(tiny_engine,'rank',[2,3,4],[2,3,5])
    loss.backward()
    assert torch.isfinite(loss) and any(p.grad is not None and p.grad.abs().sum()>0 for p in tiny_engine.params.values())


def test_prepare_train_shards_score(tmp_path, tiny_engine, monkeypatch):
    import torch
    from experiments.posttraining_v1.data import target
    from experiments.feedback_active_v3 import cpu
    case,gold,wrong=problem()
    monkeypatch.setattr(r,'RUN',tmp_path)
    monkeypatch.setattr(r,'CFG',dict(r.CFG,shards=1,epochs=1,temperatures=[.5],decisions=1,window_steps=2,probe_steps=4))
    save(tmp_path/'inputs/train.json',[dict(case=case,gold=gold)])
    save(tmp_path/'manifest.json',{})
    r.prepare_case((dict(case=case,gold=gold),wrong))
    cid=case['id']
    evidence=tmp_path/'evidence'/f'{cid}.json'
    assert load(evidence)['cards']
    parent=tmp_path/'parent.pt'
    torch.save(tiny_engine.snapshot(),parent)
    pred=tmp_path/'base.json'
    save(pred,dict(model=wrong))
    manifest=dict(adapter=str(parent),predictions={cid:str(pred)})
    ready=dict(files={cid:r.digest(evidence)})
    original=tiny_engine.snapshot()
    monkeypatch.setattr(tiny_engine,'evaluate',lambda *args:dict(choices=[dict(finish_reason='stop',message=dict(content=target(gold)))]))
    # Tiny BPE fixture need not have distinct single-token A/B/C.
    monkeypatch.setattr(r.v1,'action_ids',lambda e:[1,2,3])
    for arm in r.ARMS:
        tiny_engine.restore(original)
        r.train(tiny_engine,arm,manifest,ready)
    assert load(tmp_path/'trained/paired/steps/00000.json')['pair']
    for task in r.tasks():
        assert r.eligible(task)
        tiny_engine.restore(torch.load(tmp_path/'trained'/task[2]/'adapter.pt',weights_only=True))
        r.run_shard(task,[case],tiny_engine,manifest)
        save(r.marker(task),{})
    monkeypatch.setattr(cpu,'execute',lambda *a:dict(status='optimal',optimality_proven=True,objective=0))
    reporting.score_case((dict(case,stage='main'),gold),manifest['predictions'])
    row=load(tmp_path/'scores'/f'{cid}.json')
    stats=reporting.summarize([row])['all']
    assert len(stats)==12 and all(s['n']==1 for s in stats.values())
