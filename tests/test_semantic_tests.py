import copy
import itertools
import json
import numpy as np
import pytest
from experiments.semantic_tests_v1 import policy, audit
from qihc.modeling.active_protocol_v2 import compact
from qihc.experiment_runtime.storage import save

def fixture():
    model=dict(schema='bounded-integer-v2',variables=[dict(name=f'x{i}',lower=0,upper=1) for i in range(4)],
               objective=dict(sense='max',linear=[1,2,3,4],quadratic=[]),constraints=[],logic=[])
    fixed=copy.deepcopy(model)
    fixed['constraints']=[dict(id='capacity',a=[1,1,1,1],op='le',rhs=2)]
    case=dict(id='case',question='Choose at most two items. Maximize the stated reward.',variable_names=[f'x{i}' for i in range(4)])
    item=dict(case=case,stage='development',seed=17,model=model,baseline=model,reviewed=model)
    return item,fixed

def test_hypotheses_cover_all_supported_blocks_and_preserve_domains():
    item,fixed=fixture()
    fixed['objective']['sense']='min'
    fixed['logic']=[dict(id='l',expr={'var':0})]
    base=policy.shared(item,lambda *_:json.dumps({'model':compact(fixed)}))
    assert {h['block'] for h in base['hypotheses']}=={'objective','constraints','logic'}
    for h in base['hypotheses']:
        assert sum(h['model'][k]!=item['reviewed'][k] for k in ('objective','constraints','logic'))==1
        assert h['model']['variables']==item['reviewed']['variables']
    fixed['variables'][0]['upper']=2
    fixed['logic']=[]
    base=policy.shared(item,lambda *_:json.dumps({'model':compact(fixed)}))
    assert base['action']=='candidate_domain_change' and base['hypotheses']==[]

def test_generic_arithmetic_cards():
    item,fixed=fixture();old=item['reviewed']
    test=policy.card(old,fixed,[[1,1,1,0]])
    assert test['kind']=='feasibility' and test['predictions']==[True,False]
    alternative=copy.deepcopy(old);alternative['objective']['sense']='min'
    test=policy.card(old,alternative,[[0,0,0,0],[1,0,0,0]])
    assert test['kind']=='objective_order' and test['predictions']==[1,-1]
    alternative=copy.deepcopy(old);alternative['logic']=[dict(id='l',expr={'var':0})]
    assert policy.card(old,alternative,[[0,0,0,0]])['predictions']==[True,False]

def test_local_protocol_and_position_are_strict():
    item,fixed=fixture()
    for flip in (False,True):
        text=json.dumps(dict(choice=int(not flip),evidence=[0],reason='The constraint is stated.'))
        assert policy.parse_review(text,item['case'],flip)['choice']==1
    for bad in (dict(choice=True,evidence=[0],reason='x'),dict(choice=1,evidence=[99],reason='x'),dict(choice=1,evidence=[],reason='x')):
        assert policy.parse_review(json.dumps(bad),item['case'],False)['status']=='review_parse_error'
    assert policy.parse_review('{"choice":null,"evidence":[],"reason":"uncertain"}',item['case'],False)['status']=='abstain'

def test_fresh_test_and_second_review_veto(monkeypatch):
    item,fixed=fixture()
    base=policy.shared(item,lambda *_:json.dumps({'model':compact(fixed)}))
    states=list(itertools.product([0,1],repeat=4))
    monkeypatch.setattr(policy,'collect',lambda *a:(states,dict(seconds=.01,pooled_states=len(states))))
    calls=[]
    def choose(prompt,seed):
        options=json.loads(prompt.split('\nInterpretations 0 and 1: ')[1].split('\nComputed contrast')[0])
        calls.append(options)
        return json.dumps(dict(choice=options.index(compact(fixed)['constraints']),evidence=[0],reason='At most two.'))
    row=policy.decide(item,'Q',base,choose)
    assert row['action']=='accepted' and len(calls)==2
    assert calls[0]==calls[1][::-1]
    assert row['evidence'][0]['test']['assignments']!=row['evidence'][1]['test']['assignments']
    row.update(case_id='case')
    assert audit.audit(row,fixed)['wrong_labels']==0
    assert audit.audit(row,item['reviewed'])['wrong_labels']==2
    calls.clear()
    def veto(prompt,seed):
        good=json.loads(choose(prompt,seed))
        if len(calls)==2:good['choice']=1-good['choice']
        return json.dumps(good)
    assert policy.decide(item,'Q',base,veto)['action']=='review_keep'
    monkeypatch.setattr(policy,'collect',lambda *a:([[1,1,1,0]],dict(seconds=.01)))
    assert policy.decide(item,'Q',base,choose)['action']=='no_fresh_test'

def test_real_search_budgeted_kernels():
    item,fixed=fixture()
    for arm in ('S','P','Q'):
        states,stats=policy.collect([item['reviewed'],fixed],arm,17,.2)
        assert states and stats['seconds']>0
        assert 0<=policy.contrast_score(item['reviewed'],fixed,states)<=2

def test_gate_attempt_denominator(tmp_path,monkeypatch):
    from experiments.semantic_tests_v1 import gate
    monkeypatch.setattr(gate,'RUN',tmp_path)
    save(tmp_path/'pilot/summary.json',dict(complete=True,expected=168))
    save(tmp_path/'pilot/models/17/Q/predictions/a.json',dict(evidence=[dict(verdict=dict(status='review_parse_error'))]))
    with pytest.raises(RuntimeError,match='1/1'):
        gate.check()

def test_report_partial_denominators(tmp_path,monkeypatch,capsys):
    from experiments.semantic_tests_v1 import data, report
    item,_=fixture()
    save(tmp_path/'inputs.json',[item,dict(item,case=dict(item['case'],id='pending'))])
    for a in data.ARMS:
        score=dict(optimal_hit=True,original_feasible=True)
        save(tmp_path/f'formal/models/17/{a}/evaluations/case.json',dict(case_id='case',seed=17,stage='development',
             arm=a,action='accepted',semantic='verified_correct',scores={'20':dict(pbit=score,scip=score)}))
    monkeypatch.setattr(data,'RUN',tmp_path);monkeypatch.setattr(report,'RUN',tmp_path)
    result=report.report('formal')
    assert result['expected']==14 and result['completed']==7 and not result['complete']
    assert '1/2 (50.00%)' in capsys.readouterr().out
