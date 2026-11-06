import copy
import json
import pytest
from qihc.modeling import active_protocol as old, active_protocol_v2 as new
from experiments.feedback_active_v3.data import make_case, public_view


def test_empty_format_uniform_and_history_unchanged():
    case=make_case('KP',8,17); obj={'model':old.compact(case['gold'])}
    obj['model']['objective'].pop('quadratic');obj['model']['logic']={}
    text=json.dumps(obj)
    with pytest.raises(ValueError): old.parse_model(text,public_view(case))
    assert new.parse_model(text,public_view(case))==case['gold']
    patch=json.dumps({'edits':[{'target':'objective','value':obj['model']['objective']}]})
    assert new.guard(public_view(case),case['gold'],patch)['status']=='accepted'
    with pytest.raises(ValueError):new.parse_model('{"model":{},"model":{}}',public_view(case))


def test_checks_budget_and_bad_entry_isolated():
    case=make_case('KP',8,17);c=dict(case['gold']['constraints'][0],kind='constraint',quote='Choose items',confidence=.8)
    rows=[dict(c,id=str(i)) for i in range(20)];rows[0]=None
    result=new.parse_checks(json.dumps({'checks':rows}),public_view(case))
    assert len(result['checks'])==15 and result['budget_dropped']==4
    assert result['rejected'] and all(not c['trusted'] for c in result['checks'])
    assert result['checks'][-1]['id']=='15'
    with pytest.raises(ValueError):new.parse_checks('{"checks":{}}',public_view(case))


def test_development_gate(tmp_path):
    from experiments.feedback_active_v4.gate import check
    from qihc.experiment_runtime.storage import save,load
    data=tmp_path/'data';run=tmp_path/'run'
    save(data/'cases.public.json',[dict(id='a',split='development')])
    save(run/'initial/a.json',dict(case_id='a',model=None,checks={'checks':[]}))
    cfg={'development_gate':{'compiled_min':.6,'checks_min':.5}}
    with pytest.raises(RuntimeError):check(run,data,cfg)
    assert not load(run/'development_gate.json')['passed']
    save(run/'initial/a.json',dict(case_id='a',model={},checks={'checks':[{}]}))
    check(run,data,cfg)
    assert load(run/'development_gate.json')['passed']
