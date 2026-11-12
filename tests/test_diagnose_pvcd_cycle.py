import importlib.util
import json
from pathlib import Path
import pytest

spec=importlib.util.spec_from_file_location('diagnose',Path(__file__).resolve().parents[1]/'tools/diagnose_pvcd_cycle.py')
diagnose=importlib.util.module_from_spec(spec);spec.loader.exec_module(diagnose)

def test_failure_funnel_exports_chains_without_modifying_inputs(tmp_path):
    def save(path,obj):
        path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(obj),encoding='utf-8')
    save(tmp_path/'inputs/train.json',[dict(case=dict(id='a'),gold={},answer='target')])
    rounds=[dict(round=0,status='generated',cards=[{'kind':'feasibility'}],reply='broken',tokens=4096,
                 semantic=dict(status='invalid'),error='parse'),
            dict(round=1,status='generated',cards=[{'kind':'feasibility'}],reply='different',tokens=10,
                 semantic=dict(status='unknown',reason='enumeration_limit_no_counterexample'))]
    path=tmp_path/'pilot/models/17/cycle/trace/0000.json'
    save(path,dict(case_id='a',verified_repairs=0,reward_update=True,guard=False,
                  samples=[dict(text='original',semantic=dict(status='disproved'),repair_trace=rounds)]))
    before={str(p):p.read_bytes() for p in tmp_path.rglob('*.json')}
    result=diagnose.collect(tmp_path)
    c=result['arms']['cycle']['counts']
    assert c['generated']==2 and c['repair:invalid']==1 and c['repair:unknown']==1
    assert c['at_4096_token_limit']==1 and c['candidates_with_evidence']==1
    assert len(result['examples'])==2 and result['examples'][0]['repair_trace']==rounds
    assert before=={str(p):p.read_bytes() for p in tmp_path.rglob('*.json')}
    save(tmp_path/'inputs/train.json',[])
    with pytest.raises(ValueError,match='not a training case'):diagnose.collect(tmp_path)

def test_missing_traces_is_not_reported_as_zero_repairs(tmp_path):
    (tmp_path/'inputs').mkdir();(tmp_path/'inputs/train.json').write_text('[]')
    with pytest.raises(ValueError,match='No training traces'):diagnose.collect(tmp_path)
