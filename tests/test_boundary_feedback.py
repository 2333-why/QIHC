import copy
import itertools
import json
import numpy as np
from experiments.boundary_feedback_v1 import policy
from experiments.boundary_feedback_v1.audit import audit
from qihc.experiment_runtime.storage import save

def fixture():
    old = dict(schema='bounded-integer-v2', variables=[dict(name=f'x{i}', lower=0, upper=1) for i in range(4)],
               objective=dict(sense='max', linear=[1, 2, 3, 4], quadratic=[]), constraints=[], logic=[])
    fixed = copy.deepcopy(old)
    fixed['constraints'] = [dict(id='capacity', a=[1, 1, 1, 1], op='le', rhs=2)]
    item = dict(case=dict(id='fixture', question='Choose at most two items. Maximize x0+2*x1+3*x2+4*x3.', variable_names=[f'x{i}' for i in range(4)]),
                model=old, baseline=old, seed=17, stage='development')
    return item, dict(model=old, candidate=fixed, action='candidate_ready')

def judge(gold, prompt):
    xs = json.loads(prompt.split('\nAssignments: ')[1].split('\nPairs')[0])
    pairs = json.loads(prompt.split('\nPairs (assignment indices): ')[1].split('\nAssignment IDs:')[0])
    feasible, costs = policy.signatures(gold, xs)
    return json.dumps(dict(feasible=feasible.tolist(), orders=[int(np.sign(costs[i]-costs[j])) if feasible[i] and feasible[j] else None for i, j in pairs], quote='Choose at most two items.'))

def test_real_search_all_kernels_and_blind_prompt():
    item, base = fixture()
    original = copy.deepcopy(base)
    for arm in ('S', 'P', 'Q'):
        states, stats = policy.collect([base['model'], base['candidate']], arm, 17, .25)
        assert states and stats['seconds'] > 0
        xs, pairs = policy.witnesses([base['model'], base['candidate']], states)
        prompt = policy.judge_prompt(item['case'], xs, pairs)
        assert 'DRAFT' not in prompt and 'schema' not in prompt and 'gold' not in prompt
        verdict = policy.assess(judge(base['candidate'], prompt), item['case'], [base['model'], base['candidate']], xs, pairs)
        assert verdict['status'] == 'judged' and verdict['errors'][1] == 0
    assert base == original

def test_accept_requires_fresh_independent_judgment(monkeypatch):
    item, base = fixture()
    states = list(itertools.product([0, 1], repeat=4))
    monkeypatch.setattr(policy, 'collect', lambda *args: (states, dict(seconds=.1, windows=[])))
    row = policy.decide(item, 'Q', base, lambda prompt, seed: judge(base['candidate'], prompt))
    assert row['action'] == 'accepted'
    first, second = [set(map(tuple, e['assignments'])) for e in row['evidence']]
    assert first.isdisjoint(second)
    assert audit(row, base['candidate'])['wrong_labels'] == 0
    assert audit(row, base['model'])['wrong_labels'] > 0
    # A second independent judgment can veto a discovery-stage approval.
    calls = []
    def reversed_judge(prompt, seed):
        calls.append(seed)
        result = json.loads(judge(base['candidate'], prompt))
        if len(calls) == 2:
            result['feasible'][0] = not result['feasible'][0]
            result['orders'] = [None]*len(result['orders'])
        return json.dumps(result)
    row = policy.decide(item, 'Q', base, reversed_judge)
    assert len(calls) == 2
    assert row['action'] == 'candidate_contradicted'

def test_unknown_malformed_and_wrong_candidate_are_distinct(monkeypatch):
    item, base = fixture()
    states = list(itertools.product([0, 1], repeat=4))
    monkeypatch.setattr(policy, 'collect', lambda *args: (states, dict(seconds=.1, windows=[])))
    assert policy.decide(item, 'P', base, lambda *_:'{"keep":true}')['action'] == 'judge_parse_error'
    def unknown(prompt, seed):
        o = json.loads(judge(base['candidate'], prompt))
        o['feasible'] = [None]*len(o['feasible'])
        o['orders'] = [None]*len(o['orders'])
        return json.dumps(o)
    assert policy.decide(item, 'P', base, unknown)['action'] == 'judge_unknown'
    assert policy.decide(item, 'P', base, lambda p, s:judge(base['model'], p))['action'] == 'candidate_contradicted'
    # Strict labels reject bool-as-int and unsupported confidence wrappers.
    xs, pairs = policy.witnesses([base['model'], base['candidate']], states)
    text = json.loads(judge(base['candidate'], policy.judge_prompt(item['case'], xs, pairs)))
    text['orders'][0] = True
    assert policy.assess(json.dumps(text), item['case'], [base['model'], base['candidate']], xs, pairs)['status'] == 'judge_parse_error'

def test_shared_parser_and_baseline_reuse():
    item, base = fixture()
    ans = json.dumps(dict(model=policy.compact(base['candidate'])))
    shared = policy.shared(item, lambda *_:ans)
    assert shared['candidate'] == base['candidate']
    assert policy.parse_candidate(json.dumps(policy.compact(base['candidate'])), item['case']) == base['candidate']
    assert policy.decide(item, 'A', shared, None)['model'] == item['model']
    assert policy.decide(item, 'B', shared, None)['model'] == item['baseline']
    assert policy.shared(item, lambda *_:'garbled')['action'] == 'candidate_parse_error'
    altered = copy.deepcopy(base['candidate'])
    altered['variables'][0]['upper'] = 2
    assert policy.shared(item, lambda *_:json.dumps(dict(model=policy.compact(altered))))['action'] == 'candidate_domain_change'

def test_numbered_judgments_never_guess_missing_labels():
    item, base = fixture()
    xs = [[0,0,0,0], [1,0,0,0], [1,1,1,1]]
    pairs = [(0,1), (0,2)]
    prompt = policy.judge_prompt(item['case'], xs, pairs)
    obj = json.loads(prompt.split('OUTPUT TEMPLATE (fill values only): ')[1])
    assert len(obj['feasible']) == 3 and len(obj['orders']) == 2
    models = [base['model'], base['candidate']]
    result = policy.assess(json.dumps(obj), item['case'], models, xs, pairs)
    assert result['status'] == 'judged' and result['covered'] == 0
    obj.update(feasible={'x2':False,'x0':True,'x1':True}, orders={'p1':None,'p0':1}, quote='Choose at most two items.')
    result = policy.assess(json.dumps(obj), item['case'], models, xs, pairs)
    assert result['errors'][1] == 0 and result['labels']['feasible'] == [True,True,False]
    obj['feasible'].pop('x1')
    assert policy.assess(json.dumps(obj), item['case'], models, xs, pairs)['status'] == 'judge_parse_error'

def test_partial_report_retains_planned_denominator(tmp_path, monkeypatch, capsys):
    from experiments.boundary_feedback_v1 import data, report
    item, _ = fixture()
    save(tmp_path/'inputs.json', [item, dict(item, case=dict(item['case'], id='missing'))])
    for arm in data.ARMS:
        score = dict(optimal_hit=True, original_feasible=True)
        save(tmp_path/'formal/models/17'/arm/'evaluations/fixture.json', dict(case_id='fixture', seed=17,
             stage='development', arm=arm, action='accepted', semantic='verified_correct', scores={'20':dict(pbit=score, scip=score)}))
    monkeypatch.setattr(data, 'RUN', tmp_path)
    monkeypatch.setattr(report, 'RUN', tmp_path)
    result = report.report('formal')
    assert not result['complete'] and result['expected'] == 12 and result['completed'] == 6
    assert result['stages']['development']['arms']['Q']['missing'] == 1
    assert '1/2 (50.00%)' in capsys.readouterr().out
