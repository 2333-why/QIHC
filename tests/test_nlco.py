import copy
import json
from pathlib import Path
import pytest
from tools.benchmarks.nlco import convert, cases_from_repo
from tools.benchmarks import comparison as c


def row():
    return {'id':'MIS_S_001','task_id':'MIS','difficulty_tier':'S','surface_format':'nl',
            'prompt':'Choose the most nonadjacent vertices A, B, C. Edges A-B, B-C.',
            'instance_canonical_json':{'num_nodes':3,'edges':[{'u':1,'v':2},{'u':2,'v':3}]},
            'instance_surface_json':{'num_nodes':3,'edges':[{'u':'A','v':'B'},{'u':'B','v':'C'}]},
            'reference_solution_canonical_json':[1,3], 'reference_objective_value':2}


def test_mapping_and_gold_separation():
    case = convert(row())
    assert '["A", "B", "C"]' in case['question']
    assert c.evaluate(case, case['gold'], {'x':[1,0,1],'feasible':True})['verified_objective'] == 2
    assert not c.evaluate(case, case['gold'], {'x':[1,1,1],'feasible':True})['verified_feasible']
    case['reference_answer'] = 'DO_NOT_LEAK'
    assert 'DO_NOT_LEAK' not in c.modeling_prompt(case)+c.direct_prompt(case)


def test_reject_wrong_reference_and_bad_mapping():
    r = row(); r['reference_objective_value'] = 3
    with pytest.raises(ValueError, match='Reference'): convert(r)
    r = row(); r['instance_surface_json']['edges'][1]['u'] = 'X'
    with pytest.raises(ValueError, match='Inconsistent'): convert(r)


def test_maxcut_quadratic():
    r = row(); r.update(task_id='MAXCUT',reference_solution_canonical_json=[[1,3],[2]])
    case = convert(r)
    assert c.evaluate(case, case['gold'], {'x':[1,0,1],'feasible':True})['verified_objective'] == 2


def test_real_release_if_available():
    repo = Path('deployment/nlco-inspect')
    if not repo.exists(): pytest.skip('Pinned source integration fixture not downloaded')
    cases, report = cases_from_repo(repo)
    assert len(cases) == 240 and not report['excluded']
    assert len({x['dataset'] for x in cases}) == 12


def test_nlco_cpu_roundtrip_and_summary(tmp_path, monkeypatch):
    def deliver(run, case, phase, response):
        path = run/'requests'/f"{c.key(case['id'], phase)}.json"
        c.save(run/'responses'/path.name, {'response': response, 'seconds': 1,
               'request_sha256': c.hashlib.sha256(path.read_bytes()).hexdigest()})
    case = convert(row())
    def solve(*a, **kw):
        return {'x':[1,0,1], 'feasible':True, 'objective':2, 'seconds':0.01}
    monkeypatch.setattr(c, 'exact', solve); monkeypatch.setattr(c, 'sample', solve)
    gold = case['gold']
    targets = ['objective']+['variable:'+v['name'] for v in gold['variables']]+['constraint:'+v['id'] for v in gold['constraints']]
    raw = json.dumps({'model':gold,'grounding':[{'target':t,'quote':case['question'],'explanation':'test'} for t in targets]})
    for phase in ('model','direct'): c.enqueue(tmp_path,case,phase,'question')
    deliver(tmp_path,case,'model',raw); deliver(tmp_path,case,'direct','{"x":[1,0,1]}')
    cfg = {'solver_seconds':1,'seeds':2,'steps':10,'chains':2}
    assert not c.process_case(tmp_path,case,cfg)
    for phase in ('qihc','solver_feedback'): deliver(tmp_path,case,phase,raw)
    assert c.process_case(tmp_path,case,cfg)
    c.summarize(tmp_path,[case])
    stats = c.load(tmp_path/'summary.json')['datasets'][case['dataset']]['arms']['qihc']
    assert stats['verified_feasible'] == 1
    assert stats['mean_signed_gap_to_published_reference_feasible_instances'] == 0


def test_preparation_downloads_only_active_dataset(tmp_path, monkeypatch):
    from tools.benchmarks import prepare_co as p
    cloned = []
    monkeypatch.setattr(p, 'checkout', lambda item, dest: cloned.append(item['name']) or tmp_path)
    monkeypatch.setattr(p, 'cases_from_repo', lambda *a: ([convert(row())], {'excluded': [], 'requested_per_task_tier':20}))
    monkeypatch.setattr(p.subprocess, 'run', lambda *a, **k: None)
    monkeypatch.setattr(p.sys, 'argv', ['prepare_co', '--base', str(tmp_path), '--run', str(tmp_path/'run')])
    p.main()
    assert cloned == ['NLCO']
    assert (tmp_path/'run/coverage.json').exists()
