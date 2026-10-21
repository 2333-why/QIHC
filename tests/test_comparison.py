import json
from types import SimpleNamespace
import pytest
from tools.benchmarks import comparison as m
from qihc.generic.bounded import VERSION


def case():
    gold = {'schema': VERSION, 'variables': [{'name': 'x0', 'lower': 0, 'upper': 1}],
            'objective': {'sense': 'max', 'linear': [1], 'quadratic': []}, 'constraints': [], 'logic': []}
    return {'id': 'example', 'dataset': 'synthetic_knapsack', 'question': 'Select x0 to maximize its value 1.',
            'gold': gold, 'reference_answer': 'HIDDEN_SECRET'}


def raw(c):
    return json.dumps({'model': c['gold'], 'grounding': [
        {'target': t, 'quote': c['question'], 'explanation': 'Binary choice stated in text'}
        for t in ('objective', 'variable:x0')]})


def deliver(run, c, phase, response):
    path = run/'requests'/f"{m.key(c['id'], phase)}.json"
    m.save(run/'responses'/path.name, {'response': response, 'status': 'generated', 'seconds': 1,
                                    'output_tokens': 10, 'request_sha256': m.hashlib.sha256(path.read_bytes()).hexdigest()})


def fake_solve(spec, *a, **kw):
    return {'x': [1], 'objective': 1, 'feasible': True, 'seconds': .1, 'optimality_proven': True}


def test_complete_cpu_queue_no_ablation_or_training(tmp_path, monkeypatch):
    c = case(); cfg = {'solver_seconds': 1, 'seeds': 2, 'steps': 10, 'chains': 2}
    monkeypatch.setattr(m, 'exact', fake_solve); monkeypatch.setattr(m, 'sample', fake_solve)
    m.enqueue(tmp_path, c, 'model', m.modeling_prompt(c))
    m.enqueue(tmp_path, c, 'direct', m.direct_prompt(c))
    assert not m.process_case(tmp_path, c, cfg)
    deliver(tmp_path, c, 'model', raw(c)); deliver(tmp_path, c, 'direct', '{"x":[1]}')
    assert not m.process_case(tmp_path, c, cfg)
    for phase in ('qihc', 'solver_feedback'):
        p = tmp_path/'requests'/f"{m.key(c['id'], phase)}.json"
        assert 'HIDDEN_SECRET' not in p.read_text()
        deliver(tmp_path, c, phase, raw(c))
    assert m.process_case(tmp_path, c, cfg)
    final = next((tmp_path/'final').glob('*.json'))
    first = final.read_bytes()
    assert m.process_case(tmp_path, c, cfg) and final.read_bytes() == first
    r = m.load(final)
    assert set(r['arms']) == {'direct_llm','llm_cp_sat','solver_feedback_cp_sat','qihc','gold_cp_sat','gold_simulated_annealing','gold_random'}
    assert len(list((tmp_path/'requests').glob('*.json'))) == 4
    m.summarize(tmp_path, [c])
    stats = m.load(tmp_path/'summary.json')['datasets'][c['dataset']]['arms']
    assert stats['qihc']['llm_calls'] == 2
    assert stats['qihc']['search_seconds_all_seeds'] == pytest.approx(.4)
    assert stats['llm_cp_sat']['llm_calls'] == 1


def test_independent_gold_checks_wrong_model():
    c = case()
    bad = json.loads(json.dumps(c['gold'])); bad['variables'][0]['name'] = 'invented'
    assert m.evaluate(c, bad, fake_solve(bad))['verified_feasible'] is False
    assert m.evaluate(c, c['gold'], {'x': [1, 0], 'feasible': True})['verified_feasible'] is False


def test_corrupt_request_or_response_fails(tmp_path):
    c = case(); m.enqueue(tmp_path, c, 'model', 'original')
    with pytest.raises(ValueError): m.enqueue(tmp_path, c, 'model', 'changed')
    deliver(tmp_path, c, 'model', raw(c))
    p = tmp_path/'requests'/f"{m.key(c['id'], 'model')}.json"
    m.save(p, {'changed': True})
    with pytest.raises(ValueError, match='mismatch'): m.process_case(tmp_path, c, {})


def test_invalid_models_remain_explicit():
    r = m.solve_model(case(), 'bad JSON', 'pbit', {})
    assert r['status'] == 'invalid_model' and r['runs'] == []
    r = m.solve_model(case(), '{"unsupported":"no data"}', 'pbit', {})
    assert r['status'] == 'unsupported'


def test_os_lock_releases(tmp_path):
    with m.worker_lock(tmp_path/'worker.lock'): pass
    with m.worker_lock(tmp_path/'worker.lock'): pass


def test_public_answer_is_not_verified_feasibility():
    c = case(); spec = c.pop('gold'); c['gold'] = None; c['reference_answer'] = 1
    out = m.evaluate(c, spec, fake_solve(spec))
    assert out['verified_feasible'] is None and out['reference_answer_agreement'] is True


def test_launcher_separation():
    root = m.ROOT/'tools/benchmarks'
    gpu = (root/'02_comparison_gpu.sh').read_text()
    cpu = (root/'01_comparison_cpu.sh').read_text()
    assert 'comparison.py gpu' in gpu and 'comparison.py cpu' not in gpu
    assert 'comparison.py cpu' in cpu and 'prepare_comparison.py' in cpu
    assert 'nvidia-smi' not in gpu


def test_default_synthetic_grid_prepares_and_resumes(tmp_path, monkeypatch):
    monkeypatch.setattr(m, 'model_identity', lambda p: [{'test': True}])
    monkeypatch.setattr(m, 'audit', lambda *a: ({}, [{'id': 'pub', 'dataset': 'public', 'question': 'A binary choice'}],
                                             [{'id': 'pub', 'reference_answer': 1}]))
    args = SimpleNamespace(run=tmp_path/'run', data=tmp_path/'data', model=tmp_path/'model',
                           per_dataset=40, synthetic_seeds=5, steps=10, chains=2, seeds=1, solver_seconds=1)
    m.prepare(args)
    assert m.load(args.run/'manifest.json')['cases'] == 61
    assert len(list((args.run/'requests').glob('*.json'))) == 122
    m.prepare(args)
    assert m.validate_run(args.run)['cases'] == 61
