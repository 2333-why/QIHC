import itertools
import json

import numpy as np
import pytest

from qihc.generic.bounded import objective, violations
from qihc.generic.solver_baselines.milp import compile_model
from qihc.generic.solver_baselines.registry import BACKENDS, solve
from qihc.generic.solver_baselines.sampling import solve as sample, swap_log_ratio
from qihc.experiment_runtime import storage as s
from experiments.solver_comparison_v1.smoke import case
from experiments.solver_comparison_v1 import run, report


@pytest.mark.parametrize('sense', ['min', 'max'])
@pytest.mark.parametrize('seed', range(5))
def test_exact_backends_match_exhaustive_integer_quadratic_logic(seed, sense):
    spec = case(); rng = np.random.default_rng(seed)
    spec['objective']['sense'] = sense
    spec['objective']['linear'] = rng.integers(-5, 6, 3).tolist()
    spec['objective']['quadratic'] = [[0, 0, int(rng.integers(-3, 4))], [0, 1, int(rng.integers(-3, 4))]]
    if seed % 2:
        spec['logic'][0]['expr']['op'] = 'iff'
    values = [float(objective(spec, x)) for x in itertools.product(range(-1, 3), range(2), range(2)) if violations(spec, x).sum() == 0]
    target = (min if sense == 'min' else max)(values)
    for backend in ('cp_sat', 'scip', 'highs'):
        row = solve(spec, backend, 3., seed)
        assert row['feasible'] and row['optimality_proven'], row
        assert row['objective'] == pytest.approx(target)
        assert row['bound'] == pytest.approx(target)


def test_binary_product_reformulation_extensional():
    spec = case(); spec['logic'] = []
    ir = compile_model(spec)
    original_feasible = {x: objective(spec, x) for x in itertools.product(range(-1, 3), range(2), range(2)) if violations(spec, x).sum() == 0}
    encoded = {}
    for bits in itertools.product(range(2), repeat=len(ir.costs)):
        valid = True
        for a, op, rhs in ir.rows:
            lhs = sum(v*bits[k] for k, v in a.items())
            valid &= lhs <= rhs if op == 'le' else lhs >= rhs if op == 'ge' else lhs == rhs
        if valid:
            x = tuple(ir.decode(bits)); value = ir.sign*(sum(c*b for c, b in zip(ir.costs, bits))+ir.offset)
            assert value == original_feasible[x]
            encoded[x] = value
    assert encoded == original_feasible


@pytest.mark.parametrize('method', ['pbit', 'sa', 'pt', 'penalty_pt'])
def test_sampling_reproducible_scoring_budget_and_validation(method):
    spec = case()
    a = sample(spec, method, 20, 3, 4, max_evaluations=256)
    b = sample(spec, method, 20, 3, 4, max_evaluations=256)
    assert a['x'] == b['x'] and a['candidate_evaluations'] == 256
    assert a['feasible'] == (violations(spec, a['x']).sum() == 0)
    assert a['objective'] == objective(spec, a['x'])
    assert not a['optimality_proven']


def test_swap_detailed_balance_and_noop_temperature():
    ratio = swap_log_ratio(1, 4, 2, 7, .1, .9, 3, 3)
    assert ratio == pytest.approx((1/.1-1/.9)*((1+6)-(4+21)))
    assert swap_log_ratio(1, 4, 2, 7, .1, .9, 3, 3) == pytest.approx(-swap_log_ratio(4, 1, 7, 2, .1, .9, 3, 3))
    assert swap_log_ratio(1, 4, 2, 7, 1, 1, 3, 3) == 0


@pytest.mark.parametrize('backend', ['cp_sat', 'scip', 'highs'])
def test_infeasible_and_fixed_models(backend):
    spec = case(); spec['constraints'].append({'id': 'impossible', 'a': [0, 0, 0], 'op': 'eq', 'rhs': 1})
    row = solve(spec, backend, 2)
    assert not row['feasible'] and not row['optimality_proven']
    spec = {'schema': 'bounded-integer-v2', 'variables': [{'name': 'x', 'lower': -2, 'upper': -2}],
        'objective': {'sense': 'max', 'linear': [3], 'quadratic': [[0, 0, 2]]}, 'constraints': [], 'logic': []}
    row = solve(spec, backend, 2)
    assert row['feasible'] and row['objective'] == 2


def source_fixture(tmp_path):
    source = tmp_path/'source'; root = tmp_path/'comparison'
    c = {'id': 'toy', 'dataset': 'synthetic_toy', 'question': 'A small integer optimization model.', 'gold': case(), 'reference_answer': None}
    s.save(source/'formal/cases.CPU_ONLY.json', [c]); request_hashes = {}
    for arm in ('legacy_flat', 'semantic_staged'):
        identity = s.key(c['id'], arm)
        req = {'contract': 'typed'}; s.save(source/'formal/requests'/(identity+'.json'), req)
        rh = s.digest(source/'formal/requests'/(identity+'.json')); request_hashes[identity] = rh
        s.save(source/'formal/responses'/(identity+'.json'), {'response': json.dumps({'model': c['gold']}),
            'mathematical_validity': 'valid', 'status': 'generated', 'request_sha256': rh,
            'input_tokens': 10, 'output_tokens': 20, 'generation_attempts': 1, 'seconds': 1.})
    s.save(source/'manifest.json', {'config': {'arms': ['legacy_flat', 'semantic_staged']},
        'phase_cases_sha256': {'formal': s.digest(source/'formal/cases.CPU_ONLY.json')},
        'code_commit': 'fixture', 'model': 'fixture-model', 'request_hashes': request_hashes})
    return source, root, c


def test_import_frozen_model_resume_reporting_and_tamper(tmp_path, monkeypatch):
    source, root, c = source_fixture(tmp_path)
    cfg = {'experiment_id': 'E002-1.1', 'source_arms': ['legacy_flat', 'semantic_staged'], 'backends': ['cp_sat', 'pbit'],
        'seconds': [0.1], 'seeds': [0, 1], 'chains': 4, 'cpu_workers': 1, 'reference_seconds': 2, 'report_interval_seconds': 1}
    s.save(tmp_path/'config.json', cfg); monkeypatch.setattr(run, 'HERE', tmp_path)
    manifest = run.prepare(root, source)
    assert run.import_models(root, manifest, [c]) == 2
    run.reference_one(root, c, 2)
    pending = list(run.tasks(root, [c], cfg)); assert len(pending) == 8
    for _, fn, args in pending:
        fn(*args)
    assert not list(run.tasks(root, [c], cfg))
    summary = report.report(root)
    assert summary['status'] == 'complete' and summary['completed_scores'] == 8
    assert len(summary['generation_costs_once_per_model']) == 2
    assert all(p['paired_cases'] == 1 for p in summary['paired'])
    assert all(p['predicted_objective_cases'] == 1 for p in summary['paired'])
    path = next((root/'scores').rglob('*.json')); data = s.load(path)
    data['response_sha256'] = 'tampered'; s.save(path, data)
    with pytest.raises(ValueError, match='identity'):
        report.report(root)
    response = next((source/'formal/responses').glob('*.json')); data = s.load(response)
    data['response'] += ' '; s.save(response, data)
    with pytest.raises(ValueError, match='frozen response'):
        run.import_models(root, manifest, [c])


def test_resume_rechecks_original_request(tmp_path):
    source, root, c = source_fixture(tmp_path)
    manifest = run.prepare(root, source)
    run.import_models(root, manifest, [c])
    req = next((source/'formal/requests').glob('*.json'))
    s.save(req, {'contract': 'typed', 'question': 'mutated'})
    with pytest.raises(ValueError, match='request identity'):
        run.import_models(root, manifest, [c])


def test_real_process_pool_streaming_resume(tmp_path, monkeypatch):
    source, root, _ = source_fixture(tmp_path)
    cfg = {'experiment_id': 'E002-1.1', 'source_arms': ['legacy_flat', 'semantic_staged'],
        'backends': list(BACKENDS), 'seconds': [.1], 'seeds': [0], 'chains': 4,
        'cpu_workers': 2, 'reference_seconds': 2, 'report_interval_seconds': 1}
    s.save(tmp_path/'config.json', cfg); monkeypatch.setattr(run, 'HERE', tmp_path)
    assert run.run(root, source) == 0
    before = {p: p.stat().st_mtime_ns for p in (root/'scores').rglob('*.json')}
    assert len(before) == 14 and s.load(root/'summary.json')['status'] == 'complete'
    assert run.run(root, source) == 0
    assert before == {p: p.stat().st_mtime_ns for p in before}


def test_standalone_requires_all_models_without_modifying_source(tmp_path):
    from experiments.solver_comparison_v1.standalone import require_models
    with pytest.raises(ValueError, match='separately first'):
        require_models(tmp_path/'absent')
    source, root, _ = source_fixture(tmp_path)
    before = {p: s.digest(p) for p in source.rglob('*') if p.is_file()}
    assert require_models(source) == 2
    assert before == {p: s.digest(p) for p in source.rglob('*') if p.is_file()}
    assert not root.exists()
    path = next((source/'formal/responses').glob('*.json'))
    path.unlink()
    with pytest.raises(ValueError, match='1/2 model outputs'):
        require_models(source)
    assert not root.exists()
