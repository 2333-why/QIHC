"""Numerical invariants, fairness, isolation and complete mocked-inference plumbing."""
import copy
import hashlib
import itertools
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from qihc.generic.bounded import VERSION, objective, violations
from qihc.generic.distribution import search as s
from tools.benchmarks import comparison as old
from tools.benchmarks.distribution import experiment as e
from tools.benchmarks.distribution import inference as g


def spec():
    return {'schema': VERSION,
            'variables': [{'name': 'x0', 'lower': 0, 'upper': 1}, {'name': 'x1', 'lower': 0, 'upper': 1}],
            'objective': {'sense': 'max', 'linear': [3, 1], 'quadratic': []},
            'constraints': [{'id': 'one', 'a': [2, 2], 'op': 'eq', 'rhs': 2}], 'logic': []}


def case():
    return {'id': 'one', 'dataset': 'fixture', 'question': 'Choose exactly one of x0 and x1. Binary choices have values 3 and 1; maximize total value.',
            'gold': spec(), 'reference_answer': 3, 'reference_status': 'test_exhaustive_optimal'}


def raw(case):
    model = case['gold']
    targets = ['objective']+['variable:'+v['name'] for v in model['variables']]+[
        'constraint:'+c['id'] for c in model['constraints']+model['logic']]
    return json.dumps({'model': model, 'grounding': [{'target': t, 'quote': case['question'],
                        'explanation': 'Fixture only, not a model-generated semantic proof.'} for t in targets]})


def deliver(run, case, phase, text):
    p = run/'requests'/(old.key(case['id'], phase)+'.json')
    old.save(run/'responses'/p.name, {'response': text, 'status': 'generated', 'seconds': .1,
             'output_tokens': 10, 'request_sha256': hashlib.sha256(p.read_bytes()).hexdigest()})


@pytest.mark.parametrize('representation', s.REPRESENTATIONS)
@pytest.mark.parametrize('sense', ['min', 'max'])
def test_exact_minimum_preserves_original_feasible_optimum(representation, sense):
    model = spec(); model['objective']['sense'] = sense
    x = np.array(list(itertools.product([0, 1], repeat=2)))
    energy = s.energy_values(model, x, np.array([3.]), representation)
    winner = x[np.argmin(energy)]
    assert violations(model, winner).sum() == 0
    feasible = x[violations(model, x).sum(axis=1) == 0]
    f = objective(model, feasible)
    assert objective(model, winner) == (min(f) if sense == 'min' else max(f))
    assert s.primitive_divisors(model).tolist() == [2.]


def test_integer_logic_domains_and_policy_supports():
    model = spec()
    model['variables'].append({'name': 'z', 'lower': -2, 'upper': 3})
    model['objective']['linear'].append(2)
    model['objective']['quadratic'] = [[0, 2, 1]]
    model['constraints'][0]['a'].append(0)
    model['logic'] = [{'id': 'logic', 'expr': {'op': 'implies', 'args': [{'var': 0}, {'op': 'not', 'args': [{'var': 1}]}]}}]
    for policy in (s.Policy(r, u) for r in s.REPRESENTATIONS for u in s.UPDATES):
        r = s.window(model, policy=policy, method='pbit', seed=3, chains=3, budget=80)
        assert violations(model, r['x'])[0] == 0
        assert r['candidate_evaluations'] == 240
    assert any(len(support) > 1 for support in s.factor_sites(model, s.encoding(model)[0]))


@pytest.mark.parametrize('selection', ['fixed', 'energy', 'update', 'joint', 'random'])
@pytest.mark.parametrize('method', ['pbit', 'metropolis'])
def test_equal_budget_includes_every_probe(selection, method):
    r = s.solve(spec(), seed=3, chains=3, budget=257, method=method, selection=selection, interventions=True)
    assert r['candidate_evaluations'] == r['candidate_budget'] == 771
    assert sum(w['candidate_evaluations'] for w in r['probe_ledger']) == 771
    assert len(r['probe_ledger']) == 12
    assert r['unused_budget'] == 0
    assert r['feasible'] == (violations(spec(), r['x']).sum() == 0)
    assert r['optimality_proven'] is False


def test_sham_pair_has_zero_difference_and_gibbs_alias_is_exact():
    r = s.solve(spec(), seed=7, budget=256, chains=2)
    assert all(v['delta'] == 0 for pair in r['diagnostic']['interventions']
               for v in pair['violation_rate_delta_high_minus_low'])
    kwargs = dict(seed=7, budget=256, chains=2, selection='joint', interventions=True)
    a, b = s.solve(spec(), method='pbit', **kwargs), s.solve(spec(), method='gibbs', **kwargs)
    for key in ('x', 'diagnostic', 'candidate_evaluations', 'selected_policy', 'probe_ledger'):
        assert a[key] == b[key]


def test_pair_heat_bath_can_move_between_one_hot_states_at_low_temperature():
    model = spec(); model['objective']['linear'] = [0, 0]
    initial = np.tile([1, 0], (32, 1))
    kwargs = dict(method='pbit', seed=4, chains=32, budget=64, initial_bits=initial, temperature=(.001, .001))
    single = s.window(model, policy=s.Policy(update='single'), **kwargs)
    block = s.window(model, policy=s.Policy(update='factor_pair'), **kwargs)
    assert np.array_equal(single['bits'], initial)
    assert not np.array_equal(block['bits'], initial)
    assert block['diagnostic']['observed_feasible_rate'] == 1


def test_constant_model_and_no_constraints():
    model = spec()
    model['variables'] = [{'name': 'x0', 'lower': 0, 'upper': 0}, {'name': 'x1', 'lower': 1, 'upper': 1}]
    r = s.solve(model, budget=256, chains=2, interventions=True, selection='joint')
    assert r['x'] == [0, 1] and r['bits'] == 0 and r['feasible']
    assert r['candidate_evaluations'] == 24
    model = spec(); model['constraints'] = []
    r = s.solve(model, budget=256, chains=2, interventions=True)
    assert r['diagnostic']['distribution']['constraints'] == []
    assert r['diagnostic']['interventions'] == []


def test_metropolis_does_not_pay_for_unused_alternatives():
    kwargs = dict(policy=s.Policy(update='factor_pair'), seed=1, chains=3, budget=256)
    a = s.window(spec(), method='pbit', **kwargs)
    b = s.window(spec(), method='metropolis', **kwargs)
    assert a['candidate_evaluations'] == b['candidate_evaluations'] == 768
    assert b['transition_attempts'] > a['transition_attempts']


@pytest.mark.parametrize('kwargs', [{'budget': 0}, {'budget': True}, {'chains': 0}, {'selection': 'magic'}, {'method': 'magic'}])
def test_validation(kwargs):
    with pytest.raises(ValueError):
        s.solve(spec(), **kwargs)


def test_feedback_visibility_and_no_reference_leak():
    cse = case(); cse['reference_answer'] = 'HIDDEN_ANSWER_912233'
    r = s.solve(spec(), budget=256, chains=2, interventions=True)
    for arm in e.ARMS:
        if arm == 'cp_sat_feedback':
            continue
        text = e.review_prompt(cse, raw(cse), r['diagnostic'], arm)
        assert 'HIDDEN_ANSWER_912233' not in text
        payload = e.feedback_payload(r['diagnostic'], arm)
        assert ('distribution' in payload) == (e.ARMS[arm]['feedback'] in ('distribution', 'intervention'))
        assert ('interventions' in payload) == (e.ARMS[arm]['feedback'] == 'intervention')


def test_prepare_is_read_only_resume_guard_and_source_isolation(tmp_path, monkeypatch):
    legacy, run, model = tmp_path/'old', tmp_path/'new', tmp_path/'model'
    model.mkdir(); (model/'config.json').write_text('{}')
    cse = case()
    old.save(legacy/'cases.CPU_ONLY.json', [cse])
    old.save(legacy/'manifest.json', {'cases_sha256': e.digest(legacy/'cases.CPU_ONLY.json'),
             'settings': {'source': old.code_hash(), 'model': str(model), 'model_files': e.model_identity(model)}})
    old.enqueue(legacy, cse, 'model', old.modeling_prompt(cse))
    deliver(legacy, cse, 'model', raw(cse))
    before = {p.relative_to(legacy).as_posix(): p.read_bytes() for p in legacy.rglob('*') if p.is_file()}
    args = SimpleNamespace(from_run=legacy, run=run, budget=256, chains=2, seeds=1, solver_seconds=.1,
                           limit=0, arms=['pbit_best', 'pbit_selected'])
    e.prepare(args); e.prepare(args)
    after = {p.relative_to(legacy).as_posix(): p.read_bytes() for p in legacy.rglob('*') if p.is_file()}
    assert before == after
    assert e.validate(run)['reused_initial_responses'] == ['one']
    args.chains = 3
    with pytest.raises(ValueError, match='settings changed'):
        e.prepare(args)
    monkeypatch.setattr(e, 'source_hash', lambda: 'changed')
    with pytest.raises(ValueError, match='source changed'):
        e.validate(run)


def test_complete_all_arms_real_cpu_search_mocked_model_and_resume(tmp_path):
    pytest.importorskip('ortools')
    cse = case()
    cfg = {'budget': 256, 'chains': 2, 'seeds': 1, 'solver_seconds': .2, 'arms': list(e.ARMS)}
    old.enqueue(tmp_path, cse, 'model', old.modeling_prompt(cse))
    deliver(tmp_path, cse, 'model', raw(cse))
    for _ in range(60):
        state = e.process_case(tmp_path, cse, cfg)
        for p in (tmp_path/'requests').glob('*.json'):
            request = old.load(p)
            if not (tmp_path/'responses'/p.name).exists():
                deliver(tmp_path, cse, request['phase'], raw(cse))
        if state == 'complete':
            break
    assert state == 'complete'
    final = next((tmp_path/'final').glob('*.json'))
    previous = final.read_bytes()
    assert e.process_case(tmp_path, cse, cfg) == 'complete' and final.read_bytes() == previous
    assert len(list((tmp_path/'requests').glob('*.json'))) == len(e.ARMS)+1
    summary = e.summarize(tmp_path, [cse], cfg)
    assert summary['status'] == 'complete'
    stats = summary['datasets']['fixture']['arms']
    for arm in e.ARMS:
        assert stats[arm]['attributed_llm_calls'] == 2
        assert stats[arm]['attributed_output_tokens'] == 20
        if arm != 'cp_sat_feedback':
            assert stats[arm]['candidate_evaluations_both_stages'] == 1024
    assert (tmp_path/'summary.md').exists()


def test_invalid_models_and_partial_denominators(tmp_path):
    cse = case()
    cfg = {'budget': 256, 'chains': 2, 'seeds': 1, 'solver_seconds': .1, 'arms': ['pbit_best']}
    bad = e.solve_one(cse, 'not json', 'pbit_best', cfg, 0)
    assert bad['status'] == 'invalid_model' and not bad['runs']
    old.enqueue(tmp_path, cse, 'model', old.modeling_prompt(cse))
    deliver(tmp_path, cse, 'model', 'not json')
    for _ in range(5):
        e.process_case(tmp_path, cse, cfg)
        if (tmp_path/'requests'/(old.key(cse['id'], 'review_pbit_best')+'.json')).exists():
            deliver(tmp_path, cse, 'review_pbit_best', 'not json')
    other = dict(cse, id='not_finished')
    report = e.summarize(tmp_path, [cse, other], cfg)
    assert report['status'] == 'partial' and report['completed'] == 1 and report['expected'] == 2
    assert report['datasets']['fixture']['arms']['pbit_best']['verified_feasible'] == 0


def test_gpu_queue_shards_feedback_priority_and_no_gold_read(tmp_path):
    for i in range(20):
        for phase in ('model', 'review_pbit_selected'):
            old.enqueue(tmp_path, {'id': str(i)}, phase, 'question only')
    shards = [g.pending(tmp_path, rank, 2, 100) for rank in range(2)]
    names = [{p.name for p, _ in shard} for shard in shards]
    assert not names[0] & names[1] and len(names[0] | names[1]) == 40
    for shard in shards:
        flags = [r['phase'].startswith('review_') for _, r in shard]
        assert flags == sorted(flags, reverse=True)
    source = Path(g.__file__).read_text()
    assert 'cases.CPU_ONLY.json' not in source
    assert 'cpu=False' in source


def test_legacy_hash_is_unchanged_by_nested_modules():
    # Regression guard for the user's live experiment, pinned to 7cbc780.
    assert old.code_hash() == '2559c9609a22b94258d18e296c37cc6873fb0a385b52b8a75db453f507bdf787'


def test_wrong_semantics_cannot_become_verified_success():
    cse = case(); wrong = copy.deepcopy(cse)
    wrong['gold']['constraints'] = []  # Model omits exactly-one constraint.
    cfg = {'budget': 256, 'chains': 4, 'seeds': 1}
    r = e.solve_one(cse, raw(wrong), 'pbit_selected', cfg, 0)['runs'][0]
    assert r['predicted_model_feasible'] is True
    assert r['x'] == [1, 1]
    assert r['verified_feasible'] is False


def test_active_intervention_blind_control_has_identical_search():
    cfg = {'budget': 256, 'chains': 2}
    a = e.solve_one(case(), raw(case()), 'pbit_probe_blind', cfg, 0)
    b = e.solve_one(case(), raw(case()), 'pbit_intervention', cfg, 0)
    assert a['diagnostic'] == b['diagnostic']
    assert 'interventions' not in e.feedback_payload(a['diagnostic'], 'pbit_probe_blind')
    assert 'interventions' in e.feedback_payload(b['diagnostic'], 'pbit_intervention')


def test_random_policy_has_same_pilot_costs_as_adaptive():
    kwargs = dict(seed=1, chains=2, budget=256, interventions=True)
    a, b = s.solve(spec(), selection='random', **kwargs), s.solve(spec(), selection='joint', **kwargs)
    assert a['probe_ledger'][:11] == b['probe_ledger'][:11]
    assert a['candidate_evaluations'] == b['candidate_evaluations']


def test_primitive_energy_is_invariant_to_integer_constraint_rescaling():
    original, scaled = spec(), spec()
    scaled['constraints'][0]['a'] = [14, 14]
    scaled['constraints'][0]['rhs'] = 14
    x = np.array(list(itertools.product([0, 1], repeat=2)))
    for representation in ('primitive_square', 'primitive_linear'):
        assert np.array_equal(s.energy_values(original, x, np.array([3.]), representation),
                              s.energy_values(scaled, x, np.array([3.]), representation))


def test_exhaustive_random_small_integer_models():
    rng = np.random.default_rng(41)
    x = np.array(list(itertools.product(range(-1, 2), repeat=3)))
    checked = 0
    for _ in range(50):
        model = {'schema': VERSION, 'variables': [{'name': f'x{i}', 'lower': -1, 'upper': 1} for i in range(3)],
            'objective': {'sense': 'min' if rng.random() < .5 else 'max',
                          'linear': rng.integers(-5, 6, 3).tolist(), 'quadratic': [[0, 1, int(rng.integers(-5, 6))]]},
            'constraints': [{'id': str(i), 'a': (rng.integers(-2, 3, 3)*2).tolist(),
                             'op': ('le', 'ge', 'eq')[int(rng.integers(3))], 'rhs': int(rng.integers(-2, 3))*2} for i in range(2)],
            'logic': []}
        valid = violations(model, x).sum(axis=1) == 0
        if not valid.any():
            continue
        checked += 1
        sign = 1 if model['objective']['sense'] == 'min' else -1
        optimum = min(sign*objective(model, x[valid]))
        for representation in s.REPRESENTATIONS:
            scores = s.energy_values(model, x, np.full(2, 3.), representation)
            winners = x[np.isclose(scores, scores.min(), atol=1e-12, rtol=0)]
            assert np.all(violations(model, winners).sum(axis=1) == 0)
            assert np.all(sign*objective(model, winners) == optimum)
    assert checked >= 10


def test_counterfactual_diagnostics_stay_bounded_for_many_constraints():
    model = spec()
    model['constraints'] = [dict(model['constraints'][0], id=f'constraint{i}') for i in range(30)]
    r = s.solve(model, budget=256, chains=2, interventions=True)
    d = r['diagnostic']['distribution']
    assert d['constraints_total'] == 30 and len(d['constraints']) == 8
    assert len(d['co_violations']) <= 8
    assert len(r['diagnostic']['interventions']) == 2


def test_gpu_validation_never_opens_cpu_case_file(tmp_path, monkeypatch):
    old.save(tmp_path/'manifest.json', {'experiment': 'distribution_feedback_v1',
             'settings': {'source': e.source_hash(), 'model': 'mock', 'model_files': []}, 'cases_sha256': 'NOT_A_REAL_FILE'})
    monkeypatch.setattr(e, 'model_identity', lambda _: [])
    e.validate(tmp_path, cpu=False, model=True)
    with pytest.raises(FileNotFoundError):
        e.validate(tmp_path, cpu=True)


def test_tampered_response_stops_instead_of_mixing_checkpoints(tmp_path):
    cse = case()
    old.enqueue(tmp_path, cse, 'model', old.modeling_prompt(cse))
    deliver(tmp_path, cse, 'model', raw(cse))
    path = tmp_path/'requests'/(old.key(cse['id'], 'model')+'.json')
    old.save(path, {'changed': True})
    with pytest.raises(ValueError, match='mismatch'):
        e.response(tmp_path, cse, 'model')


def test_changed_model_response_invalidates_existing_search_checkpoint(tmp_path):
    cse = case()
    cfg = {'budget': 256, 'chains': 2, 'seeds': 1, 'arms': ['pbit_best']}
    old.enqueue(tmp_path, cse, 'model', old.modeling_prompt(cse))
    deliver(tmp_path, cse, 'model', raw(cse))
    assert e.process_case(tmp_path, cse, cfg) == 'searched_initial'
    # Same request, different answer: cannot reuse earlier search even though the
    # queue request checksum alone still passes.
    deliver(tmp_path, cse, 'model', '{"unsupported":"changed answer"}')
    with pytest.raises(ValueError, match='checkpoint input changed'):
        e.process_case(tmp_path, cse, cfg)
