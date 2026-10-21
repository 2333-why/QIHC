import itertools
import json
from types import SimpleNamespace
import numpy as np
import pytest
from qihc.generic.bounded import VERSION, violations, objective
from qihc.generic.factor_search import solve, energy, objective_scale
from qihc.generic.public_modeling import parse, prompt, search_response, review_prompt, answer_agreement
from tools.benchmarks.public_simulation import score, write, read, seal


QUESTION = 'Choose at most one of two items, with values 2 and 3, to maximize value.'


def model():
    return {'schema': VERSION, 'variables': [{'name': 'a', 'lower': 0, 'upper': 1}, {'name': 'b', 'lower': 0, 'upper': 1}],
            'objective': {'sense': 'max', 'linear': [2, 3], 'quadratic': []},
            'constraints': [{'id': 'cap', 'a': [1, 1], 'op': 'le', 'rhs': 1}], 'logic': []}


def response():
    return json.dumps({'model': model(), 'grounding': [
        {'target': t, 'quote': QUESTION, 'explanation': 'The stated choice is binary with an at-most-one limit.'}
        for t in ['objective', 'variable:a', 'variable:b', 'constraint:cap']]})


def test_original_energy_table():
    s = model()
    x = np.array(list(itertools.product([0, 1], repeat=2)))
    np.testing.assert_allclose(energy(s, x, np.array([2.])), -objective(s, x)/5+2*violations(s, x)[:, 1]**2)
    assert x[np.argmin(energy(s, x, np.array([2.])))].tolist() == [0, 1]


@pytest.mark.parametrize('method', ['pbit', 'metropolis', 'uniform'])
def test_factor_search_reproducible_and_budget(method):
    a = solve(model(), steps=50, chains=4, method=method)
    b = solve(model(), steps=50, chains=4, method=method)
    assert a['history'] == b['history'] and a['x'] == b['x']
    assert a['auxiliary_bits'] == 0 and a['energy_evaluations'] == 400
    assert a['feasible'] == (violations(model(), a['x']).sum() == 0)


def test_integer_logic_and_no_hardware_cap():
    s = model()
    s['variables'][0]['lower'], s['variables'][0]['upper'] = -2, 3
    s['logic'] = [{'id': 'b_true', 'expr': {'var': 1}}]
    r = solve(s, steps=128, chains=4)
    assert r['bits'] == 4 and r['auxiliary_bits'] == 0
    assert r['violation'] == violations(s, r['x']).sum()
    big = {'schema': VERSION, 'variables': [{'name': str(i), 'lower': 0, 'upper': 7} for i in range(200)],
           'objective': {'sense': 'min', 'linear': [1]*200, 'quadratic': []}, 'constraints': [], 'logic': []}
    assert solve(big, steps=2, chains=1)['bits'] == 600


@pytest.mark.parametrize('family', ['knapsack', 'vertex_cover', 'assignment', 'balanced_cut'])
def test_prior_problem_families_use_identical_backend(family):
    from qihc.generic.benchmark import make_case
    old = make_case(family, 3, 7, 'test')['gold']
    spec = {'schema': VERSION, 'variables': [{'name': f'x{i}', 'lower': 0, 'upper': 1} for i in range(old['n'])],
            'objective': old['objective'], 'constraints': old['constraints'], 'logic': []}
    r = solve(spec, steps=50, chains=4)
    assert r['violation'] == violations(spec, r['x']).sum()
    assert r['objective'] == objective(spec, r['x'])


def test_source_grounding_and_explicit_unsupported():
    assert parse(response(), QUESTION)[0] == model()
    assert parse('{"unsupported":"missing data"}', QUESTION) == (None, 'missing data')
    bad = json.loads(response()); bad['grounding'][0]['quote'] = 'invented quote'
    with pytest.raises(ValueError): parse(json.dumps(bad), QUESTION)
    bad = json.loads(response()); bad['grounding'].pop()
    with pytest.raises(ValueError): parse(json.dumps(bad), QUESTION)


def test_feedback_has_no_gold_and_does_not_claim_semantics():
    r = search_response(QUESTION, response(), steps=20, chains=2, seeds=(0,))
    assert len(r['runs']) == 4 and r['semantic_correctness'] == 'unverified'
    assert r['original_question_feasibility'] is None
    p = review_prompt(QUESTION, response(), r, with_search_feedback=True)
    assert 'reference_answer' not in p
    assert 'search failure does not prove infeasibility' in p
    assert answer_agreement({'feasible': False, 'objective': 3}, 3) is False
    assert answer_agreement({'feasible': True, 'objective': 3}, 'infeasible') is None


def test_checkpoint_and_cpu_stage(tmp_path):
    request = {'id': 'test', 'dataset': 'test', 'question': QUESTION, 'prompt': prompt(QUESTION), 'round': 0, 'branch': 'base'}
    write(tmp_path/'requests.jsonl', [request])
    write(tmp_path/'predictions.jsonl', [{'id': 'test', 'status': 'generated', 'response': response()}])
    write(tmp_path/'answers.jsonl', [{'id': 'test', 'reference_answer': 'SECRET_REFERENCE'}])
    args = SimpleNamespace(requests=tmp_path/'requests.jsonl', predictions=tmp_path/'predictions.jsonl',
                           answers=tmp_path/'answers.jsonl', output=tmp_path/'scores.jsonl', steps=20, chains=2, seeds=1)
    score(args)
    before = args.output.read_bytes()
    score(args)
    assert args.output.read_bytes() == before
    assert 'SECRET_REFERENCE' not in (tmp_path/'scores.feedback.requests.jsonl').read_text()
    assert 'SECRET_REFERENCE' not in (tmp_path/'scores.self_review.requests.jsonl').read_text()
    args.steps = 21
    with pytest.raises(ValueError, match='changed'): score(args)


def test_invalid_model_kept_in_denominator(tmp_path):
    r = search_response(QUESTION, 'not JSON', steps=1, chains=1)
    assert r['status'] == 'invalid_model' and r['runs'] == []
