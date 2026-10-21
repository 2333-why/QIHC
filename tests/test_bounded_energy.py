import itertools
import json
import numpy as np
import pytest

from qihc.generic.bounded import VERSION, validate, compile_energy, objective, violations, parse, prompt
from qihc.generic.block_search import conditional_qubo, solve


def model(bounds=((0, 1), (0, 1))):
    return {'schema': VERSION, 'variables': [{'name': f'x{i}', 'lower': lo, 'upper': hi} for i, (lo, hi) in enumerate(bounds)],
            'objective': {'sense': 'min', 'linear': [1]*len(bounds), 'quadratic': []},
            'constraints': [], 'logic': []}


def states(n):
    return np.array(list(itertools.product([0, 1], repeat=n)), dtype=float).reshape(-1, n) if n else np.zeros((1, 0))


def min_penalties(c):
    """Exhaust gates; independently minimize every bounded slack analytically."""
    n = c.decision_bits+c.gate_bits
    bits = states(n)
    a = c.energy.residual_a
    r = bits @ a[:, :n].T-c.energy.rhs
    slack_lo = np.minimum(a[:, n:], 0).sum(axis=1)
    slack_hi = np.maximum(a[:, n:], 0).sum(axis=1)
    # Each slack's bounded_weights spans every integer in its interval.
    penalties = np.maximum(r+slack_lo, 0)**2 + np.maximum(-r-slack_hi, 0)**2
    full = np.zeros((len(bits), c.bits))
    full[:, :n] = bits
    return c.decode(full), penalties.sum(axis=1)


@pytest.mark.parametrize('op', ['le', 'ge', 'eq'])
@pytest.mark.parametrize('rhs', [-4, 0, 3])
def test_integer_constraint_exact(op, rhs):
    s = model(((-2, 2), (1, 3)))
    s['constraints'] = [{'id': 'cap', 'a': [2, -1], 'op': op, 'rhs': rhs}]
    c = compile_energy(s)
    x, p = min_penalties(c)
    assert set(map(tuple, x)) == set(itertools.product(range(-2, 3), range(1, 4)))
    assert np.all((p == 0) == (violations(s, x).sum(axis=1) == 0))


@pytest.mark.parametrize('sense', ['min', 'max'])
def test_polynomial_energy_with_offset_and_squares(sense):
    s = model(((-2, 2), (1, 3), (4, 4)))
    s['objective'] = {'sense': sense, 'linear': [3, -2, 1], 'quadratic': [[0, 0, 2], [0, 1, -3], [1, 2, 2]]}
    s['constraints'] = [{'id': 'cap', 'a': [1, 1, -1], 'op': 'le', 'rhs': 0}]
    c = compile_energy(s)
    rng = np.random.default_rng(2)
    z = rng.integers(0, 2, (100, c.bits))
    q, offset = c.qubo([2])
    expected = (1 if sense == 'min' else -1)*objective(s, c.decode(z))/c.energy.bound
    expected += 2*((z @ c.energy.residual_a.T-c.energy.rhs)**2).sum(axis=1)
    np.testing.assert_allclose(np.einsum('bi,ij,bj->b', z, q, z)+offset, expected, atol=1e-8)


@pytest.mark.parametrize('op', ['not', 'and', 'or', 'implies', 'iff'])
def test_logic_truth_table_and_auxiliary_equivalence(op):
    s = model()
    args = [{'var': 0}] if op == 'not' else [{'var': 0}, {'var': 1}]
    s['logic'] = [{'id': 'rule', 'expr': {'op': op, 'args': args}}]
    c = compile_energy(s)
    x, penalty = min_penalties(c)
    for original in states(2):
        best = penalty[(x == original).all(axis=1)].min()
        assert (best == 0) == (violations(s, original).sum() == 0)


def test_nested_repeated_literals():
    s = model()
    s['logic'] = [{'id': 'rule', 'expr': {'op': 'or', 'args': [
        {'op': 'and', 'args': [{'var': 0}, {'var': 0}]}, {'op': 'not', 'args': [{'var': 1}]}]}}]
    c = compile_energy(s)
    x, p = min_penalties(c)
    for original in states(2):
        assert (p[(x == original).all(axis=1)].min() == 0) == (violations(s, original).sum() == 0)


def test_conditioned_qubo_exact_cross_block_energy():
    rng = np.random.default_rng(9)
    q = rng.normal(size=(7, 7)); q = (q+q.T)/2
    z = rng.integers(0, 2, 7)
    block = [0, 3, 6]
    local, offset = conditional_qubo(q, z, block)
    for candidate in states(3):
        full = z.copy(); full[block] = candidate
        assert full @ q @ full == pytest.approx(candidate @ local @ candidate+offset)


def test_fixed_variables_and_infeasibility():
    s = model(((2, 2),))
    s['constraints'] = [{'id': 'impossible', 'a': [1], 'op': 'eq', 'rhs': 3}]
    result = solve(s, steps=16, chains=2)
    assert not result['feasible']
    assert result['x'] == [2]
    assert result['violation'] == 1


@pytest.mark.parametrize('method', ['pbit', 'metropolis', 'uniform'])
def test_search_feedback_budget_and_original_verification(method):
    s = model(((0, 3), (0, 1)))
    s['constraints'] = [{'id': 'limit', 'a': [1, -1], 'op': 'le', 'rhs': 1}]
    a = solve(s, steps=100, chains=4, block_size=2, block_updates=5, method=method)
    b = solve(s, steps=100, chains=4, block_size=2, block_updates=5, method=method)
    assert a['x'] == b['x'] and a['history'] == b['history']
    assert a['bit_updates'] == 400 and a['block_calls'] == 20
    assert a['feasible'] == (violations(s, a['x']).sum() == 0)
    assert a['objective'] == objective(s, a['x'])
    no = solve(s, steps=100, chains=4, block_size=2, feedback=False, method=method)
    assert no['bit_updates'] == a['bit_updates']
    assert no['final_weights'] == ([.1] if method == 'uniform' else [2.])


def test_over_128_bits_runs_in_bounded_blocks():
    s = model(tuple((0, 1) for _ in range(140)))
    s['constraints'] = [{'id': 'cross', 'a': [1]*140, 'op': 'eq', 'rhs': 70}]
    r = solve(s, chains=2, steps=40, block_updates=4, block_size=128)
    assert r['bits'] == 140 and r['block_size'] == 128 and r['block_calls'] == 10
    assert r['violation'] == violations(s, r['x']).sum()


def test_invalid_inputs_rejected():
    s = model(((0, 3),))
    s['logic'] = [{'id': 'bad', 'expr': {'var': 0}}]
    with pytest.raises(ValueError): validate(s)
    s = model(); s['variables'][0]['lower'] = False
    with pytest.raises(ValueError): validate(s)
    with pytest.raises(ValueError): compile_energy(model(((0, 1000),)), max_bits=2)
    with pytest.raises(ValueError): solve(model(), block_size=129)
    with pytest.raises(ValueError): conditional_qubo(np.eye(2), [0, 1], [0, 0])
    with pytest.raises(ValueError): violations(model(), [np.nan, 0])


def test_decode_domains():
    s = model(((-1, 3),))
    c = compile_energy(s)
    assert set(c.decode(states(c.bits)).flatten()) == set(range(-1, 4))
    with pytest.raises(ValueError): c.decode([.5]*c.bits)
    assert violations(s, [1.5]).sum() > 0


def test_llm_wire_format_and_unsupported():
    s = model()
    assert parse(json.dumps(s)) == s
    assert VERSION in prompt('Select two items')
    with pytest.raises(ValueError):
        parse('{"unsupported":"continuous variable"}')


def test_feedback_rows_retain_original_source():
    s = model()
    s['constraints'] = [{'id': 'linear', 'a': [1, 1], 'op': 'le', 'rhs': 1}]
    s['logic'] = [{'id': 'not_first', 'expr': {'op': 'not', 'args': [{'var': 0}]}},
                  {'id': 'second', 'expr': {'var': 1}}]
    c = compile_energy(s)
    assert c.row_sources == [0, 1, 1, 2]
    result = solve(s, steps=128, chains=4, block_size=2)
    assert len(result['history'][-1]['original_violation_rates']) == 3
    assert len(result['final_weights']) == 4
