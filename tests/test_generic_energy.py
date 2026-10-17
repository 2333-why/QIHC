import itertools
import json
import numpy as np
import pytest

from qihc.generic.energy import compile_energy, objective, violations, solve, bounded_weights
from qihc.generic.benchmark import make_case, make_dataset, infer, semantic_equal


def tiny(op="le", rhs=2):
    return {"n": 2, "objective": {"sense": "max", "linear": [3, -1], "quadratic": [[0, 1, 2]]},
            "constraints": [{"id": "c0", "a": [2, -1], "op": op, "rhs": rhs}]}


@pytest.mark.parametrize("upper", range(20))
def test_bounded_slack_covers_exact_range(upper):
    ws = bounded_weights(upper)
    assert {sum(a * b for a, b in zip(bits, ws)) for bits in itertools.product([0, 1], repeat=len(ws))} == set(range(upper + 1))


@pytest.mark.parametrize("op", ["le", "ge", "eq"])
@pytest.mark.parametrize("rhs", [-3, -1, 0, 1, 3])
def test_qubo_ising_and_penalty_gap_exhaustively(op, rhs):
    spec = tiny(op, rhs)
    e = compile_energy(spec)
    q, c = e.qubo([2])
    w, h, offset = e.ising([2])
    all_bits = np.asarray(list(itertools.product([0, 1], repeat=e.bits)))
    values = np.einsum("bi,ij,bj->b", all_bits, q, all_bits) + c
    explicit = -objective(spec, all_bits[:, :2]) / e.bound + 2 * ((all_bits @ e.residual_a.T - e.rhs) ** 2).sum(axis=1)
    spins = 2 * all_bits - 1
    ising = -.5 * np.einsum("bi,ij,bj->b", spins, w, spins) - spins @ h + offset
    np.testing.assert_allclose(values, explicit, atol=1e-10)
    np.testing.assert_allclose(values, ising, atol=1e-10)
    original = np.asarray(list(itertools.product([0, 1], repeat=2)))
    feasible = violations(spec, original).sum(axis=1) == 0
    for x, ok in zip(original, feasible):
        subset = all_bits[(all_bits[:, :2] == x).all(axis=1)]
        minimum = ((subset @ e.residual_a.T - e.rhs) ** 2).sum(axis=1).min()
        assert (minimum == 0) == ok
        assert ok or minimum >= 1
    if feasible.any():
        minimizers = all_bits[np.isclose(values, values.min())]
        assert np.all(violations(spec, minimizers[:, :2]) == 0)


def test_conditional_energy_difference():
    e = compile_energy(tiny())
    q, _ = e.qubo([2])
    rng = np.random.default_rng(7)
    x = rng.integers(0, 2, (12, e.bits)).astype(float)
    fields = 2 * x @ q - np.diag(q)
    for i in range(e.bits):
        delta = fields[:, i] + 2 * q[i, i] * (1 - x[:, i])
        zero, one = x.copy(), x.copy()
        zero[:, i], one[:, i] = 0, 1
        direct = np.einsum("bi,ij,bj->b", one, q, one) - np.einsum("bi,ij,bj->b", zero, q, zero)
        np.testing.assert_allclose(delta, direct)


@pytest.mark.parametrize("family,size", [("knapsack", 6), ("vertex_cover", 6), ("balanced_cut", 6), ("assignment", 2)])
def test_multi_problem_search_and_independent_verification(family, size):
    spec = make_case(family, size, 17)["gold"]
    result = solve(spec, seed=0, steps=256, chains=32)
    x = np.array(result["x"])
    assert result["feasible"] == (violations(spec, x).sum() == 0)
    assert result["objective"] == objective(spec, x)
    assert result["feasible"]
    again = solve(spec, seed=0, steps=256, chains=32)
    assert result["x"] == again["x"]


def test_bad_model_not_silently_replaced():
    class Bad:
        def generate(self, prompt):
            return '{"n":2,"objective":{"sense":"min","linear":[0,0],"quadratic":[]},"extra":1}'
    with pytest.raises(ValueError):
        infer(Bad(), "test")
    spec = tiny()
    modified = json.loads(json.dumps(spec))
    modified["constraints"] = []
    assert not semantic_equal(modified, spec)


def test_two_stage_inference_has_no_gold_input():
    case = make_case("knapsack", 4, 1)
    gold = case["gold"]
    class Good:
        calls = 0
        def generate(self, prompt):
            self.calls += 1
            return json.dumps({"n": gold["n"], "objective": gold["objective"]} if self.calls == 1 else {"constraints": gold["constraints"], "policy": "fixed"})
    spec, policy, traces = infer(Good(), case["description"])
    assert spec == gold and policy == "fixed" and len(traces) == 2


def test_dataset_no_id_leakage():
    rows = make_dataset(12)
    assert len({r["id"] for r in rows}) == 48
    assert {r["split"] for r in rows} == {"train", "validation", "test"}
    for row in rows:
        compile_energy(row["gold"])


def test_no_constraints_and_impossible_constraints():
    spec = tiny()
    spec["constraints"] = []
    assert solve(spec, steps=10, chains=4)["feasible"]
    impossible = tiny("ge", 3)
    result = solve(impossible, steps=10, chains=4)
    assert not result["feasible"]  # NOT a proof of infeasibility in general


@pytest.mark.parametrize("mutation", [lambda s: s.update(n=True), lambda s: s["objective"]["linear"].__setitem__(0, float("nan")),
                                     lambda s: s["constraints"][0].update(op="eval"), lambda s: s["objective"].update(quadratic=[[1, 0, 2]])])
def test_reject_unsafe_ir(mutation):
    spec = tiny()
    mutation(spec)
    with pytest.raises(ValueError):
        compile_energy(spec)


def test_fixed_kernel_preserves_boltzmann_distribution():
    spec = tiny("eq", 1)
    e = compile_energy(spec)
    q, c = e.qubo([2])
    states = np.asarray(list(itertools.product([0, 1], repeat=e.bits)))
    energies = np.einsum("bi,ij,bj->b", states, q, states) + c
    pi = np.exp(-(energies - energies.min()))
    pi /= pi.sum()
    transition = np.zeros((len(states), len(states)))
    for row, x in enumerate(states):
        for i in range(e.bits):
            delta = q[i, i] + 2 * (x @ q[:, i] - x[i] * q[i, i])
            prob = 1 / (1 + np.exp(delta))
            for bit, mass in [(0, 1 - prob), (1, prob)]:
                y = x.copy()
                y[i] = bit
                col = np.flatnonzero((states == y).all(axis=1))[0]
                transition[row, col] += mass / e.bits
    np.testing.assert_allclose(pi @ transition, pi, atol=1e-12)


def test_failed_translation_keeps_all_evaluation_denominators(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from experiments import generic_energy as runner
    class Bad:
        calls = input_tokens = output_tokens = 0
        def __init__(self, args):
            pass
        def generate(self, prompt):
            return "invalid json"
    monkeypatch.setattr(runner, "Generator", Bad)
    args = SimpleNamespace(output=tmp_path, tag="base", mode="llm", holdout_family="balanced_cut", seeds=3)
    runner.evaluate(args, [make_case("knapsack", 4, 1, "test")])
    rows = runner.read_rows(tmp_path / "base.jsonl")
    assert len(rows) == 10
    assert all(not r["feasible"] and "error" in r for r in rows)
    summary = runner.summarize(tmp_path)
    assert all(r["feasible_rate"] == 0 for r in summary["rows"])
    assert sorted(r["jobs"] for r in summary["rows"]) == [1, 3, 3, 3]


def test_teacher_keeps_constraints_identical_and_excludes_heldout(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from experiments import generic_energy as runner
    def fake(spec, policy, **kwargs):
        return {"feasible": True, "objective": {"fixed": 10, "adaptive": 8, "warm": 6, "cold": 4}[policy], "violation": 0}
    monkeypatch.setattr(runner, "solve", fake)
    args = SimpleNamespace(output=tmp_path, holdout_family="balanced_cut", chains=2, steps=2)
    cases = [make_case("knapsack", 4, 1), make_case("balanced_cut", 4, 2), make_case("knapsack", 4, 3, "test")]
    runner.teacher(args, cases)
    sft = runner.read_rows(tmp_path / "sft.jsonl")
    dpo = runner.read_rows(tmp_path / "dpo.jsonl")
    assert len(sft) == 2 and len(dpo) == 1
    assert {r["case_id"] for r in sft + dpo} == {cases[0]["id"]}
    assert json.loads(dpo[0]["chosen"])["constraints"] == json.loads(dpo[0]["rejected"])["constraints"]


def test_predicted_model_cannot_self_certify_feasibility(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from experiments import generic_energy as runner
    case = make_case("knapsack", 4, 7, "test")
    gold = case["gold"]
    class Wrong:
        calls = input_tokens = output_tokens = 0
        def __init__(self, args):
            pass
        def generate(self, prompt):
            self.calls += 1
            return json.dumps({"n": 4, "objective": gold["objective"]} if self.calls == 1 else {"constraints": [], "policy": "fixed"})
    def reference(spec, seconds):
        assert spec["constraints"] == []  # same predicted model, not hidden gold
        return {"x": [1] * 4, "objective": sum(gold["objective"]["linear"]), "status": "OPTIMAL", "seconds": 0, "bound": 0}
    monkeypatch.setattr(runner, "Generator", Wrong)
    monkeypatch.setattr(runner, "reference_solve", reference)
    args = SimpleNamespace(output=tmp_path, tag="base", mode="llm", holdout_family="balanced_cut", seeds=1,
                           steps=16, chains=8, reference_seconds=1)
    runner.evaluate(args, [case])
    rows = runner.read_rows(tmp_path / "base.jsonl")
    assert len(rows) == 4
    for row in rows:
        assert row["feasible"] == (violations(gold, np.asarray(row["x"])).sum() == 0)
        assert not row["semantic_match"]
    assert not next(r for r in rows if r["arm"] == "base_cp_sat")["feasible"]


@pytest.mark.parametrize("steps", [1, 2, 3, 5, 9, 17, 64])
def test_final_hardening_is_used_and_no_feedback_is_matched(steps):
    spec = tiny("ge", 3)  # impossible, feedback rates always positive
    active = solve(spec, steps=steps, chains=4)
    control = solve(spec, steps=steps, chains=4, feedback=False)
    assert min(active["history"][-1]["weights"]) >= 2
    assert control["final_weights"] == [2]
    assert all(row["weights"][0] in (.1, 2) for row in control["history"])
    assert active["bit_updates"] == control["bit_updates"]


def test_fixed_policy_identical_when_feedback_disabled():
    a = solve(tiny(), policy="fixed", steps=64, chains=8)
    b = solve(tiny(), policy="fixed", steps=64, chains=8, feedback=False)
    assert a["x"] == b["x"] and a["history"] == b["history"]


def test_missing_and_duplicate_results_rejected(tmp_path):
    from experiments.generic_energy import audit_evaluation, write_rows, summarize
    case = make_case("knapsack", 4, 1, "test")
    row = {"case_id": case["id"], "arm": "direct", "seed": None}
    path = tmp_path / "direct.jsonl"
    write_rows(path, [])
    with pytest.raises(RuntimeError, match="Incomplete"):
        audit_evaluation(path, [case], "direct", "direct", 1)
    write_rows(path, [row, row])
    with pytest.raises(RuntimeError, match="Incomplete"):
        audit_evaluation(path, [case], "direct", "direct", 1)
    with pytest.raises(RuntimeError, match="Duplicate"):
        summarize(tmp_path)


def test_solver_exception_is_not_misreported_as_llm_failure(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from experiments import generic_energy as runner
    case = make_case("knapsack", 4, 1, "test")
    class Good:
        calls = input_tokens = output_tokens = 0
        def __init__(self, args):
            pass
    monkeypatch.setattr(runner, "Generator", Good)
    monkeypatch.setattr(runner, "infer", lambda *a: (case["gold"], "fixed", []))
    def fail(*args):
        raise ValueError("backend broken")
    monkeypatch.setattr(runner, "reference_solve", fail)
    args = SimpleNamespace(output=tmp_path, tag="base", mode="llm", holdout_family="balanced_cut", seeds=1, reference_seconds=1)
    with pytest.raises(ValueError, match="backend broken"):
        runner.evaluate(args, [case])
    assert not (tmp_path / "base.jsonl").exists()


def test_unstable_teacher_cannot_generate_preference(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from experiments import generic_energy as runner
    case = make_case("knapsack", 4, 1)
    def fake(spec, policy, seed, **kwargs):
        return {"feasible": seed != 47, "objective": {"fixed": 10, "adaptive": 8, "warm": 6, "cold": 4}[policy], "violation": int(seed == 47)}
    monkeypatch.setattr(runner, "solve", fake)
    args = SimpleNamespace(output=tmp_path, holdout_family="balanced_cut", chains=2, steps=2)
    runner.teacher(args, [case])
    assert runner.read_rows(tmp_path / "dpo.jsonl") == []
    assert runner.audit_teacher(tmp_path, [case], "balanced_cut")["sft_rows"] == 2
    rows = runner.read_rows(tmp_path / "sft.jsonl")
    rows[1]["completion"] = json.dumps({"constraints": [], "policy": "fixed"})
    runner.write_rows(tmp_path / "sft.jsonl", rows)
    with pytest.raises(RuntimeError, match="semantics"):
        runner.audit_teacher(tmp_path, [case], "balanced_cut")


def test_mixed_constraints_compose_without_cross_penalty_terms():
    spec = tiny()
    spec["constraints"] = [{"id": "a", "a": [1, 1], "op": "ge", "rhs": 1},
                           {"id": "b", "a": [1, -1], "op": "eq", "rhs": 0},
                           {"id": "c", "a": [-2, 1], "op": "le", "rhs": 0}]
    e = compile_energy(spec)
    weights = np.array([1.01, 4, 2])
    bits = np.array(list(itertools.product([0, 1], repeat=e.bits)))
    q, c = e.qubo(weights)
    values = np.einsum("bi,ij,bj->b", bits, q, bits) + c
    direct = -objective(spec, bits[:, :2]) / e.bound + ((bits @ e.residual_a.T - e.rhs) ** 2) @ weights
    np.testing.assert_allclose(values, direct, atol=1e-10)
    assert np.all(violations(spec, bits[np.isclose(values, values.min()), :2]) == 0)


def test_case_pairing_includes_deterministic_baselines_once(tmp_path):
    from experiments.generic_energy import write_rows, summarize
    common = {"case_id": "k1", "family": "knapsack", "split": "test", "objective_sense": "max", "feasible": True, "seconds": 1}
    write_rows(tmp_path / "base.jsonl", [{**common, "arm": "base", "seed": s, "objective": 10} for s in range(3)])
    write_rows(tmp_path / "direct.jsonl", [{**common, "arm": "direct", "seed": None, "objective": 12}])
    summary = summarize(tmp_path)
    paired = next(r for r in summary["case_paired_vs_base"] if r["arm"] == "direct")
    assert paired["paired_cases"] == 1
    assert paired["mean_relative_improvement_both_all_feasible"] == pytest.approx(.2)
