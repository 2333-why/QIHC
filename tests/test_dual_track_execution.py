from argparse import Namespace

import numpy as np

from experiments.run_cvrp_lns import prepare_compiled_instance
from experiments.prepare_hard_nlcvrp import add_constraints, align_solution_ids
from experiments.download_hard_cvrplib import discover
from qihc.problems.cvrp import generate_synthetic_instance
from qihc.problems.cvrp.baselines import direct_llm_prompt, parse_direct_llm_solution
from qihc.problems.cvrp.cold_start import _co_route_groups, construct_with_pbit
from qihc.problems.cvrp.instance import RouteSolution
from qihc.problems.cvrp.verifier import verify_solution
from qihc.problems.cvrp.llm_selector import PBitRouteTokenLogitsProcessor
from qihc.problems.cvrp.scheduler import LNSConfig, QIHCLNSSolver
from qihc.ising.batched import PBitSampleBatch
from qihc.s2e.cpp import ConstraintProgramPackage


class FakeTokenizer:
    def encode(self, value, add_special_tokens=False):
        return [int(value)]

    def decode(self, ids, skip_special_tokens=True):
        return self.text


def test_feedback_changes_only_route_token_logits():
    tokenizer = FakeTokenizer()
    processor = PBitRouteTokenLogitsProcessor(tokenizer, 0, {7: {2: 1.5}}, strength=2.0)
    scores = np.zeros((1, 10))
    tokenizer.text = '{"candidate_routes":{"7":['
    changed = processor(np.array([[1, 2]]), scores)
    assert changed[0, 2] == 3.0
    assert changed[0, 7] == 0.0
    tokenizer.text = '{"confidence":'
    assert processor(np.array([[1, 2]]), np.zeros((1, 10)))[0, 2] == 0.0


def test_cpp_is_used_by_verifier_and_compiled_sampler():
    instance = generate_synthetic_instance(12, 4, 25, seed=13, semantic_constraints=True)
    cpp = ConstraintProgramPackage.from_specs(
        instance.name, instance.description, instance.customer_ids, instance.constraints
    )
    compiled = prepare_compiled_instance(
        instance, Namespace(constraint_source="cpp"), None,
        {instance.name: {"status": "ok", "cpp": cpp.to_dict()}},
    )
    assert compiled is not instance
    assert compiled.metadata["constraint_compilation"]["constraints"]
    result = QIHCLNSSolver(LNSConfig(
        iterations=1, sampling_steps=5, num_chains=4, top_samples=2,
    )).solve(compiled)
    assert result.verification.feasible
    assert result.records[0].sampler_backend == "pdit-mfc-numpy"


def test_pbit_cold_start_without_supplied_incumbent():
    instance = generate_synthetic_instance(20, 5, 30, seed=11, semantic_constraints=True)
    result = QIHCLNSSolver(LNSConfig(
        iterations=2, sampling_steps=20, num_chains=32, top_samples=16,
        initialization="pbit-cold", cold_batch_size=4,
    )).solve(instance)
    assert result.verification.feasible
    assert result.construction_batches > 0
    assert result.construction_pbit_s > 0


def test_compiled_cold_start_uses_hybrid_sampler():
    instance = generate_synthetic_instance(20, 5, 30, seed=11, semantic_constraints=True)
    cpp = ConstraintProgramPackage.from_specs(
        instance.name, instance.description, instance.customer_ids, instance.constraints
    )
    compiled = prepare_compiled_instance(
        instance, Namespace(constraint_source="cpp"), None,
        {instance.name: {"status": "ok", "cpp": cpp.to_dict()}},
    )
    result = QIHCLNSSolver(LNSConfig(
        iterations=1, sampling_steps=8, num_chains=8, top_samples=4,
        initialization="pbit-cold", cold_batch_size=4,
    )).solve(compiled)
    assert result.verification.feasible
    assert result.construction_batches > 0
    assert result.records[0].sampler_backend == "pdit-mfc-numpy"


def test_cold_start_keeps_hard_co_route_constraints_atomic():
    instance = generate_synthetic_instance(20, 5, 30, seed=11, semantic_constraints=True)
    groups = _co_route_groups(instance)
    assert any({1, 2}.issubset(group) for group in map(set, groups))
    assert any({5, 6}.issubset(group) for group in map(set, groups))


def test_cold_start_has_batch_guard_when_sampler_returns_no_candidates():
    instance = generate_synthetic_instance(20, 5, 30, seed=11, semantic_constraints=True)

    class EmptySampler:
        num_chains = 1
        steps = 1
        top_k = 1

        def solve(self, weight, field, initial_bits=None):
            return PBitSampleBatch(
                bits=np.empty((0, len(field)), dtype=np.int8),
                energies=np.empty(0),
                elapsed_s=0.001,
            )

    result = construct_with_pbit(
        instance,
        lambda seed: EmptySampler(),
        batch_size=4,
        routes_per_customer=4,
        seed=7,
    )
    assert verify_solution(instance, result.solution).feasible
    assert result.solution.metadata["guarded_batches"] > 0
    assert result.solution.metadata["construction_attempts"] >= 1


def test_hard_nl_cases_do_not_export_witness_routes():
    instance = generate_synthetic_instance(18, 3, 100, seed=4, semantic_constraints=False)
    witness = RouteSolution([list(range(1, 7)), list(range(7, 13)), list(range(13, 19))])
    assert verify_solution(instance, witness).feasible
    hard = add_constraints(instance, witness, seed=12, count=1)
    assert verify_solution(hard, witness).feasible
    assert len(hard.constraints) == 3
    assert "known_feasible_routes" not in hard.to_dict()["metadata"]


def test_cvrplib_solution_numbering_is_aligned_to_node_ids():
    instance = generate_synthetic_instance(6, 2, 100, seed=3, semantic_constraints=False)
    instance.depot = type(instance.depot)(1, 0.0, 0.0, 0)
    instance.customers = {customer + 1: type(node)(customer + 1, node.x, node.y, node.demand)
                          for customer, node in instance.customers.items()}
    witness = RouteSolution([[1, 2, 3], [4, 5, 6]])
    aligned = align_solution_ids(instance, witness)
    assert aligned.metadata["solution_id_offset"] == 1
    assert verify_solution(instance, aligned).feasible


def test_official_index_discovery():
    html = '<a href="/cvrplib/en/download/instance/180" title="Instance File"> X-n204-k19 </a>'
    assert discover(html) == [("X-n204-k19", "180")]


def test_direct_llm_baseline_uses_language_and_strict_route_json():
    instance = generate_synthetic_instance(6, 2, 100, seed=3, semantic_constraints=True)
    instance.description = "客户 1 与客户 2 必须由同一辆车配送。"
    prompt = direct_llm_prompt(instance)
    assert instance.description in prompt
    assert "same_resource" not in prompt
    assert "customers_as_id_x_y_demand" in prompt

    solution = parse_direct_llm_solution('```json\n{"routes":[[1,2,3],[4,5,6]]}\n```')
    assert solution.routes == [[1, 2, 3], [4, 5, 6]]
    assert solution.source == "llm_direct_unrepaired"


def test_direct_llm_baseline_does_not_complete_missing_customers():
    instance = generate_synthetic_instance(6, 2, 100, seed=3, semantic_constraints=False)
    solution = parse_direct_llm_solution('{"routes":[[1,2],[3,4]]}')
    verification = verify_solution(instance, solution)
    assert not verification.feasible
    assert any(item["type"] == "missing_customer" for item in verification.violations)
