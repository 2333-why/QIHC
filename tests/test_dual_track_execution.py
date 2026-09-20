from argparse import Namespace

import numpy as np

from experiments.run_cvrp_lns import prepare_compiled_instance
from experiments.prepare_hard_nlcvrp import add_constraints
from qihc.problems.cvrp import generate_synthetic_instance
from qihc.problems.cvrp.instance import RouteSolution
from qihc.problems.cvrp.verifier import verify_solution
from qihc.problems.cvrp.llm_selector import PBitRouteTokenLogitsProcessor
from qihc.problems.cvrp.scheduler import LNSConfig, QIHCLNSSolver
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


def test_hard_nl_cases_do_not_export_witness_routes():
    instance = generate_synthetic_instance(18, 3, 100, seed=4, semantic_constraints=False)
    witness = RouteSolution([list(range(1, 7)), list(range(7, 13)), list(range(13, 19))])
    assert verify_solution(instance, witness).feasible
    hard = add_constraints(instance, witness, seed=12, count=1)
    assert verify_solution(hard, witness).feasible
    assert len(hard.constraints) == 3
    assert "known_feasible_routes" not in hard.to_dict()["metadata"]
