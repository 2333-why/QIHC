import json

import pytest
import numpy as np

from qihc.problems.cvrp.instance import ConstraintSpec, generate_synthetic_instance
from qihc.s2e import CPPValidator, ConstraintProgramPackage, FeedbackRecord, PDitMFCSampler, Representation, TorchPDitMFCSampler, build_dpo_pairs, compile_cpp, repair_cpp
from qihc.problems.cvrp import KNNNeighborhoodSelector, greedy_initial_solution
from qihc.problems.cvrp import PBitLogitFeedback
from qihc.s2e.synthesizer import ConstraintSynthesizer, HeuristicConstraintBackend


def test_cpp_roundtrip_checksum_and_a0_a4():
    instance = generate_synthetic_instance(6, 3, 30, seed=4)
    cpp = ConstraintProgramPackage.from_specs(instance.name, instance.description, instance.customer_ids, instance.constraints)
    restored = ConstraintProgramPackage.from_dict(json.loads(cpp.canonical_json()))
    assert restored.checksum == cpp.checksum
    report = CPPValidator(exact_customer_limit=6).validate(cpp, instance)
    assert report.passed, report.to_dict()


def test_heuristic_constraint_synthesis():
    instance = generate_synthetic_instance(6, 3, 30, seed=1, semantic_constraints=False)
    instance.description = "客户 1 和客户 2 必须同车；客户 3 和客户 4 不能同车；客户 5 必须先于客户 6。"
    cpp = ConstraintSynthesizer(HeuristicConstraintBackend()).synthesize(instance)
    assert [p.type for p in cpp.programs] == ["same_resource", "mutual_exclusion", "precedence"]


def test_compile_rejects_unvalidated_cpp_and_selects_representations():
    instance = generate_synthetic_instance(6, 3, 30, seed=2)
    cpp = ConstraintProgramPackage.from_specs(instance.name, instance.description, instance.customer_ids, instance.constraints)
    validator = CPPValidator(exact_customer_limit=6)
    report = validator.validate(cpp, instance)
    plan = compile_cpp(cpp, instance, report)
    assert len(plan.constraints) == len(cpp.programs)
    assert all(c.representation in set(Representation) for c in plan.constraints)
    with pytest.raises(ValueError):
        compile_cpp(cpp, instance, validator.validate(cpp, instance, through=0))


def test_feedback_builds_preference_pairs():
    good = FeedbackRecord("q", "good", True, True, True, gap=0.1)
    bad = FeedbackRecord("q", "bad", True, False, False, counterexamples=[{"x": 1}])
    pairs = build_dpo_pairs([bad, good])
    assert pairs[0]["chosen"] == "good"
    assert pairs[0]["chosen_reward"] > pairs[0]["rejected_reward"]


def test_auto_repair_drops_unknown_entities_and_records_actions():
    instance = generate_synthetic_instance(6, 3, 30, seed=3, semantic_constraints=False)
    bad = ConstraintProgramPackage.from_specs(instance.name, "bad", instance.customer_ids, [ConstraintSpec("same_vehicle", params={"entities": [1, 99]})])
    validator = CPPValidator(exact_customer_limit=6); report = validator.validate(bad, instance)
    fixed = repair_cpp(bad, report)
    assert fixed.repair_history
    assert not fixed.programs
    assert validator.validate(fixed, instance).passed


def test_pdit_mfc_sampler_executes_assignment_subproblem():
    instance = generate_synthetic_instance(8, 3, 30, seed=5, semantic_constraints=False)
    incumbent = greedy_initial_solution(instance)
    proposal = KNNNeighborhoodSelector(4, 3).propose(instance, incumbent, 0, 5)
    batch = PDitMFCSampler(num_chains=12, steps=3, top_k=4, seed=5).solve(instance, incumbent, proposal)
    assert batch.assignments
    assert set(batch.assignments[0]) == set(proposal.destroy_customers)
    assert batch.metadata["backend"] == "pdit-mfc-numpy"


def test_torch_pdit_mfc_sampler_uses_one_compute_dtype():
    pytest.importorskip("torch")
    instance = generate_synthetic_instance(8, 3, 30, seed=6, semantic_constraints=False)
    incumbent = greedy_initial_solution(instance)
    proposal = KNNNeighborhoodSelector(4, 3).propose(instance, incumbent, 0, 6)
    fixed = [customer for customer in instance.customer_ids if customer not in proposal.destroy_customers]
    assert len(fixed) >= 2
    instance.constraints.append(
        ConstraintSpec("same_resource", params={"entities": fixed[:2]})
    )
    batch = TorchPDitMFCSampler(
        num_chains=12, steps=3, top_k=4, seed=6, device="cpu"
    ).solve(instance, incumbent, proposal)
    assert batch.assignments
    assert batch.metadata["backend"] == "pdit-mfc-torch"


def test_pbit_feedback_updates_structural_llm_logits():
    instance = generate_synthetic_instance(8, 3, 30, seed=7, semantic_constraints=False)
    incumbent = greedy_initial_solution(instance)
    proposal = KNNNeighborhoodSelector(2, 2).propose(instance, incumbent, 0, 7)
    variables = [
        ("assign", customer, route)
        for customer in proposal.destroy_customers
        for route in proposal.candidate_routes[customer]
    ]
    bits = np.zeros((8, len(variables)), dtype=np.int8)
    first_customer = proposal.destroy_customers[0]
    preferred = proposal.candidate_routes[first_customer][0]
    alternate = proposal.candidate_routes[first_customer][1]
    bits[:2, variables.index(("assign", first_customer, preferred))] = 1
    bits[2:, variables.index(("assign", first_customer, alternate))] = 1
    controller = PBitLogitFeedback(learning_rate=1.0)
    updates = controller.observe(
        proposal, variables, bits, accepted=True, improvement=5.0,
        feasible_mask=np.ones(8, dtype=bool),
        objectives=np.asarray([1.0, 1.1, 5.0, 5.1, 5.2, 5.3, 5.4, 5.5]),
    )
    assert updates[first_customer][preferred] > 0.0
    assert controller.prompt_context([first_customer])[first_customer][preferred] > 0.0
