import json
import sys
import numpy as np
import pytest

from experiments.run_research_suite import StageRunner, SuiteLock
from experiments.prepare_joint_training_data import inference_domain, align_teacher_domain
from qihc.problems.cvrp import LNSConfig, QIHCLNSSolver
from qihc.problems.cvrp.instance import Customer, ConstraintSpec, CVRPInstance, RouteSolution, generate_synthetic_instance
from qihc.problems.cvrp.neighborhood import NeighborhoodProposal, canonicalize_proposal
from qihc.problems.cvrp.qubo import build_assignment_qubo, decode_assignment
from qihc.problems.cvrp.verifier import verify_solution
from qihc.problems.cvrp.cold_start import _sanitize_proposal
from qihc.problems.cvrp.training_reward import rollout_reward
from qihc.problems.cvrp.llm_selector import LocalLLMNeighborhoodSelector, bounded_number
from qihc.problems.cvrp.distribution_feedback import trust_region_step
from qihc.problems.cvrp.json_output import extract_answer_json


def test_time_budget_continues_past_zero_iteration_ceiling():
    instance = generate_synthetic_instance(4, 2, 40, seed=3, semantic_constraints=False)
    result = QIHCLNSSolver(LNSConfig(iterations=0, time_budget_only=True, search_time_limit_s=0.04,
        patience=0, destroy_size=2, routes_per_customer=2, num_chains=4, top_samples=2, sampling_steps=2)).solve(instance)
    assert result.records
    assert result.stop_reason == "time_limit"
    assert result.search_elapsed_s >= 0.04
    assert result.verification.feasible


def test_stage_corruption_is_rebuilt_instead_of_trusted(tmp_path):
    runner = StageRunner(tmp_path, False)
    artifact = tmp_path / "artifact.json"
    command = [sys.executable, "-c", "from pathlib import Path; import sys; Path(sys.argv[1]).write_text('{}')", str(artifact)]
    runner.run("check", command, artifact)
    artifact.write_text("corrupted", encoding="utf-8")
    runner.run("check", command, artifact)
    assert json.loads(artifact.read_text()) == {}
    marker = json.loads((tmp_path / "stages/check.json").read_text())
    assert marker["artifact_sha256"]


def test_two_launchers_cannot_own_same_result_directory(tmp_path):
    first = SuiteLock(tmp_path)
    try:
        with pytest.raises(RuntimeError, match="Another suite"):
            SuiteLock(tmp_path)
    finally:
        first.release()


def test_strict_decoder_rejects_missing_assignments_that_repair_mode_completes():
    instance = generate_synthetic_instance(4, 2, 40, seed=4, semantic_constraints=False)
    problem = build_assignment_qubo(instance, RouteSolution([[1, 2], [3, 4]]), NeighborhoodProposal([1], {1: [0, 1]}))
    empty = np.zeros(problem.num_variables, dtype=np.int8)
    assert verify_solution(instance, decode_assignment(instance, problem, empty)).feasible
    with pytest.raises(ValueError, match="exactly one"):
        decode_assignment(instance, problem, empty, strict=True)


def test_cold_neutralization_preserves_domain_order_as_well_as_zero_logits():
    domain = NeighborhoodProposal([1], {1: [2, 0, 1]})
    proposed = NeighborhoodProposal([1], {1: [1, 0, 2]}, confidence=0.1,
                                    candidate_route_logits={1: {1: 1, 0: -1, 2: 0}})
    sanitized = _sanitize_proposal(domain, proposed, [[1]], confidence_threshold=0.55, max_abs_logit=1)
    assert sanitized.candidate_routes == domain.candidate_routes
    assert all(value == 0 for value in sanitized.candidate_route_logits[1].values())


def test_json_permutation_is_not_a_search_signal():
    left = canonicalize_proposal(NeighborhoodProposal([2, 1], {2: [1, 0], 1: [0, 1]}))
    right = canonicalize_proposal(NeighborhoodProposal([1, 2], {1: [1, 0], 2: [0, 1]}))
    assert left.destroy_customers == right.destroy_customers
    assert left.candidate_routes == right.candidate_routes


def test_training_reward_allows_different_legal_destroy_subsets():
    instance = generate_synthetic_instance(4, 2, 40, seed=4, semantic_constraints=False)
    context = {"instance": instance.to_dict(), "routes": [[1, 2], [3, 4]],
               "allowed_routes": {"1": [0, 1], "2": [0, 1]}, "required_destroy_count": 1, "seed": 4}
    for customer in (1, 2):
        proposal = {"destroy_customers": [customer], "candidate_routes": {str(customer): [0, 1]}}
        assert rollout_reward(proposal, context, chains=4, steps=3) >= 0


def test_distillation_never_invents_route_ids_absent_from_prompt():
    features = [{"customer": 1, "current_route": 0, "moves": [{"route": 1}]}]
    prompt = "Objective-aware move features: " + json.dumps(features) + "\n"
    allowed = inference_domain(prompt, {"1": [0, 1, 9]})
    teacher = align_teacher_domain({"destroy_customers": [1], "candidate_routes": {"1": [0, 1, 9]},
                                   "candidate_route_logits": {"1": {"9": 2, "0": 0, "1": 1}}}, allowed)
    assert teacher["candidate_routes"]["1"] == [0, 1]
    assert "9" not in teacher["candidate_route_logits"]["1"]


def test_model_features_include_joint_capacity_release_moves():
    instance = CVRPInstance("full-routes", Customer(0, 0, 0, 0),
                            {1: Customer(1, 1, 0, 5), 2: Customer(2, 2, 0, 5)}, 2, 5)
    selector = LocalLLMNeighborhoodSelector.__new__(LocalLLMNeighborhoodSelector)
    selector.routes_per_customer = 2; selector.destroy_size = 2
    rows = selector._move_features(instance, RouteSolution([[1], [2]]))
    assert len(rows) == 2
    assert all(row["moves"][0]["capacity_release_required"] == 5 for row in rows)


def test_time_window_from_constraint_ir_is_enforced_with_waiting():
    instance = CVRPInstance("time-window", Customer(0, 0, 0, 0), {1: Customer(1, 1, 0, 1)}, 1, 5,
                            [ConstraintSpec("time_window", params={"entity": 1, "interval": [10, 20]})])
    valid = verify_solution(instance, RouteSolution([[1]]))
    assert valid.feasible and valid.arrival_times[1] == 10
    instance.constraints = [ConstraintSpec("time_window", params={"entity": 1, "interval": [0, 0.5]})]
    assert not verify_solution(instance, RouteSolution([[1]])).feasible


def test_unknown_constraint_and_nonfinite_model_scores_cannot_pass_silently():
    instance = generate_synthetic_instance(4, 2, 40, seed=4, semantic_constraints=False)
    instance.constraints = [ConstraintSpec("unsupported_new_constraint")]
    assert not verify_solution(instance, RouteSolution([[1, 2], [3, 4]])).feasible
    with pytest.raises(ValueError):
        bounded_number(float("nan"), -2, 2)


@pytest.mark.parametrize("seed", range(8))
def test_kl_projection_random_extreme_distributions(seed):
    rng = np.random.default_rng(seed)
    p = rng.dirichlet(np.full(8, 0.2)); q = rng.dirichlet(np.full(8, 0.2))
    _, actual = trust_region_step(p, q, 0.02)
    assert np.isfinite(actual) and actual <= 0.020000001
    supported = p > 0
    step, _ = trust_region_step(p, q, 0.02)
    from qihc.problems.cvrp.distribution_feedback import softmax
    updated = softmax(np.log(p[supported]) + step[supported])
    actual_against_original = np.sum(updated * (np.log(updated) - np.log(p[supported])))
    assert actual_against_original <= 0.020000001


def test_kl_budget_is_against_original_tiny_prior_not_a_clipped_surrogate():
    from qihc.problems.cvrp.distribution_feedback import softmax
    prior = np.array([1 - 1e-20, 1e-20])
    step, reported = trust_region_step(prior, np.array([0.01, 0.99]), 0.03)
    updated = softmax(np.log(prior) + step)
    actual = np.sum(updated * (np.log(updated) - np.log(prior)))
    assert actual <= 0.030000001
    assert abs(actual - reported) < 1e-12
    with pytest.raises(ValueError):
        trust_region_step(prior, prior, -1)


def test_reasoning_examples_and_nested_json_are_not_mistaken_for_final_answer():
    text = '<think>{"routes":[[9]]}</think>Final: ```json\n{"candidate_routes":{"1":[0,1]}}\n```'
    assert extract_answer_json(text) == {"candidate_routes": {"1": [0, 1]}}
    text = '{"routes":[[9]]} Updated answer: {"routes":[[1,2]]}'
    assert extract_answer_json(text)["routes"] == [[1, 2]]


def test_truncated_outer_answer_does_not_return_its_nested_domain():
    from qihc.problems.cvrp.llm_selector import _extract_partial_neighborhood_json
    text = '{"destroy_customers":[1],"candidate_routes":{"1":[0,1]},"candidate_route_logits":'
    with pytest.raises(ValueError):
        extract_answer_json(text)
    assert _extract_partial_neighborhood_json(text)["candidate_routes"] == {"1": [0, 1]}


def test_soft_constraint_weights_are_part_of_semantic_accuracy():
    from experiments.run_s2e_pipeline import canonical_constraint, semantic_closure
    assert canonical_constraint("preferred_time", False, {"entity": 1}, 1) != canonical_constraint("preferred_time", False, {"entity": 1}, 100)
    soft_precedence = canonical_constraint("precedence", False, {"before": 1, "after": 2}, 5)
    assert semantic_closure({soft_precedence}) == {soft_precedence}


def test_reference_objective_includes_true_soft_penalties(tmp_path, monkeypatch):
    import copy
    from experiments.run_cvrp_lns import parse_args, run_job
    instance = generate_synthetic_instance(4, 2, 40, seed=4, semantic_constraints=False)
    instance.constraints = [ConstraintSpec("preferred_time", hard=False, weight=100,
                                          params={"entity": 1, "interval": [0, 0]})]
    compiled = copy.deepcopy(instance)
    compiled.constraints = [ConstraintSpec("preferred_time", hard=False, weight=1,
                                          params={"entity": 1, "interval": [0, 0]})]
    monkeypatch.setattr(sys, "argv", ["run", "--output", str(tmp_path), "--sampler", "numpy", "--iterations", "0"])
    result = run_job(compiled, "knn", 0, parse_args(), None, 0, reference_instance=instance)
    assert result["final_objective"] == pytest.approx(result["compiled_objective"] + 99)


@pytest.mark.parametrize("hard", ["false", 1, None])
def test_constraint_ir_boolean_flags_cannot_silently_change_semantics(hard):
    with pytest.raises(ValueError):
        ConstraintSpec.from_dict({"type": "same_resource", "hard": hard, "params": {"entities": [1, 2]}})


def test_hard_constraint_weight_cannot_switch_off_its_energy():
    instance = generate_synthetic_instance(4, 2, 40, seed=4, semantic_constraints=False)
    instance.constraints = [ConstraintSpec("same_resource", hard=True, weight=0, params={"entities": [1, 2]})]
    current = RouteSolution([[1, 2], [3, 4]])
    proposal = NeighborhoodProposal([1, 2], {1: [0, 1], 2: [0, 1]})
    zero = build_assignment_qubo(instance, current, proposal)
    instance.constraints = [ConstraintSpec("same_resource", hard=True, weight=1, params={"entities": [1, 2]})]
    one = build_assignment_qubo(instance, current, proposal)
    assert zero.model.linear == one.model.linear and zero.model.quadratic == one.model.quadratic


def test_extra_capacity_rule_is_not_falsely_reported_as_lowered_into_base_capacity():
    from qihc.s2e.cpp import ConstraintProgramPackage
    from qihc.s2e.validator import CPPValidator
    from qihc.s2e.compiler import compile_cpp
    instance = generate_synthetic_instance(4, 2, 40, seed=4, semantic_constraints=False)
    instance.constraints = [ConstraintSpec("capacity", params={"limit": 20})]
    cpp = ConstraintProgramPackage.from_specs(instance.name, "capacity 20", instance.customer_ids, instance.constraints)
    plan = compile_cpp(cpp, instance, CPPValidator().validate(cpp, instance))
    assert plan.constraints[0].execution == "verifier_only"
