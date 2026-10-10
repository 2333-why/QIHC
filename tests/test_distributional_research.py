import json

import numpy as np
import pytest

from qihc.problems.cvrp.distribution_feedback import DistributionalFeedback, softmax, trust_region_step
from qihc.problems.cvrp.instance import CVRPInstance, Customer, ConstraintSpec, RouteSolution, generate_synthetic_instance
from qihc.problems.cvrp.neighborhood import NeighborhoodProposal
from qihc.problems.cvrp.qubo import build_assignment_qubo
from qihc.problems.cvrp.classical_assignment import solve_classical_assignment
from qihc.problems.cvrp.scheduler import LNSConfig, QIHCLNSSolver
from qihc.problems.cvrp.training_reward import constraint_reward, rollout_reward
from qihc.s2e.hybrid_sampler import PDitMFCSampler, TorchPDitMFCSampler


def test_trust_region_is_bounded_when_prior_is_confident_and_reused():
    prior = np.asarray([0.99, 0.005, 0.005])
    target = np.asarray([0.01, 0.98, 0.01])
    step, divergence = trust_region_step(prior, target, 0.03)
    assert divergence <= 0.030000001
    assert step[1] > step[0]
    feedback = DistributionalFeedback(kl_budget=0.03)
    feedback.residuals = {1: {0: -10, 1: 10, 2: 0}}
    base = {0: 7, 1: -7, 2: -3}
    adjusted = feedback.adjusted_logits(1, base)
    p = softmax(np.asarray(list(base.values())))
    q = softmax(np.asarray(list(adjusted.values())))
    assert np.sum(q * np.log(q / p)) <= 0.030000001


def test_population_feedback_uses_quality_and_gates_duplicate_collapse():
    proposal = NeighborhoodProposal([1], {1: [0, 1]})
    variables = [("assign", 1, 0), ("assign", 1, 1)]
    feedback = DistributionalFeedback()
    updates = feedback.observe(proposal, variables, [[1, 0], [0, 1]], False, 0,
                               feasible_mask=[True, True], objectives=[100, 80])
    assert updates[1][1] > updates[1][0]
    updates = feedback.observe(proposal, variables, [[0, 1], [0, 1]], True, 20,
                               feasible_mask=[True, True], objectives=[80, 80])
    assert updates == {}
    assert feedback.diagnostics["gated"]


@pytest.mark.parametrize("sampler", [PDitMFCSampler, TorchPDitMFCSampler])
def test_block_sampling_preserves_hard_coroutes_in_terminal_population(sampler):
    if sampler is TorchPDitMFCSampler:
        pytest.importorskip("torch")
    instance = generate_synthetic_instance(4, 2, 40, seed=12, semantic_constraints=False)
    instance.constraints = [ConstraintSpec("same_resource", params={"entities": [1, 2]})]
    current = RouteSolution([[1, 2], [3, 4]])
    proposal = NeighborhoodProposal([1, 2], {1: [0, 1], 2: [0, 1]})
    kwargs = {"device": "cpu"} if sampler is TorchPDitMFCSampler else {}
    batch = sampler(num_chains=12, steps=4, top_k=4, block_updates=True, temperature_ladder=True, **kwargs).solve(instance, current, proposal)
    assert len(batch.population_assignments) == 12
    assert all(row[1] == row[2] for row in batch.population_assignments)
    assert batch.metadata["temperature_ladder"]


def test_classical_metropolis_returns_original_population_and_same_energy():
    instance = generate_synthetic_instance(4, 2, 40, seed=2, semantic_constraints=False)
    problem = build_assignment_qubo(instance, RouteSolution([[1, 2], [3, 4]]), NeighborhoodProposal([1], {1: [0, 1]}))
    batch = solve_classical_assignment(problem, "metropolis", seed=1, num_chains=8, steps=4, top_k=4, time_limit_s=1)
    assert batch.metadata["same_local_qubo"]
    assert batch.population_bits.shape == (8, problem.num_variables)


def test_objective_normalization_prevents_large_distance_from_rewarding_missing_assignments():
    instance = CVRPInstance("large-distances", Customer(0, 0, 0, 0),
                            {1: Customer(1, 10000, 0, 1), 2: Customer(2, 10000, 0, 1)}, 2, 1)
    proposal = NeighborhoodProposal([1, 2], {1: [0, 1], 2: [0, 1]})
    raw = build_assignment_qubo(instance, RouteSolution([[], []]), proposal)
    normalized = build_assignment_qubo(instance, RouteSolution([[], []]), proposal, normalize_objective=True)
    feasible = np.zeros(raw.num_variables, dtype=np.int8)
    feasible[raw.model.index[("assign", 1, 0)]] = 1
    feasible[raw.model.index[("assign", 2, 1)]] = 1
    assert raw.model.energy(feasible) > raw.model.energy(raw.initial_bits)
    assert normalized.model.energy(feasible) < normalized.model.energy(normalized.initial_bits)


def test_cp_sat_optimizes_same_qubo_with_available_dependency():
    pytest.importorskip("ortools")
    instance = generate_synthetic_instance(4, 2, 40, seed=2, semantic_constraints=False)
    problem = build_assignment_qubo(instance, RouteSolution([[1, 2], [3, 4]]), NeighborhoodProposal([1], {1: [0, 1]}))
    batch = solve_classical_assignment(problem, "cp-sat", seed=1, num_chains=8, steps=4, top_k=4, time_limit_s=1)
    assert len(batch.bits) > 0
    assert min(problem.model.energy(row) for row in batch.bits) <= problem.model.energy(problem.initial_bits) + 1e-6


def test_training_semantic_reward_ignores_order_and_metadata_but_not_wrong_meaning():
    gold = {"constraints": [{"type": "same_resource", "params": {"entities": [1, 2]}}]}
    reordered = {"constraints": [{"type": "same_vehicle", "confidence": 0.4, "params": {"entities": [2, 1]}}]}
    wrong = {"constraints": [{"type": "mutual_exclusion", "params": {"entities": [1, 2]}}]}
    assert constraint_reward(reordered, gold) == 2
    assert constraint_reward(wrong, gold) < 0


def test_rollout_reward_runs_a_verified_solution_and_rejects_invented_routes():
    instance = generate_synthetic_instance(4, 2, 40, seed=4, semantic_constraints=False)
    context = {"instance": instance.to_dict(), "routes": [[1, 2], [3, 4]], "allowed_routes": {"1": [0, 1]}, "seed": 3}
    legal = {"destroy_customers": [1], "candidate_routes": {"1": [0, 1]}}
    invalid = {"destroy_customers": [1], "candidate_routes": {"1": [9]}}
    assert rollout_reward(legal, context, chains=4, steps=3) >= 0
    assert rollout_reward(invalid, context, chains=4, steps=3) == -2
