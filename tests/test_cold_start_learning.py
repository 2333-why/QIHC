import json

from qihc.problems.cvrp.cold_start import _cold_batch_context, _sanitize_proposal
from qihc.problems.cvrp.cold_start import construct_with_pbit
from qihc.ising.batched import NumpyPBitSampler
from qihc.problems.cvrp.instance import RouteSolution, generate_synthetic_instance
from qihc.problems.cvrp.llm_selector import LocalLLMNeighborhoodSelector
from qihc.problems.cvrp.neighborhood import NeighborhoodProposal
from qihc.problems.cvrp.verifier import verify_solution


def test_cold_context_exposes_geometry_capacity_and_lookahead():
    instance = generate_synthetic_instance(6, 3, 30, seed=11, semantic_constraints=False)
    current = RouteSolution([[1], [], []])
    groups = [[2], [3]]
    domains = {2: [0, 1], 3: [0, 1]}
    context = _cold_batch_context(instance, current, groups, domains, groups + [[4]])
    assert context["routes"][0]["load"] == instance.customers[1].demand
    assert context["customers"]["2"]["coordinates"]
    assert "insertion_delta" in context["customers"]["2"]["candidates"][0]
    assert context["lookahead"]["remaining_group_count"] == 3


def test_llm_cold_scores_are_bounded_and_low_confidence_is_neutralized():
    instance = generate_synthetic_instance(4, 2, 40, seed=5, semantic_constraints=False)
    selector = LocalLLMNeighborhoodSelector.__new__(LocalLLMNeighborhoodSelector)
    selector.routes_per_customer = 2
    selector.audit_path = None
    selector._generate = lambda prompt: json.dumps({
        "candidate_routes": {"1": [1, 0]},
        "candidate_route_scores": {"1": {"0": -9, "1": 9}},
        "confidence": 0.4,
    })
    domain = NeighborhoodProposal(
        [1], {1: [0, 1]}, source="domain", raw={"cold_context": {"lookahead": {}}}
    )
    proposed = selector.propose_cold_batch(instance, RouteSolution([[], []]), domain, 0, 7)
    assert proposed.candidate_route_logits[1] == {1: 1.0, 0: -1.0}
    sanitized = _sanitize_proposal(
        domain, proposed, [[1]], confidence_threshold=0.55, max_abs_logit=1.0
    )
    assert set(sanitized.candidate_routes[1]) == {0, 1}
    assert set(sanitized.candidate_route_logits[1].values()) == {0.0}


def test_quality_multistart_keeps_only_winning_rollout_labels():
    instance = generate_synthetic_instance(8, 4, 40, seed=17, semantic_constraints=False)
    result = construct_with_pbit(
        instance,
        lambda seed: NumpyPBitSampler(num_chains=24, steps=12, top_k=6, seed=seed),
        batch_size=4,
        routes_per_customer=3,
        seed=9,
        quality_attempts=2,
    )
    records = result.solution.metadata["cold_start_records"]
    assert verify_solution(instance, result.solution).feasible
    assert result.solution.metadata["successful_construction_attempts"] == 2
    assert {record["selected_attempt"] for record in records} == {False, True}
    assert all("rollout_objective" in record for record in records)
