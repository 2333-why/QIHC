"""Execution rewards for proposal post-training; no JSON string matching."""

from __future__ import annotations

import copy
import json

import numpy as np

from qihc.ising.batched import NumpyPBitSampler
from qihc.problems.cvrp.instance import CVRPInstance, RouteSolution
from qihc.problems.cvrp.neighborhood import NeighborhoodProposal, safely_expand_proposal, canonicalize_proposal
from qihc.problems.cvrp.qubo import build_assignment_qubo, decode_assignment
from qihc.problems.cvrp.verifier import repair_supported_constraints, verify_solution


def semantic_signature(value: dict) -> set[str]:
    if not isinstance(value.get("constraints", value.get("programs")), list):
        raise ValueError("Constraint output must contain a list")
    constraints = value.get("constraints", value.get("programs", []))
    signatures = set()
    for spec in constraints:
        params = dict(spec.get("params", {}))
        kind = spec["type"]
        if kind == "same_vehicle":
            kind = "same_resource"
        if kind in {"same_resource", "mutual_exclusion"}:
            params["entities"] = sorted(int(x) for x in params["entities"])
        hard = bool(spec.get("hard", True))
        signatures.add(json.dumps([kind, hard, params, None if hard else float(spec.get("weight", 1.0))], sort_keys=True))
    return signatures


def constraint_reward(predicted: dict, gold: dict) -> float:
    actual, expected = semantic_signature(predicted), semantic_signature(gold)
    if actual == expected:
        return 2.0
    union = actual | expected
    return -1.0 + 2.0 * len(actual & expected) / max(len(union), 1)


def rollout_reward(predicted: dict, context: dict, *, chains: int = 16, steps: int = 24) -> float:
    """Use common random numbers for all completions of one state.

    The reward measures verified downstream objective improvement, not matching
    the logged teacher's text. This CPU rollout is an explicit training cost.
    """
    instance = CVRPInstance.from_dict(context["instance"])
    current = RouteSolution([list(route) for route in context["routes"]])
    allowed = {int(c): [int(r) for r in routes] for c, routes in context["allowed_routes"].items()}
    destroyed = [int(c) for c in predicted.get("destroy_customers", allowed)]
    required = int(context.get("required_destroy_count", len(allowed)))
    if (len(destroyed) != required or len(set(destroyed)) != len(destroyed)
            or any(customer not in allowed for customer in destroyed)):
        return -2.0
    raw_routes = predicted.get("candidate_routes", {})
    raw_logits = predicted.get("candidate_route_logits", predicted.get("candidate_route_scores", {}))
    routes, logits = {}, {}
    for customer in destroyed:
        selected = [int(r) for r in raw_routes.get(str(customer), raw_routes.get(customer, []))]
        if not selected or any(r not in allowed[customer] for r in selected):
            return -2.0
        routes[customer] = list(dict.fromkeys(selected))
        values = raw_logits.get(str(customer), raw_logits.get(customer, {}))
        logits[customer] = {r: float(np.clip(values.get(str(r), values.get(r, 0.0)), -2, 2)) for r in routes[customer]}
        if not all(np.isfinite(value) for value in logits[customer].values()):
            return -2.0
    proposal = NeighborhoodProposal(destroyed, routes, candidate_route_logits=logits)
    proposal.validate(instance)
    cfg = context.get("solver_config", {})
    partial = bool(context.get("partial"))
    normalize = cfg.get("normalize_objective", True) if not partial else False
    bias = float(cfg.get("proposal_bias", 0.25)) if not partial else float(cfg.get("cold_proposal_bias", 0.25))
    if partial:
        from qihc.problems.cvrp.cold_start import _co_route_groups, _sanitize_proposal
        confidence = float(predicted.get("confidence", 0.5))
        if not np.isfinite(confidence):
            return -2.0
        proposal.confidence = float(np.clip(confidence, 0, 1))
        proposal.candidate_route_logits = {c: {r: float(np.clip(v, -1, 1)) for r, v in values.items()}
                                          for c, values in proposal.candidate_route_logits.items()}
        groups = [group for group in _co_route_groups(instance) if set(group) & set(destroyed)]
        if any(not set(group) <= set(destroyed) for group in groups):
            return -2.0
        domain = NeighborhoodProposal(destroyed, {c: allowed[c] for c in destroyed})
        proposal = _sanitize_proposal(domain, proposal, groups,
            confidence_threshold=float(cfg.get("cold_confidence_threshold", 0.55)),
            max_abs_logit=float(cfg.get("cold_max_abs_logit", 1.0)))
        for customer in destroyed:
            candidates = context.get("cold_context", {}).get("customers", {}).get(str(customer), {}).get("candidates", [])
            deltas = [float(row["insertion_delta"]) for row in candidates]
            scale = max(1.0, max(deltas) - min(deltas)) if deltas else 1.0
            proposal.candidate_route_logits[customer] = {route: value * scale for route, value in proposal.candidate_route_logits[customer].items()}
    else:
        current_routes = {c: r for r, route in enumerate(current.routes) for c in route}
        limit = int(cfg.get("routes_per_customer", instance.vehicle_count))
        for customer in destroyed:
            choices = proposal.candidate_routes[customer][:limit]
            if current_routes[customer] not in choices:
                if len(choices) >= limit:
                    choices[-1] = current_routes[customer]
                else:
                    choices.append(current_routes[customer])
            proposal.candidate_routes[customer] = choices
            proposal.candidate_route_logits[customer] = {r: proposal.candidate_route_logits[customer].get(r, 0.0) for r in choices}
        if cfg.get("safe_candidate_expansion", True):
            proposal = safely_expand_proposal(instance, current, proposal,
                heuristic_routes=int(cfg.get("safe_heuristic_routes", 1)), random_routes=int(cfg.get("safe_random_routes", 1)),
                seed=int(context.get("seed", 0)) + 3571 * int(context.get("iteration", 0)))
    feedback = context.get("energy_feedback_context", {})
    if not partial:
        proposal = canonicalize_proposal(proposal)
    if feedback and not partial:
        from qihc.problems.cvrp.distribution_feedback import DistributionalFeedback
        adapter = DistributionalFeedback(kl_budget=float(cfg.get("feedback_kl_budget", 0.05)),
                                         use_trust_region=bool(cfg.get("feedback_trust_region", True)))
        adapter.residuals = {int(c): {int(r): float(v) for r, v in values.items()} for c, values in feedback.items()}
        for customer in destroyed:
            proposal.candidate_route_logits[customer] = adapter.adjusted_logits(customer, proposal.candidate_route_logits[customer])
    semantic_penalty = 20.0 if partial else float(cfg.get("semantic_penalty", 20.0))
    problem = build_assignment_qubo(instance, current, proposal, proposal_bias=bias, normalize_objective=normalize,
                                   assignment_penalty=40.0 if partial else float(cfg.get("assignment_penalty", 40.0)),
                                   capacity_penalty=12.0 if partial else float(cfg.get("capacity_penalty", 12.0)),
                                   semantic_penalty=semantic_penalty)
    from qihc.s2e.hybrid_sampler import PDitMFCSampler
    seed = int(context.get("seed", 0)) + 7919 * int(context.get("iteration", 0))
    if "ising" in context.get("sampler_backend", ""):
        weight, field, _ = problem.model.to_ising()
        sampled_bits = NumpyPBitSampler(num_chains=chains, steps=steps, top_k=min(chains, 8), seed=seed).solve(weight, field, problem.initial_bits).bits
    else:
        try:
            sampled = PDitMFCSampler(num_chains=chains, steps=steps, top_k=min(chains, 8),
                                    seed=seed, proposal_bias=bias,
                                    semantic_penalty=semantic_penalty,
                                    block_updates=not partial and cfg.get("block_updates", True),
                                    temperature_ladder=not partial and cfg.get("temperature_ladder", True),
                                    normalize_objective=normalize).solve(instance, current, proposal)
        except ValueError:
            return -1.0
        sampled_bits = np.zeros((len(sampled.assignments), problem.num_variables), dtype=np.int8)
        for row_idx, assignment in enumerate(sampled.assignments):
            for customer, route in assignment.items():
                sampled_bits[row_idx, problem.model.index[("assign", customer, route)]] = 1
    baseline = verify_solution(instance, current)
    best = baseline.objective if baseline.feasible else None
    feasible = 0
    for bits in sampled_bits:
        try:
            decoded = decode_assignment(instance, problem, bits, strict=not partial and cfg.get("strict_assignment_decode", False))
            # For cold batches, unassigned future customers are not part of this rollout.
            check_instance = instance
            if context.get("partial"):
                from qihc.problems.cvrp.cold_start import _applicable
                assigned = {c for route in decoded.routes for c in route}
                check_instance = copy.copy(instance)
                check_instance.customers = {c: instance.customers[c] for c in assigned}
                check_instance.constraints = [c for c in instance.constraints if _applicable(c, assigned)]
            if partial or cfg.get("constraint_aware_decode", True):
                decoded = repair_supported_constraints(check_instance, decoded)
            verified = verify_solution(check_instance, decoded)
        except ValueError:
            continue
        if verified.feasible:
            feasible += 1
            best = verified.objective if best is None else min(best, verified.objective)
    if not feasible:
        return -1.0
    reference = float(context.get("reference_objective", baseline.objective))
    gain = (reference - float(best)) / max(abs(reference), 1.0)
    return float(1.0 + np.clip(10.0 * gain, -1.0, 1.0))
