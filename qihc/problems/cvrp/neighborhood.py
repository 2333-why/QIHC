"""Candidate-neighborhood selectors for QIHC-LNS."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from qihc.problems.cvrp.instance import CVRPInstance, RouteSolution


@dataclass
class NeighborhoodProposal:
    destroy_customers: list[int]
    candidate_routes: dict[int, list[int]]
    confidence: float = 1.0
    source: str = "unknown"
    raw: dict = field(default_factory=dict)
    candidate_route_logits: dict[int, dict[int, float]] = field(default_factory=dict)
    candidate_edges: list[tuple[int, int]] = field(default_factory=list)

    def validate(self, instance: CVRPInstance) -> None:
        unknown = set(self.destroy_customers) - set(instance.customer_ids)
        if unknown:
            raise ValueError(f"Unknown destroy customers: {sorted(unknown)}")
        if len(set(self.destroy_customers)) != len(self.destroy_customers):
            raise ValueError("destroy_customers contains duplicates")
        for customer in self.destroy_customers:
            routes = self.candidate_routes.get(customer, [])
            if not routes:
                raise ValueError(f"Customer {customer} has no candidate routes")
            if any(route < 0 or route >= instance.vehicle_count for route in routes):
                raise ValueError(f"Customer {customer} has invalid candidate route")
            logits = self.candidate_route_logits.get(customer, {})
            if any(route not in routes for route in logits):
                raise ValueError(f"Customer {customer} has logits outside candidate routes")


class NeighborhoodSelector(Protocol):
    def propose(
        self, instance: CVRPInstance, solution: RouteSolution, iteration: int, seed: int
    ) -> NeighborhoodProposal: ...


def _customer_route_map(solution: RouteSolution) -> dict[int, int]:
    return {customer: route_idx for route_idx, route in enumerate(solution.routes) for customer in route}


class RandomNeighborhoodSelector:
    def __init__(self, destroy_size: int = 8, routes_per_customer: int = 3):
        self.destroy_size = destroy_size
        self.routes_per_customer = routes_per_customer

    def propose(self, instance: CVRPInstance, solution: RouteSolution, iteration: int, seed: int) -> NeighborhoodProposal:
        rng = np.random.default_rng(seed + 1009 * iteration)
        count = min(self.destroy_size, len(instance.customer_ids))
        destroyed = sorted(int(x) for x in rng.choice(instance.customer_ids, size=count, replace=False))
        current = _customer_route_map(solution)
        candidate_routes: dict[int, list[int]] = {}
        for customer in destroyed:
            route_ids = list(range(instance.vehicle_count))
            rng.shuffle(route_ids)
            selected = route_ids[: min(self.routes_per_customer, len(route_ids))]
            if current.get(customer) not in selected:
                selected[-1] = current[customer]
            candidate_routes[customer] = sorted(set(selected))
        return NeighborhoodProposal(destroyed, candidate_routes, source="random")


class KNNNeighborhoodSelector:
    def __init__(self, destroy_size: int = 8, routes_per_customer: int = 3):
        self.destroy_size = destroy_size
        self.routes_per_customer = routes_per_customer

    def propose(self, instance: CVRPInstance, solution: RouteSolution, iteration: int, seed: int) -> NeighborhoodProposal:
        rng = np.random.default_rng(seed + 2027 * iteration)
        anchor = int(rng.choice(instance.customer_ids))
        neighbors = sorted(instance.customer_ids, key=lambda x: (instance.distance(anchor, x), x))
        destroyed = neighbors[: min(self.destroy_size, len(neighbors))]
        current = _customer_route_map(solution)
        candidate_routes: dict[int, list[int]] = {}
        for customer in destroyed:
            route_scores: list[tuple[float, int]] = []
            for route_idx, route in enumerate(solution.routes):
                if route:
                    score = min(instance.distance(customer, other) for other in route)
                else:
                    score = instance.distance(customer, instance.depot.id)
                route_scores.append((score, route_idx))
            selected = [route for _, route in sorted(route_scores)[: self.routes_per_customer]]
            if current.get(customer) not in selected:
                selected[-1] = current[customer]
            candidate_routes[customer] = sorted(set(selected))
        return NeighborhoodProposal(destroyed, candidate_routes, source="knn")


def safely_expand_proposal(
    instance: CVRPInstance,
    solution: RouteSolution,
    proposal: NeighborhoodProposal,
    heuristic_routes: int = 1,
    random_routes: int = 1,
    seed: int = 0,
) -> NeighborhoodProposal:
    """Add incumbent, geometric and exploration routes without deleting LLM choices."""

    proposal.validate(instance)
    rng = np.random.default_rng(seed)
    current = _customer_route_map(solution)
    expanded: dict[int, list[int]] = {}
    logits = {customer: dict(values) for customer, values in proposal.candidate_route_logits.items()}
    for customer in proposal.destroy_customers:
        choices = list(proposal.candidate_routes[customer])
        if current[customer] not in choices:
            choices.append(current[customer])
        route_scores = []
        for route_idx, route in enumerate(solution.routes):
            distance = (
                min(instance.distance(customer, other) for other in route if other != customer)
                if any(other != customer for other in route)
                else instance.distance(customer, instance.depot.id)
            )
            route_scores.append((distance, route_idx))
        for _, route_idx in sorted(route_scores)[: max(0, heuristic_routes)]:
            if route_idx not in choices:
                choices.append(route_idx)
        remaining = [route for route in range(instance.vehicle_count) if route not in choices]
        rng.shuffle(remaining)
        choices.extend(remaining[: max(0, random_routes)])
        expanded[customer] = choices
        base = logits.setdefault(customer, {})
        for route in choices:
            base.setdefault(route, 0.0)
    result = NeighborhoodProposal(
        destroy_customers=list(proposal.destroy_customers),
        candidate_routes=expanded,
        confidence=proposal.confidence,
        source=f"{proposal.source}+safe",
        raw={**proposal.raw, "safe_expansion": True},
        candidate_route_logits=logits,
        candidate_edges=list(proposal.candidate_edges),
    )
    result.validate(instance)
    return result
