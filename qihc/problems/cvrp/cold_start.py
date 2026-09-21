"""Construct a full CVRP solution with p-bit batches, without an incumbent."""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass

import numpy as np

from qihc.problems.cvrp.instance import CVRPInstance, RouteSolution
from qihc.problems.cvrp.neighborhood import NeighborhoodProposal
from qihc.problems.cvrp.qubo import build_assignment_qubo, decode_assignment
from qihc.problems.cvrp.verifier import best_insertion, repair_supported_constraints, verify_solution


@dataclass
class ColdStartResult:
    solution: RouteSolution
    pbit_elapsed_s: float
    batches: int


def _constraint_entities(spec) -> list[int]:
    if spec.type == "precedence":
        return [int(spec.params["before"]), int(spec.params["after"])]
    return [
        int(value)
        for value in spec.params.get(
            "entities",
            spec.params.get("customer_ids", spec.params.get("customers", [])),
        )
    ]


def _applicable(spec, assigned: set[int]) -> bool:
    entities = _constraint_entities(spec)
    if spec.type in {"same_resource", "same_vehicle", "mutual_exclusion", "precedence"}:
        return bool(entities) and all(entity in assigned for entity in entities)
    if spec.type in {"preferred_time", "time_window"}:
        entity = int(spec.params.get("entity", spec.params.get("customer", -1)))
        return entity in assigned
    return True


def _validated_partial(instance: CVRPInstance, solution: RouteSolution) -> RouteSolution | None:
    assigned = {customer for route in solution.routes for customer in route}
    partial = copy.copy(instance)
    partial.customers = {customer: instance.customers[customer] for customer in assigned}
    partial.constraints = [spec for spec in instance.constraints if _applicable(spec, assigned)]
    repaired = repair_supported_constraints(partial, solution)
    return repaired if verify_solution(partial, repaired).feasible else None


def _co_route_groups(instance: CVRPInstance) -> list[list[int]]:
    """Return atomic groups that can never be split between construction batches."""
    parent = {customer: customer for customer in instance.customer_ids}

    def find(customer: int) -> int:
        while parent[customer] != customer:
            parent[customer] = parent[parent[customer]]
            customer = parent[customer]
        return customer

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[max(left_root, right_root)] = min(left_root, right_root)

    for spec in instance.constraints:
        if not spec.hard or spec.type not in {"same_resource", "same_vehicle", "precedence"}:
            continue
        entities = [entity for entity in _constraint_entities(spec) if entity in parent]
        for entity in entities[1:]:
            union(entities[0], entity)

    grouped: dict[int, list[int]] = {}
    for customer in instance.customer_ids:
        grouped.setdefault(find(customer), []).append(customer)
    groups = [sorted(group) for group in grouped.values()]
    for group in groups:
        demand = sum(instance.customers[customer].demand for customer in group)
        if demand > instance.vehicle_capacity:
            raise ValueError(
                "Hard co-route component exceeds vehicle capacity: "
                f"customers={group}, demand={demand}, capacity={instance.vehicle_capacity}"
            )
    return groups


def _ordered_groups(instance: CVRPInstance, seed: int, attempt: int) -> list[list[int]]:
    groups = _co_route_groups(instance)
    rng = random.Random(seed + 104729 * attempt)
    tie_break = {tuple(group): rng.random() for group in groups}
    return sorted(
        groups,
        key=lambda group: (
            -sum(instance.customers[customer].demand for customer in group),
            tie_break[tuple(group)] if attempt else min(group),
        ),
    )


def _next_group_batch(groups: list[list[int]], batch_size: int) -> list[list[int]]:
    selected: list[list[int]] = []
    customers = 0
    for group in groups:
        if selected and customers + len(group) > batch_size:
            break
        selected.append(group)
        customers += len(group)
    return selected


def _group_domains(
    instance: CVRPInstance,
    current: RouteSolution,
    groups: list[list[int]],
    route_limit: int,
) -> dict[int, list[int]]:
    loads = [sum(instance.customers[customer].demand for customer in route) for route in current.routes]
    domains: dict[int, list[int]] = {}
    for group in groups:
        group_demand = sum(instance.customers[customer].demand for customer in group)
        scored: list[tuple[float, int]] = []
        for route_idx, route in enumerate(current.routes):
            if loads[route_idx] + group_demand > instance.vehicle_capacity:
                continue
            scratch = list(route)
            delta_sum = 0.0
            for customer in group:
                position, delta = best_insertion(instance, scratch, customer)
                scratch.insert(position, customer)
                delta_sum += delta
            scored.append((delta_sum, route_idx))
        allowed = [route for _, route in sorted(scored)[:route_limit]]
        if not allowed:
            raise ValueError(
                f"No capacity-feasible route for co-route group {group} "
                f"with demand {group_demand}"
            )
        for customer in group:
            domains[customer] = list(allowed)
    return domains


def _sanitize_proposal(
    domain: NeighborhoodProposal,
    proposed: NeighborhoodProposal,
    groups: list[list[int]],
) -> NeighborhoodProposal:
    """Keep LLM ordering/logits but restore safe, common domains for coupled customers."""
    safe: dict[int, list[int]] = {}
    logits: dict[int, dict[int, float]] = {}
    for group in groups:
        common = set(domain.candidate_routes[group[0]])
        for customer in group[1:]:
            common &= set(domain.candidate_routes[customer])
        ordered: list[int] = []
        for customer in group:
            for route in proposed.candidate_routes.get(customer, []):
                if route in common and route not in ordered:
                    ordered.append(route)
        ordered.extend(
            route
            for route in domain.candidate_routes[group[0]]
            if route in common and route not in ordered
        )
        if not ordered:
            ordered = list(domain.candidate_routes[group[0]])
        for customer in group:
            safe[customer] = list(ordered)
            logits[customer] = {
                route: float(proposed.candidate_route_logits.get(customer, {}).get(route, 0.0))
                for route in ordered
            }
    sanitized = NeighborhoodProposal(
        list(domain.destroy_customers),
        safe,
        source=proposed.source,
        confidence=proposed.confidence,
        raw=proposed.raw,
        candidate_route_logits=logits,
    )
    return sanitized


def _mutual_exclusions(instance: CVRPInstance) -> set[frozenset[int]]:
    pairs: set[frozenset[int]] = set()
    for spec in instance.constraints:
        if not spec.hard or spec.type != "mutual_exclusion":
            continue
        entities = _constraint_entities(spec)
        if len(entities) >= 2:
            pairs.add(frozenset(entities[:2]))
    return pairs


def _guarded_extension(
    instance: CVRPInstance,
    current: RouteSolution,
    groups: list[list[int]],
    domains: dict[int, list[int]],
    seed: int,
) -> RouteSolution | None:
    """Find one batch-feasible extension used only when p-bit samples all fail."""
    loads = [sum(instance.customers[customer].demand for customer in route) for route in current.routes]
    locations = {customer: route for route, values in enumerate(current.routes) for customer in values}
    exclusions = _mutual_exclusions(instance)
    group_of = {customer: index for index, group in enumerate(groups) for customer in group}
    assigned_groups: dict[int, int] = {}
    rng = random.Random(seed)
    ordered = sorted(
        range(len(groups)),
        key=lambda index: (
            len(domains[groups[index][0]]),
            -sum(instance.customers[customer].demand for customer in groups[index]),
            min(groups[index]),
        ),
    )

    def conflicts(group_index: int, route: int) -> bool:
        for customer in groups[group_index]:
            for pair in exclusions:
                if customer not in pair:
                    continue
                other = next(iter(pair - {customer}))
                if locations.get(other) == route:
                    return True
                other_group = group_of.get(other)
                if other_group is not None and assigned_groups.get(other_group) == route:
                    return True
        return False

    def search(position: int) -> bool:
        if position == len(ordered):
            return True
        group_index = ordered[position]
        group = groups[group_index]
        demand = sum(instance.customers[customer].demand for customer in group)
        routes = list(domains[group[0]])
        rng.shuffle(routes)
        routes.sort(key=lambda route: instance.vehicle_capacity - loads[route] - demand)
        for route in routes:
            if loads[route] + demand > instance.vehicle_capacity or conflicts(group_index, route):
                continue
            assigned_groups[group_index] = route
            loads[route] += demand
            if search(position + 1):
                return True
            loads[route] -= demand
            assigned_groups.pop(group_index, None)
        return False

    if not search(0):
        return None
    routes = [list(route) for route in current.routes]
    for group_index, route_idx in assigned_groups.items():
        for customer in groups[group_index]:
            position, _ = best_insertion(instance, routes[route_idx], customer)
            routes[route_idx].insert(position, customer)
    return _validated_partial(instance, RouteSolution(routes, source="pbit_batch_feasibility_guard"))


def _construct_attempt(
    instance: CVRPInstance,
    sampler_factory,
    *,
    groups: list[list[int]],
    batch_size: int,
    route_limit: int,
    seed: int,
    selector,
) -> tuple[RouteSolution, float, int, int]:
    remaining = [list(group) for group in groups]
    current = RouteSolution([[] for _ in range(instance.vehicle_count)], source="pbit_cold_start")
    elapsed = 0.0
    batches = 0
    guarded_batches = 0
    while remaining:
        batch_groups = _next_group_batch(remaining, batch_size)
        accepted = None
        while batch_groups and accepted is None:
            batch = [customer for group in batch_groups for customer in group]
            domains = _group_domains(instance, current, batch_groups, route_limit)
            proposal = NeighborhoodProposal(batch, domains, source="pbit_cold_domain")
            if selector is not None and hasattr(selector, "propose_cold_batch"):
                proposed = selector.propose_cold_batch(instance, current, proposal, batches, seed)
                proposal = _sanitize_proposal(proposal, proposed, batch_groups)
            proposal.validate(instance)
            problem = build_assignment_qubo(instance, current, proposal)
            base_sampler = sampler_factory(seed + 7919 * batches)
            plan = instance.metadata.get("constraint_compilation") or {}
            use_hybrid = any(
                item.get("representation") in {"pdit", "mfc"}
                and item.get("execution") != "verifier_only"
                for item in plan.get("constraints", [])
            )
            if use_hybrid:
                from qihc.s2e.hybrid_sampler import PDitMFCSampler, TorchPDitMFCSampler

                active = {customer for route in current.routes for customer in route} | set(batch)
                partial = copy.copy(instance)
                partial.constraints = [spec for spec in instance.constraints if _applicable(spec, active)]
                sampler_type = TorchPDitMFCSampler if hasattr(base_sampler, "device") else PDitMFCSampler
                kwargs = dict(
                    num_chains=base_sampler.num_chains,
                    steps=base_sampler.steps,
                    top_k=base_sampler.top_k,
                    seed=seed + 7919 * batches,
                )
                if sampler_type is TorchPDitMFCSampler:
                    kwargs["device"] = base_sampler.device
                sampled = sampler_type(**kwargs).solve(partial, current, proposal)
                sampled_bits = np.zeros((len(sampled.assignments), problem.num_variables), dtype=np.int8)
                for sample_idx, assignment in enumerate(sampled.assignments):
                    for customer, route in assignment.items():
                        variable_idx = problem.model.index.get(("assign", customer, route))
                        if variable_idx is not None:
                            sampled_bits[sample_idx, variable_idx] = 1
            else:
                weight, field, _ = problem.model.to_ising()
                sampled = base_sampler.solve(weight, field, problem.initial_bits)
                sampled_bits = sampled.bits
            elapsed += sampled.elapsed_s
            candidates = []
            for bits in sampled_bits:
                try:
                    decoded = decode_assignment(instance, problem, bits)
                except ValueError:
                    continue
                decoded = _validated_partial(instance, decoded)
                if decoded is not None:
                    partial = copy.copy(instance)
                    assigned = {customer for route in decoded.routes for customer in route}
                    partial.customers = {customer: instance.customers[customer] for customer in assigned}
                    partial.constraints = [spec for spec in instance.constraints if _applicable(spec, assigned)]
                    candidates.append((verify_solution(partial, decoded).objective, decoded))
            if candidates:
                accepted = min(candidates, key=lambda item: item[0])[1]
            else:
                accepted = _guarded_extension(
                    instance,
                    current,
                    batch_groups,
                    proposal.candidate_routes,
                    seed + 65537 * batches,
                )
                if accepted is not None:
                    guarded_batches += 1
            if accepted is None:
                batch_groups = batch_groups[: len(batch_groups) // 2]
        if accepted is None:
            raise ValueError(
                "no feasible p-bit or guarded extension for next co-route group; "
                f"remaining_groups={len(remaining)}, next_group={remaining[0]}"
            )
        current = accepted
        remaining = remaining[len(batch_groups) :]
        batches += 1
    current = repair_supported_constraints(instance, current)
    final = verify_solution(instance, current)
    if not final.feasible:
        raise ValueError(f"p-bit cold construction is infeasible: {final.violations}")
    current.source = "pbit_cold_start"
    current.metadata["construction_batches"] = batches
    current.metadata["guarded_batches"] = guarded_batches
    return current, elapsed, batches, guarded_batches


def construct_with_pbit(
    instance: CVRPInstance,
    sampler_factory,
    *,
    batch_size: int = 6,
    routes_per_customer: int = 4,
    seed: int = 0,
    selector=None,
) -> ColdStartResult:
    """Incrementally construct a solution with p-bit samples and bounded recovery.

    Hard co-route/precedence components are sampled atomically. A small exact
    batch guard is considered only when every returned p-bit sample is invalid;
    it is not a supplied incumbent or a full-problem classical solve. If a
    partial packing still reaches a dead end, construction restarts from empty
    routes with a different seed and a wider candidate domain.
    """
    if batch_size <= 0 or routes_per_customer <= 0:
        raise ValueError("cold-start batch size and route count must be positive")
    if sum(customer.demand for customer in instance.customers.values()) > (
        instance.vehicle_count * instance.vehicle_capacity
    ):
        raise ValueError("Total demand exceeds fleet capacity")

    errors: list[str] = []
    total_elapsed = 0.0
    for attempt in range(4):
        groups = _ordered_groups(instance, seed, attempt)
        route_limit = (
            instance.vehicle_count
            if attempt == 3
            else min(instance.vehicle_count, routes_per_customer * (2 ** attempt))
        )
        attempt_seed = seed + 1000003 * attempt
        try:
            solution, elapsed, batches, guarded = _construct_attempt(
                instance,
                sampler_factory,
                groups=groups,
                batch_size=batch_size,
                route_limit=route_limit,
                seed=attempt_seed,
                selector=selector,
            )
            total_elapsed += elapsed
            solution.metadata["construction_attempts"] = attempt + 1
            solution.metadata["cold_route_limit"] = route_limit
            solution.metadata["guarded_batches"] = guarded
            return ColdStartResult(solution, total_elapsed, batches)
        except ValueError as exc:
            errors.append(f"attempt={attempt + 1}, route_limit={route_limit}: {exc}")
    raise ValueError(
        "p-bit cold construction exhausted four restart attempts: " + " | ".join(errors)
    )
