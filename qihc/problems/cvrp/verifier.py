"""Deterministic feasibility boundary for CVRP/NL-CVRP solutions."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable

from qihc.problems.cvrp.instance import CVRPInstance, RouteSolution


@dataclass
class VerificationResult:
    feasible: bool
    cost: float
    soft_penalty: float
    objective: float
    violations: list[dict] = field(default_factory=list)
    route_loads: list[int] = field(default_factory=list)
    arrival_times: dict[int, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "feasible": self.feasible,
            "cost": self.cost,
            "soft_penalty": self.soft_penalty,
            "objective": self.objective,
            "violations": self.violations,
            "route_loads": self.route_loads,
            "arrival_times": self.arrival_times,
        }


def route_cost(instance: CVRPInstance, route: Iterable[int]) -> float:
    nodes = [instance.depot.id, *list(route), instance.depot.id]
    return float(sum(instance.distance(a, b) for a, b in zip(nodes, nodes[1:])))


def _locations(routes: list[list[int]]) -> dict[int, tuple[int, int]]:
    return {customer: (route_idx, pos) for route_idx, route in enumerate(routes) for pos, customer in enumerate(route)}


def verify_solution(instance: CVRPInstance, solution: RouteSolution | list[list[int]]) -> VerificationResult:
    routes = solution.routes if isinstance(solution, RouteSolution) else solution
    violations: list[dict] = []
    seen = [customer for route in routes for customer in route]
    expected = set(instance.customer_ids)
    unknown = sorted(set(seen) - expected)
    missing = sorted(expected - set(seen))
    duplicates = sorted({customer for customer in seen if seen.count(customer) > 1})
    if unknown:
        violations.append({"type": "unknown_customer", "entities": unknown})
    if missing:
        violations.append({"type": "missing_customer", "entities": missing})
    if duplicates:
        violations.append({"type": "duplicate_customer", "entities": duplicates})
    if len(routes) > instance.vehicle_count:
        violations.append({"type": "vehicle_count", "actual": len(routes), "limit": instance.vehicle_count})

    loads: list[int] = []
    arrivals: dict[int, float] = {}
    total_cost = 0.0
    for route_idx, route in enumerate(routes):
        valid_route = [customer for customer in route if customer in instance.customers]
        load = sum(instance.customers[customer].demand for customer in valid_route)
        loads.append(load)
        if load > instance.vehicle_capacity:
            violations.append(
                {"type": "capacity", "route": route_idx, "actual": load, "limit": instance.vehicle_capacity}
            )
        total_cost += route_cost(instance, valid_route)
        clock = 0.0
        previous = instance.depot.id
        for customer_id in valid_route:
            customer = instance.customers[customer_id]
            clock += instance.distance(previous, customer_id)
            clock = max(clock, customer.ready_time)
            arrivals[customer_id] = clock
            if clock > customer.due_time:
                violations.append(
                    {"type": "time_window", "customer": customer_id, "arrival": clock, "due": customer.due_time}
                )
            clock += customer.service_time
            previous = customer_id

    locations = _locations(routes)
    soft_penalty = 0.0
    for spec in instance.constraints:
        kind, params = spec.type, spec.params
        local_violation: dict | None = None
        if kind in {"same_resource", "same_vehicle"}:
            entities = [int(x) for x in params.get("entities", params.get("customers", []))]
            route_ids = {locations[x][0] for x in entities if x in locations}
            if len(route_ids) > 1 or len(route_ids) != (1 if entities else 0):
                local_violation = {"type": kind, "entities": entities}
        elif kind == "mutual_exclusion":
            entities = [int(x) for x in params.get("entities", params.get("customers", []))]
            if len(entities) >= 2 and all(x in locations for x in entities):
                if len({locations[x][0] for x in entities}) == 1:
                    local_violation = {"type": kind, "entities": entities}
        elif kind == "precedence":
            before, after = int(params["before"]), int(params["after"])
            if before not in locations or after not in locations:
                local_violation = {"type": kind, "before": before, "after": after, "reason": "missing"}
            elif locations[before][0] != locations[after][0] or locations[before][1] >= locations[after][1]:
                local_violation = {"type": kind, "before": before, "after": after}
        elif kind == "max_route_distance":
            limit = float(params["limit"])
            for route_idx, route in enumerate(routes):
                distance = route_cost(instance, route)
                if distance > limit:
                    violation = {"type": kind, "route": route_idx, "actual": distance, "limit": limit}
                    if spec.hard:
                        violations.append(violation)
                    else:
                        soft_penalty += spec.weight * (distance - limit)
        elif kind == "preferred_time":
            entity = int(params.get("entity", params.get("customer")))
            interval = params.get("interval", [0.0, math.inf])
            arrival = arrivals.get(entity, math.inf)
            if not (float(interval[0]) <= arrival <= float(interval[1])):
                local_violation = {"type": kind, "entity": entity, "arrival": arrival, "interval": interval}

        if local_violation:
            if spec.hard:
                violations.append(local_violation)
            else:
                soft_penalty += float(spec.weight)

    return VerificationResult(
        feasible=not violations,
        cost=total_cost,
        soft_penalty=soft_penalty,
        objective=total_cost + soft_penalty,
        violations=violations,
        route_loads=loads,
        arrival_times=arrivals,
    )


def best_insertion(instance: CVRPInstance, route: list[int], customer_id: int) -> tuple[int, float]:
    best_position, best_delta = 0, math.inf
    for position in range(len(route) + 1):
        left = instance.depot.id if position == 0 else route[position - 1]
        right = instance.depot.id if position == len(route) else route[position]
        delta = instance.distance(left, customer_id) + instance.distance(customer_id, right) - instance.distance(left, right)
        if delta < best_delta:
            best_position, best_delta = position, delta
    return best_position, float(best_delta)


def greedy_initial_solution(instance: CVRPInstance) -> RouteSolution:
    """Deterministic cheapest-feasible insertion used as a safe incumbent."""

    routes: list[list[int]] = [[] for _ in range(instance.vehicle_count)]
    loads = [0] * instance.vehicle_count
    customers = sorted(instance.customer_ids, key=lambda i: (-instance.customers[i].demand, i))
    for customer_id in customers:
        demand = instance.customers[customer_id].demand
        choices: list[tuple[float, int, int]] = []
        for route_idx, route in enumerate(routes):
            if loads[route_idx] + demand <= instance.vehicle_capacity:
                position, delta = best_insertion(instance, route, customer_id)
                choices.append((delta, route_idx, position))
        if not choices:
            raise ValueError(f"Instance {instance.name} is infeasible under the declared fleet capacity")
        _, route_idx, position = min(choices)
        routes[route_idx].insert(position, customer_id)
        loads[route_idx] += demand
    solution = RouteSolution(routes, source="greedy_cheapest_insertion")
    return repair_supported_constraints(instance, solution)


def _move_customer(
    instance: CVRPInstance,
    routes: list[list[int]],
    customer: int,
    destination: int,
) -> bool:
    source = next((idx for idx, route in enumerate(routes) if customer in route), None)
    if source is None or source == destination:
        return source is not None
    demand = instance.customers[customer].demand
    destination_load = sum(instance.customers[c].demand for c in routes[destination])
    if destination_load + demand > instance.vehicle_capacity:
        return False
    routes[source].remove(customer)
    position, _ = best_insertion(instance, routes[destination], customer)
    routes[destination].insert(position, customer)
    return True


def repair_supported_constraints(instance: CVRPInstance, solution: RouteSolution) -> RouteSolution:
    """Repair the normalized constraint subset currently supported by QIHC-LNS."""

    routes = [list(route) for route in solution.routes]
    for _ in range(3):
        locations = _locations(routes)
        changed = False
        for spec in instance.constraints:
            params = spec.params
            if not spec.hard:
                continue
            if spec.type in {"same_resource", "same_vehicle"}:
                entities = [int(x) for x in params.get("entities", params.get("customers", []))]
                if len(entities) >= 2 and all(x in locations for x in entities):
                    left, right = entities[:2]
                    if locations[left][0] != locations[right][0]:
                        moved = _move_customer(instance, routes, right, locations[left][0])
                        if not moved:
                            moved = _move_customer(instance, routes, left, locations[right][0])
                        changed |= moved
            elif spec.type == "mutual_exclusion":
                entities = [int(x) for x in params.get("entities", params.get("customers", []))]
                if len(entities) >= 2 and all(x in locations for x in entities):
                    left, right = entities[:2]
                    if locations[left][0] == locations[right][0]:
                        source = locations[right][0]
                        alternatives = [
                            idx
                            for idx in range(instance.vehicle_count)
                            if idx != source
                            and sum(instance.customers[c].demand for c in routes[idx])
                            + instance.customers[right].demand
                            <= instance.vehicle_capacity
                        ]
                        if alternatives:
                            destination = min(
                                alternatives,
                                key=lambda idx: best_insertion(instance, routes[idx], right)[1],
                            )
                            changed |= _move_customer(instance, routes, right, destination)
            elif spec.type == "precedence":
                before, after = int(params["before"]), int(params["after"])
                if before in locations and after in locations:
                    before_route, before_pos = locations[before]
                    after_route, after_pos = locations[after]
                    if before_route != after_route:
                        moved = _move_customer(instance, routes, after, before_route)
                        if not moved:
                            moved = _move_customer(instance, routes, before, after_route)
                        if not moved:
                            pair_demand = (
                                instance.customers[before].demand + instance.customers[after].demand
                            )
                            alternatives = [
                                idx
                                for idx, route in enumerate(routes)
                                if sum(instance.customers[c].demand for c in route) + pair_demand
                                <= instance.vehicle_capacity
                            ]
                            if alternatives:
                                destination = min(alternatives, key=lambda idx: len(routes[idx]))
                                moved_before = _move_customer(instance, routes, before, destination)
                                moved_after = _move_customer(instance, routes, after, destination)
                                moved = moved_before and moved_after
                        changed |= moved
                    locations = _locations(routes)
                    if before in locations and after in locations and locations[before][0] == locations[after][0]:
                        route_idx = locations[before][0]
                        route = routes[route_idx]
                        if route.index(before) > route.index(after):
                            route.remove(before)
                            route.insert(route.index(after), before)
                            changed = True
        if not changed:
            break
    return RouteSolution(routes, source=solution.source + "+constraint_repair", metadata=dict(solution.metadata))
