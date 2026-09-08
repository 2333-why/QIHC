"""Restricted customer-to-route QUBO used by the p-bit repair stage."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Hashable

import numpy as np

from qihc.problems.cvrp.instance import CVRPInstance, RouteSolution
from qihc.problems.cvrp.neighborhood import NeighborhoodProposal
from qihc.problems.cvrp.verifier import best_insertion


Variable = tuple[Hashable, ...]


@dataclass
class QUBOModel:
    variables: list[Variable] = field(default_factory=list)
    index: dict[Variable, int] = field(default_factory=dict)
    linear: dict[int, float] = field(default_factory=dict)
    quadratic: dict[tuple[int, int], float] = field(default_factory=dict)
    constant: float = 0.0

    def add_variable(self, variable: Variable) -> int:
        if variable not in self.index:
            self.index[variable] = len(self.variables)
            self.variables.append(variable)
        return self.index[variable]

    def add_linear(self, variable: Variable, coefficient: float) -> None:
        idx = self.add_variable(variable)
        self.linear[idx] = self.linear.get(idx, 0.0) + float(coefficient)

    def add_quadratic(self, left: Variable, right: Variable, coefficient: float) -> None:
        i, j = self.add_variable(left), self.add_variable(right)
        if i == j:
            self.linear[i] = self.linear.get(i, 0.0) + float(coefficient)
            return
        key = (i, j) if i < j else (j, i)
        self.quadratic[key] = self.quadratic.get(key, 0.0) + float(coefficient)

    def add_squared(self, terms: dict[Variable, float], rhs: float, penalty: float) -> None:
        items = list(terms.items())
        self.constant += penalty * rhs * rhs
        for variable, coefficient in items:
            self.add_linear(variable, penalty * (coefficient * coefficient - 2.0 * rhs * coefficient))
        for pos, (left, a) in enumerate(items):
            for right, b in items[pos + 1 :]:
                self.add_quadratic(left, right, 2.0 * penalty * a * b)

    def to_ising(self) -> tuple[np.ndarray, np.ndarray, float]:
        """Convert upper-triangular binary QUBO to the repository Ising convention."""

        n = len(self.variables)
        weight = np.zeros((n, n), dtype=float)
        spin_linear = np.zeros(n, dtype=float)
        offset = float(self.constant)
        for idx, coefficient in self.linear.items():
            spin_linear[idx] += coefficient / 2.0
            offset += coefficient / 2.0
        for (i, j), coefficient in self.quadratic.items():
            weight[i, j] = weight[j, i] = -coefficient / 4.0
            spin_linear[i] += coefficient / 4.0
            spin_linear[j] += coefficient / 4.0
            offset += coefficient / 4.0
        field = -spin_linear
        return weight, field, offset

    def energy(self, bits: np.ndarray) -> float:
        bits = np.asarray(bits, dtype=float).ravel()
        energy = self.constant
        energy += sum(coefficient * bits[idx] for idx, coefficient in self.linear.items())
        energy += sum(coefficient * bits[i] * bits[j] for (i, j), coefficient in self.quadratic.items())
        return float(energy)


@dataclass
class AssignmentQUBO:
    model: QUBOModel
    proposal: NeighborhoodProposal
    fixed_routes: list[list[int]]
    insertion_costs: dict[tuple[int, int], float]
    initial_bits: np.ndarray

    @property
    def num_variables(self) -> int:
        return len(self.model.variables)


def _slack_weights(limit: int) -> list[int]:
    if limit <= 0:
        return []
    weights: list[int] = []
    remaining, value = limit, 1
    while remaining > 0:
        weight = min(value, remaining)
        weights.append(weight)
        remaining -= weight
        value *= 2
    return weights


def build_assignment_qubo(
    instance: CVRPInstance,
    solution: RouteSolution,
    proposal: NeighborhoodProposal,
    assignment_penalty: float = 40.0,
    capacity_penalty: float = 12.0,
    semantic_penalty: float = 20.0,
    proposal_bias: float = 1.0,
) -> AssignmentQUBO:
    proposal.validate(instance)
    destroyed = set(proposal.destroy_customers)
    fixed_routes = [[customer for customer in route if customer not in destroyed] for route in solution.routes]
    fixed_routes.extend([[] for _ in range(instance.vehicle_count - len(fixed_routes))])
    fixed_routes = fixed_routes[: instance.vehicle_count]
    model = QUBOModel()
    insertion_costs: dict[tuple[int, int], float] = {}

    for customer in proposal.destroy_customers:
        terms: dict[Variable, float] = {}
        for route_idx in proposal.candidate_routes[customer]:
            variable = ("assign", customer, route_idx)
            terms[variable] = 1.0
            _, delta = best_insertion(instance, fixed_routes[route_idx], customer)
            insertion_costs[(customer, route_idx)] = delta
            model.add_linear(variable, delta)
            # The LLM/p-bit prior guides search but never replaces the objective
            # or hard constraint penalties.
            logit = proposal.candidate_route_logits.get(customer, {}).get(route_idx, 0.0)
            model.add_linear(variable, -proposal_bias * float(logit))
        model.add_squared(terms, rhs=1.0, penalty=assignment_penalty)

    for route_idx in range(instance.vehicle_count):
        fixed_load = sum(instance.customers[c].demand for c in fixed_routes[route_idx])
        residual = instance.vehicle_capacity - fixed_load
        if residual < 0:
            raise ValueError("Current solution violates capacity before repair")
        terms: dict[Variable, float] = {}
        for customer in proposal.destroy_customers:
            if route_idx in proposal.candidate_routes[customer]:
                terms[("assign", customer, route_idx)] = float(instance.customers[customer].demand)
        for bit_idx, weight in enumerate(_slack_weights(residual)):
            terms[("slack", route_idx, bit_idx)] = float(weight)
        if terms:
            model.add_squared(terms, rhs=float(residual), penalty=capacity_penalty)

    current_location = {
        customer: route_idx for route_idx, route in enumerate(solution.routes) for customer in route
    }
    for spec in instance.constraints:
        entities = [int(x) for x in spec.params.get("entities", spec.params.get("customers", []))]
        if len(entities) < 2 or spec.type not in {"same_resource", "same_vehicle", "mutual_exclusion"}:
            continue
        left, right = entities[:2]
        for route_idx in range(instance.vehicle_count):
            left_var = ("assign", left, route_idx)
            right_var = ("assign", right, route_idx)
            left_active = left in destroyed and left_var in model.index
            right_active = right in destroyed and right_var in model.index
            coefficient = semantic_penalty * spec.weight
            if spec.type in {"same_resource", "same_vehicle"}:
                if left_active and right_active:
                    model.add_linear(left_var, coefficient)
                    model.add_linear(right_var, coefficient)
                    model.add_quadratic(left_var, right_var, -2.0 * coefficient)
                elif left_active and current_location.get(right) != route_idx:
                    model.add_linear(left_var, coefficient)
                elif right_active and current_location.get(left) != route_idx:
                    model.add_linear(right_var, coefficient)
            elif spec.type == "mutual_exclusion":
                if left_active and right_active:
                    model.add_quadratic(left_var, right_var, coefficient)
                elif left_active and current_location.get(right) == route_idx:
                    model.add_linear(left_var, coefficient)
                elif right_active and current_location.get(left) == route_idx:
                    model.add_linear(right_var, coefficient)

    initial_bits = np.zeros(len(model.variables), dtype=np.int8)
    for idx, variable in enumerate(model.variables):
        if variable[0] == "assign":
            _, customer, route_idx = variable
            initial_bits[idx] = int(current_location.get(int(customer)) == int(route_idx))
    for route_idx in range(instance.vehicle_count):
        selected_load = sum(
            instance.customers[c].demand
            for c in proposal.destroy_customers
            if current_location.get(c) == route_idx
        )
        fixed_load = sum(instance.customers[c].demand for c in fixed_routes[route_idx])
        unused = instance.vehicle_capacity - fixed_load - selected_load
        slack = _slack_weights(max(0, instance.vehicle_capacity - fixed_load))
        for idx in sorted(range(len(slack)), key=lambda item: slack[item], reverse=True):
            weight = slack[idx]
            variable = ("slack", route_idx, idx)
            if variable in model.index and unused >= weight:
                initial_bits[model.index[variable]] = 1
                unused -= weight
    return AssignmentQUBO(model, proposal, fixed_routes, insertion_costs, initial_bits)


def decode_assignment(instance: CVRPInstance, problem: AssignmentQUBO, bits: np.ndarray) -> RouteSolution:
    """Decode p-bit output with capacity-aware deterministic insertion and fallback choices."""

    bits = np.asarray(bits, dtype=np.int8).ravel()
    routes = [list(route) for route in problem.fixed_routes]
    loads = [sum(instance.customers[c].demand for c in route) for route in routes]
    desired: dict[int, list[int]] = {}
    for customer in problem.proposal.destroy_customers:
        active: list[int] = []
        for route_idx in problem.proposal.candidate_routes[customer]:
            variable = ("assign", customer, route_idx)
            idx = problem.model.index.get(variable)
            if idx is not None and idx < bits.size and bits[idx] > 0:
                active.append(route_idx)
        desired[customer] = active

    order = sorted(
        problem.proposal.destroy_customers,
        key=lambda customer: (-instance.customers[customer].demand, customer),
    )
    for customer in order:
        demand = instance.customers[customer].demand
        preferred = desired[customer]
        candidates = preferred + [
            route for route in problem.proposal.candidate_routes[customer] if route not in preferred
        ]
        feasible = [route for route in candidates if loads[route] + demand <= instance.vehicle_capacity]
        if not feasible:
            feasible = [route for route in range(instance.vehicle_count) if loads[route] + demand <= instance.vehicle_capacity]
        if not feasible:
            raise ValueError("Decoded assignment cannot be capacity repaired")
        scored: list[tuple[float, int, int]] = []
        for route_idx in feasible:
            position, delta = best_insertion(instance, routes[route_idx], customer)
            preference_penalty = 0.0 if route_idx in preferred else 1e-6
            scored.append((delta + preference_penalty, route_idx, position))
        _, route_idx, position = min(scored)
        routes[route_idx].insert(position, customer)
        loads[route_idx] += demand
    return RouteSolution(routes, source="pbit_assignment_repair")
