"""Construct a full CVRP solution with p-bit batches, without an incumbent."""

from __future__ import annotations

import copy
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


def _applicable(spec, assigned: set[int]) -> bool:
    params = spec.params
    if spec.type in {"same_resource", "same_vehicle", "mutual_exclusion"}:
        return all(int(x) in assigned for x in params.get("entities", params.get("customers", [])))
    if spec.type == "precedence":
        return int(params["before"]) in assigned and int(params["after"]) in assigned
    if spec.type in {"preferred_time", "time_window"}:
        return int(params.get("entity", params.get("customer", -1))) in assigned
    return True


def _validated_partial(instance: CVRPInstance, solution: RouteSolution) -> RouteSolution | None:
    assigned = {c for route in solution.routes for c in route}
    partial = copy.copy(instance)
    partial.customers = {c: instance.customers[c] for c in assigned}
    partial.constraints = [spec for spec in instance.constraints if _applicable(spec, assigned)]
    repaired = repair_supported_constraints(partial, solution)
    return repaired if verify_solution(partial, repaired).feasible else None


def construct_with_pbit(
    instance: CVRPInstance,
    sampler_factory,
    *,
    batch_size: int = 6,
    routes_per_customer: int = 4,
    seed: int = 0,
    selector=None,
) -> ColdStartResult:
    """Incrementally assign previously unserved customers using sampled QUBOs.

    The input contains *no* feasible routes. Every batch is sampled by p-bit;
    partial candidates are checked before commitment. This is a constructive
    first-solution stage, not a disguised call to the greedy incumbent builder.
    """
    if sum(c.demand for c in instance.customers.values()) > instance.vehicle_count * instance.vehicle_capacity:
        raise ValueError("Total demand exceeds fleet capacity")
    remaining = sorted(instance.customer_ids, key=lambda c: (-instance.customers[c].demand, c))
    current = RouteSolution([[] for _ in range(instance.vehicle_count)], source="pbit_cold_start")
    elapsed = 0.0
    batches = 0
    while remaining:
        take = min(batch_size, len(remaining))
        accepted = None
        while take >= 1 and accepted is None:
            batch = remaining[:take]
            domains: dict[int, list[int]] = {}
            for customer in batch:
                scored = []
                for route_idx, route in enumerate(current.routes):
                    load = sum(instance.customers[c].demand for c in route)
                    if load + instance.customers[customer].demand <= instance.vehicle_capacity:
                        _, delta = best_insertion(instance, route, customer)
                        scored.append((delta, route_idx))
                domains[customer] = [route for _, route in sorted(scored)[:routes_per_customer]]
                if not domains[customer]:
                    raise ValueError(f"No capacity-feasible route for customer {customer}")
            proposal = NeighborhoodProposal(batch, domains, source="pbit_cold_domain")
            if selector is not None and hasattr(selector, "propose_cold_batch"):
                proposal = selector.propose_cold_batch(instance, current, proposal, batches, seed)
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
                active = {c for route in current.routes for c in route} | set(batch)
                partial = copy.copy(instance)
                partial.constraints = [spec for spec in instance.constraints if _applicable(spec, active)]
                sampler_type = TorchPDitMFCSampler if hasattr(base_sampler, "device") else PDitMFCSampler
                kwargs = dict(num_chains=base_sampler.num_chains, steps=base_sampler.steps,
                              top_k=base_sampler.top_k, seed=seed + 7919 * batches)
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
                sampled = base_sampler.solve(weight, field, None)
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
                    assigned = {c for route in decoded.routes for c in route}
                    partial.customers = {c: instance.customers[c] for c in assigned}
                    partial.constraints = [spec for spec in instance.constraints if _applicable(spec, assigned)]
                    candidates.append((verify_solution(partial, decoded).objective, decoded))
            if candidates:
                accepted = min(candidates, key=lambda item: item[0])[1]
            else:
                take //= 2
        if accepted is None:
            raise ValueError("p-bit cold construction found no feasible extension")
        current = accepted
        remaining = remaining[take:]
        batches += 1
    current = repair_supported_constraints(instance, current)
    final = verify_solution(instance, current)
    if not final.feasible:
        raise ValueError(f"p-bit cold construction is infeasible: {final.violations}")
    current.source = "pbit_cold_start"
    current.metadata["construction_batches"] = batches
    return ColdStartResult(current, elapsed, batches)
