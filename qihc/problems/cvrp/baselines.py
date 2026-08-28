"""Classical CVRP baselines used in the formal comparison."""

from __future__ import annotations

import re
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

from qihc.problems.cvrp.instance import CVRPInstance, RouteSolution
from qihc.problems.cvrp.verifier import VerificationResult, verify_solution


@dataclass
class BaselineResult:
    method: str
    solution: RouteSolution
    verification: VerificationResult
    elapsed_s: float
    metadata: dict


def solve_ortools(instance: CVRPInstance, time_limit_s: int = 30, seed: int = 0) -> BaselineResult:
    """Solve capacity plus normalized same/different-vehicle constraints with OR-Tools."""

    try:
        from ortools.constraint_solver import pywrapcp, routing_enums_pb2
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("OR-Tools baseline requires the formal optional dependencies") from exc
    node_ids = [instance.depot.id, *instance.customer_ids]
    id_to_node = {customer: idx for idx, customer in enumerate(node_ids)}
    scale = 1000
    manager = pywrapcp.RoutingIndexManager(len(node_ids), instance.vehicle_count, 0)
    routing = pywrapcp.RoutingModel(manager)

    def distance_callback(from_index: int, to_index: int) -> int:
        left = node_ids[manager.IndexToNode(from_index)]
        right = node_ids[manager.IndexToNode(to_index)]
        return int(round(instance.distance(left, right) * scale))

    transit_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_index)

    def demand_callback(index: int) -> int:
        customer = node_ids[manager.IndexToNode(index)]
        return 0 if customer == instance.depot.id else instance.customers[customer].demand

    demand_index = routing.RegisterUnaryTransitCallback(demand_callback)
    routing.AddDimensionWithVehicleCapacity(
        demand_index,
        0,
        [instance.vehicle_capacity] * instance.vehicle_count,
        True,
        "Capacity",
    )
    routing.AddConstantDimension(1, len(node_ids) + 1, True, "Order")
    order_dimension = routing.GetDimensionOrDie("Order")
    solver = routing.solver()
    for spec in instance.constraints:
        if not spec.hard:
            continue
        if spec.type == "precedence":
            before = int(spec.params["before"])
            after = int(spec.params["after"])
            before_index = manager.NodeToIndex(id_to_node[before])
            after_index = manager.NodeToIndex(id_to_node[after])
            solver.Add(routing.VehicleVar(before_index) == routing.VehicleVar(after_index))
            solver.Add(order_dimension.CumulVar(before_index) < order_dimension.CumulVar(after_index))
            continue
        entities = [int(x) for x in spec.params.get("entities", spec.params.get("customers", []))]
        if len(entities) < 2:
            continue
        left = manager.NodeToIndex(id_to_node[entities[0]])
        right = manager.NodeToIndex(id_to_node[entities[1]])
        if spec.type in {"same_resource", "same_vehicle"}:
            solver.Add(routing.VehicleVar(left) == routing.VehicleVar(right))
        elif spec.type == "mutual_exclusion":
            solver.Add(routing.VehicleVar(left) != routing.VehicleVar(right))

    parameters = pywrapcp.DefaultRoutingSearchParameters()
    parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    parameters.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    parameters.time_limit.seconds = int(time_limit_s)
    parameters.log_search = False
    t0 = time.perf_counter()
    assignment = routing.SolveWithParameters(parameters)
    elapsed = time.perf_counter() - t0
    if assignment is None:
        raise RuntimeError(f"OR-Tools found no solution for {instance.name}")
    routes: list[list[int]] = []
    for vehicle in range(instance.vehicle_count):
        index = routing.Start(vehicle)
        route: list[int] = []
        while not routing.IsEnd(index):
            node = manager.IndexToNode(index)
            customer = node_ids[node]
            if customer != instance.depot.id:
                route.append(customer)
            index = assignment.Value(routing.NextVar(index))
        routes.append(route)
    solution = RouteSolution(routes, source="ortools_gls")
    return BaselineResult(
        method="ortools",
        solution=solution,
        verification=verify_solution(instance, solution),
        elapsed_s=elapsed,
        metadata={"time_limit_s": time_limit_s, "seed": seed, "scale": scale},
    )


def _parse_hgs_solution(path: Path) -> RouteSolution:
    routes: list[list[int]] = []
    cost = None
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        route_match = re.match(r"Route\s+#?\d+\s*:\s*(.*)", line, re.IGNORECASE)
        if route_match:
            routes.append([int(x) for x in route_match.group(1).split() if x.strip()])
        cost_match = re.match(r"Cost\s+([0-9.eE+-]+)", line, re.IGNORECASE)
        if cost_match:
            cost = float(cost_match.group(1))
    if not routes:
        raise ValueError(f"Could not parse HGS solution: {path}")
    return RouteSolution(routes, source="hgs_cvrp", metadata={"reported_cost": cost})


def solve_hgs(
    instance: CVRPInstance,
    binary: str | Path,
    time_limit_s: int = 30,
    seed: int = 0,
) -> BaselineResult:
    """Run the official HGS-CVRP executable for a CVRPLIB-backed instance."""

    source = Path(instance.metadata.get("source", ""))
    if not source.exists():
        raise ValueError("HGS baseline requires an instance loaded from a .vrp file")
    executable = Path(binary)
    if not executable.exists():
        raise FileNotFoundError(executable)
    with tempfile.TemporaryDirectory(prefix="qihc_hgs_") as tmp:
        solution_path = Path(tmp) / "solution.sol"
        command = [
            str(executable),
            str(source),
            str(solution_path),
            "-seed",
            str(seed),
            "-t",
            str(time_limit_s),
        ]
        t0 = time.perf_counter()
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
        elapsed = time.perf_counter() - t0
        if completed.returncode != 0 or not solution_path.exists():
            raise RuntimeError(
                f"HGS failed with code {completed.returncode}: {completed.stderr[-2000:]}"
            )
        solution = _parse_hgs_solution(solution_path)
    return BaselineResult(
        method="hgs",
        solution=solution,
        verification=verify_solution(instance, solution),
        elapsed_s=elapsed,
        metadata={
            "time_limit_s": time_limit_s,
            "seed": seed,
            "stdout_tail": completed.stdout[-2000:],
        },
    )
