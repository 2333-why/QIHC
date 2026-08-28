"""Unit tests for the offline QIHC-LNS CVRP implementation."""

import itertools

import numpy as np

from qihc.problems.cvrp import (
    KNNNeighborhoodSelector,
    build_assignment_qubo,
    generate_synthetic_instance,
    greedy_initial_solution,
    verify_solution,
)
from qihc.problems.cvrp.instance import load_cvrplib
from qihc.problems.cvrp.qubo import QUBOModel, decode_assignment


def test_synthetic_initial_solution_is_feasible():
    instance = generate_synthetic_instance(n_customers=24, vehicle_count=6, vehicle_capacity=35, seed=4)
    solution = greedy_initial_solution(instance)
    result = verify_solution(instance, solution)
    assert result.feasible, result.violations
    assert set(customer for route in solution.routes for customer in route) == set(instance.customer_ids)


def test_qubo_to_ising_preserves_energies():
    model = QUBOModel()
    model.add_linear(("x", 0), -2.0)
    model.add_linear(("x", 1), 0.75)
    model.add_quadratic(("x", 0), ("x", 1), 3.5)
    weight, field, offset = model.to_ising()
    for bits_tuple in itertools.product([0, 1], repeat=2):
        bits = np.asarray(bits_tuple)
        spins = 2 * bits - 1
        ising = -0.5 * spins @ weight @ spins - field @ spins + offset
        assert np.isclose(model.energy(bits), ising)


def test_assignment_qubo_initial_state_and_decode():
    instance = generate_synthetic_instance(
        n_customers=18,
        vehicle_count=5,
        vehicle_capacity=30,
        seed=7,
        semantic_constraints=False,
    )
    incumbent = greedy_initial_solution(instance)
    proposal = KNNNeighborhoodSelector(destroy_size=5, routes_per_customer=3).propose(
        instance, incumbent, iteration=0, seed=7
    )
    problem = build_assignment_qubo(instance, incumbent, proposal)
    assert 0 < problem.num_variables < 256
    decoded = decode_assignment(instance, problem, problem.initial_bits)
    result = verify_solution(instance, decoded)
    assert result.feasible, result.violations


def test_cvrplib_loader_rounds_euc2d_and_reads_bks(tmp_path):
    path = tmp_path / "X-n3-k1.vrp"
    path.write_text(
        """NAME : X-n3-k1
COMMENT : Optimal value: 10
TYPE : CVRP
DIMENSION : 3
EDGE_WEIGHT_TYPE : EUC_2D
CAPACITY : 10
NODE_COORD_SECTION
1 0 0
2 1 1
3 3 0
DEMAND_SECTION
1 0
2 2
3 2
DEPOT_SECTION
1
-1
EOF
""",
        encoding="utf-8",
    )
    instance = load_cvrplib(path)
    assert instance.best_known_cost == 10.0
    assert instance.distance(1, 2) == 1.0
    assert instance.vehicle_count == 1
