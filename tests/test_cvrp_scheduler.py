from qihc.problems.cvrp import LNSConfig, QIHCLNSSolver, generate_synthetic_instance
import pytest


def test_qihc_lns_smoke_is_feasible_and_no_harm():
    instance = generate_synthetic_instance(
        n_customers=20,
        vehicle_count=5,
        vehicle_capacity=30,
        seed=11,
        semantic_constraints=False,
    )
    config = LNSConfig(
        iterations=4,
        destroy_size=5,
        routes_per_customer=3,
        sampler="numpy",
        sampling_steps=20,
        num_chains=12,
        top_samples=6,
        seed=11,
    )
    result = QIHCLNSSolver(config).solve(instance)
    assert result.verification.feasible
    assert result.verification.objective <= result.initial_verification.objective + 1e-9
    assert result.records
    assert all(record.qubo_variables > 0 for record in result.records)


def test_torch_pbit_backend_on_cpu():
    pytest.importorskip("torch")
    instance = generate_synthetic_instance(
        n_customers=12,
        vehicle_count=4,
        vehicle_capacity=25,
        seed=13,
        semantic_constraints=False,
    )
    config = LNSConfig(
        iterations=2,
        destroy_size=4,
        routes_per_customer=2,
        sampler="torch",
        device="cpu",
        sampling_steps=10,
        num_chains=8,
        top_samples=4,
        seed=13,
    )
    result = QIHCLNSSolver(config).solve(instance)
    assert result.verification.feasible
    assert result.verification.objective <= result.initial_verification.objective + 1e-9
