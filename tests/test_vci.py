"""Tests for VCI co-inference loop."""

import numpy as np

from qihc.orchestrator import VCIConfig, VCIOrchestrator, compute_free_energy, demo_problem
from qihc.orchestrator.reasoning import count_exclusion_violations, generate_toy_problems, subset_to_ising
from qihc.orchestrator.types import RoutingContext
from qihc.orchestrator.frontend.mock import MockFrontend


def test_subset_ising_symmetric():
    logits = np.array([1.0, 0.5, 2.0, -0.2, 1.5, 2.2])
    w, h = subset_to_ising(logits, top_k=3, exclusion_pairs=[(2, 5)])
    assert w.shape == (6, 6)
    assert np.allclose(w, w.T)


def test_exclusion_violation_count():
    mask = np.array([0, 1, 1, 0, 0, 1], dtype=bool)
    n, pairs = count_exclusion_violations(mask, [(2, 5)])
    assert n == 1
    assert pairs == [(2, 5)]


def test_free_energy_increases_with_violation():
    logits = np.array([2.0, 1.0, 2.5, 0.5, 1.0, 2.4])
    q0 = logits.copy()
    good = np.array([1, 1, 1, 0, 0, 0], dtype=bool)
    bad = np.array([0, 1, 1, 0, 0, 1], dtype=bool)
    f_good = compute_free_energy(logits, q0, good, exclusion_pairs=[(2, 5)])
    f_bad = compute_free_energy(logits, q0, bad, exclusion_pairs=[(2, 5)])
    assert f_bad.total > f_good.total
    assert f_bad.violations == 1


def test_frontend_refine_downweights_violators():
    cfg = VCIConfig.tier_a()
    front = MockFrontend(cfg)
    logits = np.array([2.0, 1.0, 2.5, 0.5, 1.0, 2.4])
    ctx = RoutingContext(text="t", logits=logits)
    mask = np.array([0, 1, 1, 0, 0, 1], dtype=bool)
    fb = {"violation_pairs": [(2, 5)], "refine_penalty": 2.0}
    refined = front.refine(ctx, mask, fb, logits)
    assert refined.logits[2] < logits[2]
    assert refined.logits[5] < logits[5]


def test_vci2_improves_demo_feasibility():
    cfg = VCIConfig.tier_a(sampling_steps=180, seed=2)
    orch = VCIOrchestrator(cfg)
    problem = demo_problem()
    vci1 = orch.solve_subset(problem, mode="vci-1")
    vci2 = orch.solve_subset(problem, mode="vci-2")
    assert vci2.n_rounds >= 1
    if not vci1.final_feasible:
        assert vci2.final_feasible or vci2.final_free_energy <= vci1.final_free_energy


def test_trace_refine_demo_two_rounds():
    cfg = VCIConfig.tier_a(sampling_steps=150, seed=1)
    orch = VCIOrchestrator(cfg)
    trace = orch.trace_refine_demo(demo_problem())
    assert len(trace) == 2
    assert trace[0].feasible is False
    assert trace[0].free_energy.violations >= 1
    assert trace[1].feasible is True


def test_vci_batch_feasible_rate():
    cfg = VCIConfig.tier_a(sampling_steps=150, seed=3)
    orch = VCIOrchestrator(cfg)
    problems = generate_toy_problems(n_problems=12, seed=3)
    summary = orch.compare_modes(problems, modes=["greedy", "vci-2"])
    assert summary["vci-2"]["feasible_rate"] >= summary["greedy"]["feasible_rate"]
