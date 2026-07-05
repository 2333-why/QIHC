# -*- coding: utf-8 -*-
"""Smoke tests for QIHC Ising samplers."""
import networkx as nx
import numpy as np
import pytest

from qihc import IsingModel
from qihc.ising import maxcut


@pytest.fixture
def small_maxcut_problem():
    np.random.seed(0)
    G = nx.erdos_renyi_graph(12, 0.4, seed=0)
    J = maxcut.max_cut_to_ising(G)
    return G, J


@pytest.mark.parametrize(
    "method_name",
    [
        "gibbs_sampling_Maxcut",
        "parallel_tempering_Maxcut",
        "simulated_quantum_annealing_Maxcut",
        "ising_simulated_annealing_Maxcut_Syn",
        "ising_simulated_annealing_Maxcut_Asyn",
    ],
)
def test_sampler_runs(small_maxcut_problem, method_name):
    G, J = small_maxcut_problem
    model = IsingModel(size=len(G.nodes()))
    method = getattr(model, method_name)
    kwargs = dict(J=J, steps=50, T_start=5.0, T_end=0.1, k=1.0)
    if method_name == "parallel_tempering_Maxcut":
        spins, trace, _ = method(**kwargs, n_replicas=4, swap_interval=5)
    elif method_name == "simulated_quantum_annealing_Maxcut":
        spins, trace, _ = method(**kwargs, Gamma_start=2.0, Gamma_end=0.05, m_slices=4)
    else:
        spins, trace, _ = method(**kwargs)

    assert isinstance(spins, dict)
    assert len(spins) == len(G.nodes())
    assert len(trace) > 1
    cut = maxcut.calculate_cut_value(G, maxcut.convert_spins_to_cut(spins))
    assert cut >= 0


def test_stochastic_lookup_table():
    from qihc.stochastic import load_lookup_table

    out = load_lookup_table(mode="AND", p1=0.2, p2=0.3, scale_input=1.0, bit_length=8)
    assert out is not None
    val = float(np.asarray(out).ravel()[0])
    assert 0.0 <= val <= 1.0
