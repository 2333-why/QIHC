"""Tests for QIHC orchestrator (mock frontend, no torch required)."""

import numpy as np

from qihc.orchestrator import QIHCConfig, QIHCOrchestrator
from qihc.orchestrator.encoder import (
    batch_moe_logits_to_ising,
    capacity_aware_routing_score,
    greedy_top_k,
    moe_logits_to_ising,
    routing_score,
    sequential_capacity_decode,
)


def test_moe_ising_encoding_shape():
    logits = np.array([1.0, 0.5, -0.2, 0.3, 0.0, -1.0, 0.8, 0.1])
    w, h = moe_logits_to_ising(logits, top_k=2, penalty=2.0)
    assert w.shape == (8, 8)
    assert h.shape == (8,)
    assert np.allclose(w, w.T)


def test_batch_ising_encoding_shape():
    logits_batch = np.random.default_rng(0).normal(size=(4, 8))
    w, h = batch_moe_logits_to_ising(logits_batch, top_k=2, expert_capacity=2)
    assert w.shape == (32, 32)
    assert h.shape == (32,)
    assert np.allclose(w, w.T)


def test_capacity_aware_score_prefers_balanced():
    logits = np.array([3.0, 2.5, 2.0, 1.0, 0.5, 0.0, -0.5, -1.0])
    logits_batch = np.tile(logits, (8, 1))
    greedy_masks = [greedy_top_k(logits, 2) for _ in range(8)]
    balanced_masks = []
    for r in range(8):
        m = np.zeros(8, dtype=bool)
        m[r % 8] = True
        m[(r + 1) % 8] = True
        balanced_masks.append(m)
    greedy_score = capacity_aware_routing_score(logits_batch, greedy_masks, 2, 5.0)
    balanced_score = capacity_aware_routing_score(logits_batch, balanced_masks, 2, 5.0)
    assert balanced_score > greedy_score


def test_sequential_capacity_decode_is_feasible():
    logits_batch = np.random.default_rng(1).normal(size=(8, 8))
    masks = sequential_capacity_decode(logits_batch, top_k=2, expert_capacity=2)
    usage = np.sum(np.stack(masks), axis=0)
    assert np.all(usage <= 2)
    assert all(mask.sum() <= 2 for mask in masks)


def test_orchestrator_mock_batch():
    cfg = QIHCConfig.tier_a(sampling_steps=80, num_experts=8, top_k=2, seed=1)
    orch = QIHCOrchestrator(cfg)
    texts = ["route experts for token A", "route experts for token B", "another prompt"]
    result = orch.run_batch(texts)

    assert len(result.greedy) == 3
    assert len(result.pbit) == 3
    assert result.metrics["mean_time_s_greedy"] >= 0
    assert result.metrics["mean_time_s_pbit"] > 0
    for d in result.pbit:
        assert d.expert_mask.sum() <= cfg.top_k
        assert len(d.expert_indices) <= cfg.top_k


def test_greedy_top_k():
    logits = np.array([3.0, 1.0, 2.0, 0.5])
    mask = greedy_top_k(logits, top_k=2)
    assert mask.sum() == 2
    assert mask[0] and mask[2]
