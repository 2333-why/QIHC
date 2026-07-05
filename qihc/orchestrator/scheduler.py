"""QIHC closed-loop scheduler: frontend -> encode -> p-bit -> feedback."""

from __future__ import annotations

import time

import numpy as np

from qihc.orchestrator.backend import PBitBackend
from qihc.orchestrator.config import QIHCConfig
from qihc.orchestrator.encoder import (
    batch_moe_logits_to_ising,
    capacity_aware_routing_score,
    greedy_top_k,
    load_balance_score,
    moe_logits_to_ising,
    mask_to_expert_indices,
    refine_mask_to_top_k,
    routing_score,
)
from qihc.orchestrator.frontend.base import build_frontend
from qihc.orchestrator.types import RoutingBatchResult, RoutingDecision


class QIHCOrchestrator:
    """
    Minimal QIHC loop (B-tier ready, upgradeable).

    Flow per request
    ----------------
    1. AI frontend produces routing logits
    2. QUBO/Ising encoding
    3. p-bit backend sampling
    4. Compare against greedy top-k baseline
  """

    def __init__(self, config: QIHCConfig | None = None):
        self.config = config or QIHCConfig.tier_b()
        self.frontend = build_frontend(self.config)
        self.backend = PBitBackend(self.config)
        np.random.seed(self.config.seed)

    def route_one_greedy(self, logits: np.ndarray, text: str = "") -> RoutingDecision:
        t0 = time.perf_counter()
        mask = greedy_top_k(logits, self.config.top_k)
        return RoutingDecision(
            text=text,
            logits=logits,
            expert_mask=mask,
            expert_indices=mask_to_expert_indices(mask),
            energy=None,
            elapsed_s=time.perf_counter() - t0,
            method="greedy_topk",
        )

    def route_one_pbit(self, logits: np.ndarray, text: str = "") -> RoutingDecision:
        weight, field = moe_logits_to_ising(
            logits,
            top_k=self.config.top_k,
            penalty=self.config.cardinality_penalty,
        )
        mask, energy, elapsed = self.backend.solve(weight, field)
        mask = refine_mask_to_top_k(logits, mask, self.config.top_k)
        return RoutingDecision(
            text=text,
            logits=logits,
            expert_mask=mask,
            expert_indices=mask_to_expert_indices(mask),
            energy=energy,
            elapsed_s=elapsed,
            method=f"pbit_{self.config.sampler}",
        )

    def route_batch_pbit_joint(self, contexts) -> list[RoutingDecision]:
        logits_batch = np.stack([ctx.logits for ctx in contexts], axis=0)
        capacity = self.config.resolve_capacity(len(contexts))
        weight, field = batch_moe_logits_to_ising(
            logits_batch,
            top_k=self.config.top_k,
            expert_capacity=capacity,
            row_penalty=self.config.cardinality_penalty,
            col_penalty=self.config.capacity_penalty,
        )

        best_masks = None
        best_score = -np.inf
        best_energy = 0.0
        total_elapsed = 0.0
        n_restarts = 3

        for restart in range(n_restarts):
            saved_seed = self.config.seed + restart
            np.random.seed(saved_seed)
            masks, energy, elapsed = self.backend.solve(weight, field, logits_batch=logits_batch)
            score = capacity_aware_routing_score(
                logits_batch,
                masks,
                capacity,
                self.config.overflow_penalty,
            )
            total_elapsed += elapsed
            if score > best_score:
                best_score = score
                best_masks = masks
                best_energy = energy

        assert best_masks is not None
        per_elapsed = total_elapsed / max(len(contexts) * n_restarts, 1)
        decisions: list[RoutingDecision] = []
        for ctx, mask in zip(contexts, best_masks):
            decisions.append(
                RoutingDecision(
                    text=ctx.text,
                    logits=ctx.logits,
                    expert_mask=mask,
                    expert_indices=mask_to_expert_indices(mask),
                    energy=best_energy,
                    elapsed_s=per_elapsed,
                    method=f"pbit_joint_{self.config.sampler}",
                )
            )
        return decisions

    def run_batch(self, texts: list[str]) -> RoutingBatchResult:
        contexts = self.frontend.encode_batch(texts)
        logits_batch = np.stack([ctx.logits for ctx in contexts], axis=0)
        capacity = self.config.resolve_capacity(len(contexts))

        greedy_decisions: list[RoutingDecision] = []
        for ctx in contexts:
            greedy_decisions.append(self.route_one_greedy(ctx.logits, ctx.text))

        if self.config.batch_joint and len(contexts) > 1:
            pbit_decisions = self.route_batch_pbit_joint(contexts)
        else:
            pbit_decisions = [
                self.route_one_pbit(ctx.logits, ctx.text) for ctx in contexts
            ]

        greedy_masks = [d.expert_mask for d in greedy_decisions]
        pbit_masks = [d.expert_mask for d in pbit_decisions]

        overflow = self.config.overflow_penalty
        greedy_scores_raw = [routing_score(d.logits, d.expert_mask) for d in greedy_decisions]
        pbit_scores_raw = [routing_score(d.logits, d.expert_mask) for d in pbit_decisions]

        greedy_score = capacity_aware_routing_score(
            logits_batch, greedy_masks, capacity, overflow
        )
        pbit_score = capacity_aware_routing_score(
            logits_batch, pbit_masks, capacity, overflow
        )

        metrics = {
            "mean_routing_score_greedy": greedy_score,
            "mean_routing_score_pbit": pbit_score,
            "routing_score_gain": pbit_score - greedy_score,
            "mean_routing_score_raw_greedy": float(np.mean(greedy_scores_raw)),
            "mean_routing_score_raw_pbit": float(np.mean(pbit_scores_raw)),
            "routing_score_gain_raw": float(np.mean(pbit_scores_raw) - np.mean(greedy_scores_raw)),
            "expert_capacity": float(capacity),
            "load_balance_greedy": load_balance_score(greedy_masks),
            "load_balance_pbit": load_balance_score(pbit_masks),
            "load_balance_gain": load_balance_score(pbit_masks) - load_balance_score(greedy_masks),
            "mean_time_s_greedy": float(np.mean([d.elapsed_s for d in greedy_decisions])),
            "mean_time_s_pbit": float(np.mean([d.elapsed_s for d in pbit_decisions])),
        }

        return RoutingBatchResult(
            greedy=greedy_decisions,
            pbit=pbit_decisions,
            metrics=metrics,
            config_summary=self.config.to_summary(),
        )
