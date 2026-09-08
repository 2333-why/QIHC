"""Online p-bit-to-LLM logit feedback for candidate-search proposals."""

from __future__ import annotations

from dataclasses import dataclass, field
import math

import numpy as np

from qihc.problems.cvrp.neighborhood import NeighborhoodProposal


@dataclass
class PBitLogitFeedback:
    """Maintain route-level logit residuals learned from p-bit sample marginals.

    This is an online inference-time feedback path.  It does not change language
    model weights; the same traces are also exported for SFT/DPO/GRPO training.
    """

    learning_rate: float = 0.8
    decay: float = 0.95
    clip: float = 4.0
    negative_weight: float = 0.5
    elite_fraction: float = 0.25
    residuals: dict[int, dict[int, float]] = field(default_factory=dict)

    def adjusted_logits(self, customer: int, base: dict[int, float]) -> dict[int, float]:
        residual = self.residuals.get(customer, {})
        return {
            int(route): float(logit) + float(residual.get(int(route), 0.0))
            for route, logit in base.items()
        }

    def observe(
        self,
        proposal: NeighborhoodProposal,
        variables: list[tuple],
        sample_bits: np.ndarray,
        accepted: bool,
        improvement: float,
        feasible_mask: np.ndarray | None = None,
        objectives: np.ndarray | None = None,
    ) -> dict[int, dict[int, float]]:
        bits = np.asarray(sample_bits, dtype=float)
        if bits.ndim == 1:
            bits = bits[None, :]
        count = len(bits)
        feasible = np.ones(count, dtype=bool) if feasible_mask is None else np.asarray(feasible_mask, dtype=bool)
        scores = np.zeros(count, dtype=float) if objectives is None else np.asarray(objectives, dtype=float)
        if feasible.shape != (count,) or scores.shape != (count,):
            raise ValueError("feedback masks/objectives must match sample count")
        feasible_indices = np.flatnonzero(feasible & np.isfinite(scores))
        elite_count = max(1, int(math.ceil(len(feasible_indices) * self.elite_fraction))) if len(feasible_indices) else 0
        elite_indices = feasible_indices[np.argsort(scores[feasible_indices])[:elite_count]] if elite_count else np.asarray([], dtype=int)
        negative_indices = np.flatnonzero(~feasible)
        if not len(negative_indices) and len(feasible_indices) > elite_count:
            negative_indices = feasible_indices[np.argsort(scores[feasible_indices])[elite_count:]]
        advantage = 1.0 + math.tanh(max(0.0, float(improvement))) if accepted else 0.25
        updates: dict[int, dict[int, float]] = {}
        for customer in proposal.destroy_customers:
            routes = proposal.candidate_routes[customer]
            base = proposal.candidate_route_logits.get(customer, {route: 0.0 for route in routes})
            # The scheduler passes the already calibrated proposal logits.  Do
            # not add residuals a second time while estimating the prior.
            values = np.asarray([base.get(route, 0.0) for route in routes], dtype=float)
            values -= values.max(initial=0.0)
            prior = np.exp(values); prior /= max(prior.sum(), 1e-12)
            positive = []
            negative = []
            for route in routes:
                try:
                    idx = variables.index(("assign", customer, route))
                    positive.append(float(bits[elite_indices, idx].mean()) if len(elite_indices) else 0.0)
                    negative.append(float(bits[negative_indices, idx].mean()) if len(negative_indices) else 0.0)
                except ValueError:
                    positive.append(0.0); negative.append(0.0)
            positive = np.asarray(positive, dtype=float)
            negative = np.asarray(negative, dtype=float)
            positive /= max(positive.sum(), 1e-12)
            negative /= max(negative.sum(), 1e-12)
            customer_updates = {}
            state = self.residuals.setdefault(customer, {})
            signal = positive - self.negative_weight * negative - prior
            for route, delta in zip(routes, advantage * signal):
                value = self.decay * state.get(route, 0.0) + self.learning_rate * float(delta)
                state[route] = float(np.clip(value, -self.clip, self.clip))
                customer_updates[route] = state[route]
            updates[customer] = customer_updates
        return updates

    def prompt_context(self, customers: list[int]) -> dict[int, dict[int, float]]:
        return {customer: dict(self.residuals[customer]) for customer in customers if customer in self.residuals}
