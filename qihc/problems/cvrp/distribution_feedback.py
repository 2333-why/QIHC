"""Finite-population, quality-weighted feedback with a KL trust region.

These are empirical search distributions, not certified Gibbs posteriors.
Updates are conditioned on the current verified incumbent and neighborhood.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math

import numpy as np

from qihc.problems.cvrp.neighborhood import NeighborhoodProposal


def softmax(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not len(values) or not np.all(np.isfinite(values)):
        raise ValueError("softmax requires a non-empty finite vector")
    weights = np.exp(values - values.max())
    return weights / weights.sum()


def trust_region_step(prior: np.ndarray, target: np.ndarray, budget: float) -> tuple[np.ndarray, float]:
    """Project a log-ratio update along its ray; KL(new || prior) <= budget."""
    prior = np.asarray(prior, dtype=float)
    target = np.asarray(target, dtype=float)
    if (prior.ndim != 1 or prior.shape != target.shape or not len(prior)
            or not np.all(np.isfinite(prior)) or not np.all(np.isfinite(target))
            or np.any(prior < 0) or np.any(target < 0) or prior.sum() <= 0 or target.sum() <= 0
            or not np.isfinite(budget) or budget < 0):
        raise ValueError("KL projection requires finite nonnegative distributions and budget")
    prior = prior / prior.sum(); target = target / target.sum()
    result = np.zeros_like(prior)
    support = prior > 0
    if budget == 0 or target[support].sum() == 0:
        return result, 0.0
    log_prior = np.log(prior[support])
    supported_target = target[support] / target[support].sum()
    direction = np.log(np.maximum(supported_target, np.finfo(float).tiny)) - log_prior
    direction -= direction.mean()
    low, high = 0.0, 1.0
    for _ in range(40):
        fraction = (low + high) / 2.0
        candidate = softmax(log_prior + fraction * direction)
        positive = candidate > 0
        divergence = float(np.sum(candidate[positive] * (np.log(candidate[positive]) - log_prior[positive])))
        if divergence <= budget:
            low = fraction
        else:
            high = fraction
    step = low * direction
    updated = softmax(log_prior + step)
    positive = updated > 0
    divergence = float(np.sum(updated[positive] * (np.log(updated[positive]) - log_prior[positive])))
    result[support] = step
    return result, divergence


@dataclass
class DistributionalFeedback:
    learning_rate: float = 0.8
    kl_budget: float = 0.05
    pseudocount: float = 4.0
    min_unique: int = 2
    pair_limit: int = 8
    use_trust_region: bool = True
    mode: str = "population"
    residuals: dict[int, dict[int, float]] = field(default_factory=dict)
    diagnostics: dict = field(default_factory=dict)

    def __post_init__(self):
        if (not np.all(np.isfinite([self.learning_rate, self.kl_budget, self.pseudocount]))
                or not 0 <= self.learning_rate <= 1 or self.kl_budget < 0 or self.pseudocount <= 0):
            raise ValueError("Invalid feedback learning rate, KL budget or pseudocount")
        if self.mode not in {"population", "best"} or self.min_unique < 1 or self.pair_limit < 0:
            raise ValueError("Invalid feedback mode or population limits")

    def adjusted_logits(self, customer: int, base: dict[int, float]) -> dict[int, float]:
        if not base:
            return {}
        routes = list(base)
        values = np.asarray([float(base[r]) for r in routes])
        residual = np.asarray([self.residuals.get(customer, {}).get(r, 0.0) for r in routes])
        if self.use_trust_region:
            prior = softmax(values)
            residual, _ = trust_region_step(prior, softmax(values + residual), self.kl_budget)
        return {r: float(value + delta) for r, value, delta in zip(routes, values, residual)}

    def prompt_context(self, customers: list[int]) -> dict[int, dict[int, float]]:
        return {c: dict(self.residuals[c]) for c in customers if c in self.residuals}

    def observe(self, proposal, variables, sample_bits, accepted, improvement,
                feasible_mask=None, objectives=None) -> dict[int, dict[int, float]]:
        bits = np.asarray(sample_bits, dtype=np.int8)
        customers = list(proposal.destroy_customers)
        index = {v: i for i, v in enumerate(variables)}
        feasible = np.asarray(feasible_mask, dtype=bool)
        costs = np.asarray(objectives, dtype=float)
        if bits.ndim != 2 or feasible.shape != (len(bits),) or costs.shape != (len(bits),):
            raise ValueError("feedback population and verification arrays must align")
        self.residuals = {}  # State-specific residuals must not accumulate across different incumbents.
        valid = np.flatnonzero(feasible & np.isfinite(costs))
        rows = []
        kept = []
        for row_idx in valid:
            assignment = []
            for customer in customers:
                selected = [r for r in proposal.candidate_routes[customer]
                            if bits[row_idx, index[("assign", customer, r)]]]
                if len(selected) != 1:
                    break
                assignment.append(selected[0])
            else:
                rows.append(assignment); kept.append(int(row_idx))
        assignments = np.asarray(rows, dtype=int).reshape((-1, len(customers)))
        unique = len(np.unique(assignments, axis=0)) if len(assignments) else 0
        self.diagnostics = {
            "population_size": len(bits), "verified_samples": len(rows),
            "unique_assignments": unique, "mode": self.mode,
            "marginals": {}, "pair_correlations": [], "gated": True,
            "distribution_kind": "verified_empirical_search_population",
        }
        if not rows or (self.mode == "population" and unique < self.min_unique):
            return {}
        scores = costs[kept]
        if self.mode == "best":
            assignments = assignments[[int(np.argmin(scores))]]
            scores = np.asarray([scores.min()])
        scale = max(float(np.std(scores)), 1e-8)
        weights = np.exp(np.clip(-(scores - scores.min()) / scale, -40.0, 0.0))
        weights /= weights.sum()
        effective_n = min(float(1.0 / np.square(weights).sum()), float(unique))
        reliability = effective_n / (effective_n + self.pseudocount)
        self.diagnostics.update({"effective_sample_size": effective_n,
                                 "reliability": reliability, "gated": False})
        for position, customer in enumerate(customers):
            routes = proposal.candidate_routes[customer]
            prior = softmax(np.asarray([proposal.candidate_route_logits.get(customer, {}).get(r, 0.0)
                                        for r in routes]))
            empirical = np.asarray([weights[assignments[:, position] == r].sum() for r in routes])
            target = (1.0 - reliability) * prior + reliability * empirical
            blended = (1.0 - self.learning_rate) * prior + self.learning_rate * target
            if self.use_trust_region:
                step, divergence = trust_region_step(prior, blended, self.kl_budget)
            else:
                step = np.log(np.clip(blended, 1e-12, 1.0)) - np.log(np.clip(prior, 1e-12, 1.0))
                step -= step.mean()
                updated = softmax(np.log(np.clip(prior, 1e-12, 1.0)) + step)
                divergence = float(np.sum(updated * (np.log(updated) - np.log(np.clip(prior, 1e-12, 1.0)))))
            self.residuals[customer] = {r: float(v) for r, v in zip(routes, step)}
            self.diagnostics["marginals"][str(customer)] = {
                "routes": routes, "prior": prior.tolist(), "target": target.tolist(),
                "entropy": float(-np.sum(target * np.log(np.clip(target, 1e-12, 1.0)))),
                "kl_step": divergence,
            }
        if self.pair_limit and len(assignments) > 1:
            pairs = []
            for left in range(len(customers)):
                for right in range(left + 1, len(customers)):
                    joint = {}
                    for row, weight in zip(assignments, weights):
                        key = (int(row[left]), int(row[right]))
                        joint[key] = joint.get(key, 0.0) + float(weight)
                    lp, rp = {}, {}
                    for (a, b), mass in joint.items():
                        lp[a] = lp.get(a, 0.0) + mass; rp[b] = rp.get(b, 0.0) + mass
                    mutual_information = sum(mass * math.log(mass / (lp[a] * rp[b]))
                                             for (a, b), mass in joint.items() if mass > 0)
                    pairs.append({"customers": [customers[left], customers[right]],
                                  "mutual_information": max(0.0, mutual_information),
                                  "same_route_probability": sum(mass for (a, b), mass in joint.items() if a == b)})
            self.diagnostics["pair_correlations"] = sorted(
                pairs, key=lambda row: row["mutual_information"], reverse=True
            )[:self.pair_limit]
        return self.prompt_context(customers)
