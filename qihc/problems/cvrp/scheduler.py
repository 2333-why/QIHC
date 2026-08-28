"""QIHC-LNS closed loop with deterministic do-no-harm acceptance."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from qihc.ising.batched import NumpyPBitSampler, PBitSampleBatch, TorchPBitSampler
from qihc.problems.cvrp.instance import CVRPInstance, RouteSolution
from qihc.problems.cvrp.neighborhood import KNNNeighborhoodSelector, NeighborhoodSelector
from qihc.problems.cvrp.qubo import build_assignment_qubo, decode_assignment
from qihc.problems.cvrp.verifier import VerificationResult, greedy_initial_solution, verify_solution
from qihc.problems.cvrp.verifier import best_insertion


@dataclass
class LNSConfig:
    iterations: int = 100
    destroy_size: int = 8
    routes_per_customer: int = 3
    assignment_penalty: float = 40.0
    capacity_penalty: float = 12.0
    semantic_penalty: float = 20.0
    sampler: str = "numpy"
    sampling_steps: int = 300
    num_chains: int = 64
    top_samples: int = 16
    temperature_start: float = 5.0
    temperature_end: float = 0.05
    update_fraction: float = 0.25
    device: str = "cpu"
    seed: int = 0
    patience: int = 30


@dataclass
class LNSIterationRecord:
    iteration: int
    incumbent_before: float
    candidate_objective: float | None
    incumbent_after: float
    accepted: bool
    feasible_samples: int
    decoded_samples: int
    qubo_variables: int
    best_ising_energy: float
    pbit_elapsed_s: float
    selector_source: str
    proposal_confidence: float
    candidate_route_recall: float
    violations: list[dict] = field(default_factory=list)


@dataclass
class LNSResult:
    instance_name: str
    solution: RouteSolution
    verification: VerificationResult
    initial_verification: VerificationResult
    records: list[LNSIterationRecord]
    total_elapsed_s: float
    config: dict[str, Any]

    def to_summary(self) -> dict[str, Any]:
        initial = self.initial_verification.objective
        final = self.verification.objective
        return {
            "instance": self.instance_name,
            "initial_objective": initial,
            "final_objective": final,
            "improvement_fraction": (initial - final) / max(abs(initial), 1e-12),
            "feasible": self.verification.feasible,
            "accepted_moves": sum(record.accepted for record in self.records),
            "iterations": len(self.records),
            "total_elapsed_s": self.total_elapsed_s,
            "pbit_elapsed_s": sum(record.pbit_elapsed_s for record in self.records),
            "mean_qubo_variables": float(np.mean([r.qubo_variables for r in self.records])) if self.records else 0.0,
            "mean_candidate_route_recall": float(
                np.mean([r.candidate_route_recall for r in self.records])
            ) if self.records else 0.0,
            "llm_fallback_rate": float(
                np.mean([r.selector_source == "llm_fallback_knn" for r in self.records])
            ) if self.records else 0.0,
            "best_known_cost": None,
            "optimality_gap": None,
            "config": self.config,
        }


class QIHCLNSSolver:
    def __init__(
        self,
        config: LNSConfig | None = None,
        selector: NeighborhoodSelector | None = None,
    ):
        self.config = config or LNSConfig()
        self.selector = selector or KNNNeighborhoodSelector(
            self.config.destroy_size, self.config.routes_per_customer
        )

    def _sampler(self, seed: int):
        cfg = self.config
        common = dict(
            num_chains=cfg.num_chains,
            steps=cfg.sampling_steps,
            temperature_start=cfg.temperature_start,
            temperature_end=cfg.temperature_end,
            update_fraction=cfg.update_fraction,
            top_k=cfg.top_samples,
            seed=seed,
        )
        if cfg.sampler == "numpy":
            return NumpyPBitSampler(**common)
        if cfg.sampler == "torch":
            return TorchPBitSampler(**common, device=cfg.device)
        raise ValueError(f"Unknown sampler: {cfg.sampler}")

    @staticmethod
    def _candidate_route_recall(
        instance: CVRPInstance,
        incumbent: RouteSolution,
        proposal,
    ) -> float:
        destroyed = set(proposal.destroy_customers)
        fixed = [[customer for customer in route if customer not in destroyed] for route in incumbent.routes]
        fixed.extend([[] for _ in range(instance.vehicle_count - len(fixed))])
        hits = 0
        for customer in proposal.destroy_customers:
            demand = instance.customers[customer].demand
            feasible = []
            for route_idx, route in enumerate(fixed[: instance.vehicle_count]):
                load = sum(instance.customers[c].demand for c in route)
                if load + demand <= instance.vehicle_capacity:
                    _, delta = best_insertion(instance, route, customer)
                    feasible.append((delta, route_idx))
            if feasible and min(feasible)[1] in proposal.candidate_routes[customer]:
                hits += 1
        return hits / max(len(proposal.destroy_customers), 1)

    def solve(self, instance: CVRPInstance, initial: RouteSolution | None = None) -> LNSResult:
        t0 = time.perf_counter()
        incumbent = initial.copy() if initial else greedy_initial_solution(instance)
        incumbent_result = verify_solution(instance, incumbent)
        if not incumbent_result.feasible:
            raise ValueError(f"Initial solution is infeasible: {incumbent_result.violations}")
        initial_result = incumbent_result
        records: list[LNSIterationRecord] = []
        stale = 0
        for iteration in range(self.config.iterations):
            proposal = self.selector.propose(
                instance, incumbent, iteration, self.config.seed
            )
            candidate_route_recall = self._candidate_route_recall(instance, incumbent, proposal)
            problem = build_assignment_qubo(
                instance,
                incumbent,
                proposal,
                assignment_penalty=self.config.assignment_penalty,
                capacity_penalty=self.config.capacity_penalty,
                semantic_penalty=self.config.semantic_penalty,
            )
            weight, field, _ = problem.model.to_ising()
            sampled: PBitSampleBatch = self._sampler(self.config.seed + 7919 * iteration).solve(
                weight, field, problem.initial_bits
            )
            candidate_solution: RouteSolution | None = None
            candidate_result: VerificationResult | None = None
            feasible_samples = 0
            last_violations: list[dict] = []
            sample_bits = [problem.initial_bits, *list(sampled.bits)]
            for bits in sample_bits:
                try:
                    decoded = decode_assignment(instance, problem, bits)
                    verified = verify_solution(instance, decoded)
                except ValueError as exc:
                    last_violations = [{"type": "decode", "message": str(exc)}]
                    continue
                if not verified.feasible:
                    last_violations = verified.violations
                    continue
                feasible_samples += 1
                if candidate_result is None or verified.objective < candidate_result.objective:
                    candidate_solution, candidate_result = decoded, verified

            before = incumbent_result.objective
            accepted = bool(candidate_result and candidate_result.objective < before - 1e-9)
            if accepted and candidate_solution and candidate_result:
                incumbent, incumbent_result = candidate_solution, candidate_result
                stale = 0
            else:
                stale += 1
            records.append(
                LNSIterationRecord(
                    iteration=iteration,
                    incumbent_before=before,
                    candidate_objective=candidate_result.objective if candidate_result else None,
                    incumbent_after=incumbent_result.objective,
                    accepted=accepted,
                    feasible_samples=feasible_samples,
                    decoded_samples=len(sample_bits),
                    qubo_variables=problem.num_variables,
                    best_ising_energy=float(sampled.energies.min()) if sampled.energies.size else float("nan"),
                    pbit_elapsed_s=sampled.elapsed_s,
                    selector_source=proposal.source,
                    proposal_confidence=proposal.confidence,
                    candidate_route_recall=candidate_route_recall,
                    violations=last_violations,
                )
            )
            if stale >= self.config.patience:
                break
        result = LNSResult(
            instance_name=instance.name,
            solution=incumbent,
            verification=incumbent_result,
            initial_verification=initial_result,
            records=records,
            total_elapsed_s=time.perf_counter() - t0,
            config=asdict(self.config),
        )
        summary = result.to_summary()
        if instance.best_known_cost:
            summary["best_known_cost"] = instance.best_known_cost
            summary["optimality_gap"] = (
                result.verification.cost - instance.best_known_cost
            ) / instance.best_known_cost
        result.solution.metadata["summary"] = summary
        return result
