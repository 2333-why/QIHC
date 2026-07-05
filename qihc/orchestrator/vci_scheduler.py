"""VCI (Variational Co-Inference) orchestrator: q <-> s alternate loop."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from qihc.orchestrator.backend import PBitBackend
from qihc.orchestrator.config import QIHCConfig
from qihc.orchestrator.encoder import greedy_top_k, refine_mask_to_top_k, routing_score
from qihc.orchestrator.free_energy import FreeEnergyResult, compute_free_energy
from qihc.orchestrator.frontend.base import build_frontend
from qihc.orchestrator.reasoning import (
    SubsetProblem,
    count_exclusion_violations,
    is_feasible,
    subset_to_ising,
)

VCIMode = Literal["greedy", "vci-0", "vci-1", "vci-2", "vci-full"]


@dataclass
class VCIRoundRecord:
    round_index: int
    mask: np.ndarray
    expert_indices: list[int]
    ising_energy: float
    elapsed_s: float
    free_energy: FreeEnergyResult
    feasible: bool


@dataclass
class VCIResult:
    problem: SubsetProblem
    mode: VCIMode
    q0_logits: np.ndarray
    final_logits: np.ndarray
    final_mask: np.ndarray
    rounds: list[VCIRoundRecord] = field(default_factory=list)
    total_elapsed_s: float = 0.0

    @property
    def n_rounds(self) -> int:
        return len(self.rounds)

    @property
    def final_feasible(self) -> bool:
        if not self.rounds:
            return False
        return self.rounds[-1].feasible

    @property
    def final_free_energy(self) -> float:
        if not self.rounds:
            return float("inf")
        return self.rounds[-1].free_energy.total

    @property
    def final_semantic_score(self) -> float:
        return routing_score(self.final_logits, self.final_mask)


@dataclass
class VCIConfig(QIHCConfig):
    """QIHC config extended with VCI loop hyper-parameters."""

    max_rounds: int = 2
    f_epsilon: float = 1e-2
    refine_penalty: float = 1.5
    exclusion_penalty: float = 4.0
    kl_weight: float = 0.5
    violation_weight: float = 3.0
    beta: float = 1.0

    @classmethod
    def tier_a(cls, **overrides) -> "VCIConfig":
        base = cls(
            frontend="mock",
            model_name="mock",
            sampling_steps=200,
            num_experts=6,
            top_k=3,
            max_rounds=2,
            seed=0,
        )
        for k, v in overrides.items():
            setattr(base, k, v)
        return base


class VCIOrchestrator:
    """
    Variational Co-Inference loop for subset-selection (Case A).

    Modes
    -----
    greedy  : top-k only
    vci-0   : alias greedy
    vci-1   : single s-step (CR limit), no q refine
    vci-2   : up to 2 rounds with q refine (NSFC default)
    vci-full: up to max_rounds with F convergence
    """

    def __init__(self, config: VCIConfig | None = None):
        self.config = config or VCIConfig.tier_a()
        self.frontend = build_frontend(self.config)
        self.backend = PBitBackend(self.config)
        np.random.seed(self.config.seed)

    def solve_subset(
        self,
        problem: SubsetProblem,
        mode: VCIMode = "vci-2",
    ) -> VCIResult:
        t0 = time.perf_counter()
        logits = np.asarray(problem.logits, dtype=float).ravel()
        q0 = logits.copy()
        ctx_logits = logits.copy()

        if mode == "greedy" or mode == "vci-0":
            mask = greedy_top_k(ctx_logits, problem.top_k)
            fe = compute_free_energy(
                ctx_logits,
                q0,
                mask,
                exclusion_pairs=problem.exclusion_pairs,
                beta=self.config.beta,
                kl_weight=self.config.kl_weight,
                violation_weight=self.config.violation_weight,
                refine_penalty=self.config.refine_penalty,
            )
            record = VCIRoundRecord(
                round_index=0,
                mask=mask,
                expert_indices=list(np.flatnonzero(mask)),
                ising_energy=0.0,
                elapsed_s=0.0,
                free_energy=fe,
                feasible=is_feasible(mask, problem.top_k, problem.exclusion_pairs),
            )
            return VCIResult(
                problem=problem,
                mode=mode,
                q0_logits=q0,
                final_logits=ctx_logits,
                final_mask=mask,
                rounds=[record],
                total_elapsed_s=time.perf_counter() - t0,
            )

        max_rounds = self._resolve_max_rounds(mode)
        rounds: list[VCIRoundRecord] = []
        prev_f = float("inf")

        for t in range(max_rounds):
            weight, field = subset_to_ising(
                ctx_logits,
                top_k=problem.top_k,
                cardinality_penalty=self.config.cardinality_penalty,
                exclusion_pairs=problem.exclusion_pairs,
                exclusion_penalty=self.config.exclusion_penalty,
            )
            mask, energy, elapsed = self.backend.solve(weight, field)
            mask = refine_mask_to_top_k(ctx_logits, mask, problem.top_k)

            fe = compute_free_energy(
                ctx_logits,
                q0,
                mask,
                ising_energy=energy,
                exclusion_pairs=problem.exclusion_pairs,
                beta=self.config.beta,
                kl_weight=self.config.kl_weight,
                violation_weight=self.config.violation_weight,
                refine_penalty=self.config.refine_penalty,
            )
            feasible = is_feasible(mask, problem.top_k, problem.exclusion_pairs)
            rounds.append(
                VCIRoundRecord(
                    round_index=t,
                    mask=mask.copy(),
                    expert_indices=list(np.flatnonzero(mask)),
                    ising_energy=float(energy),
                    elapsed_s=float(elapsed),
                    free_energy=fe,
                    feasible=feasible,
                )
            )

            if feasible:
                break
            if abs(prev_f - fe.total) < self.config.f_epsilon and t > 0:
                break
            prev_f = fe.total

            if mode == "vci-1":
                break
            if t >= max_rounds - 1:
                break

            from qihc.orchestrator.types import RoutingContext

            ctx = RoutingContext(text=problem.text, logits=ctx_logits)
            refined = self.frontend.refine(ctx, mask, fe.feedback, q0)
            ctx_logits = refined.logits

        final = rounds[-1]
        return VCIResult(
            problem=problem,
            mode=mode,
            q0_logits=q0,
            final_logits=ctx_logits,
            final_mask=final.mask.copy(),
            rounds=rounds,
            total_elapsed_s=time.perf_counter() - t0,
        )

    def compare_modes(
        self,
        problems: list[SubsetProblem],
        modes: list[VCIMode] | None = None,
    ) -> dict:
        modes = modes or ["greedy", "vci-1", "vci-2"]
        summary: dict[str, dict] = {}
        for mode in modes:
            results = [self.solve_subset(p, mode=mode) for p in problems]
            multi_round = [r for r in results if r.n_rounds > 1]
            summary[mode] = {
                "feasible_rate": float(np.mean([r.final_feasible for r in results])),
                "mean_semantic_score": float(np.mean([r.final_semantic_score for r in results])),
                "mean_rounds": float(np.mean([r.n_rounds for r in results])),
                "mean_free_energy": float(np.mean([r.final_free_energy for r in results])),
                "mean_time_s": float(np.mean([r.total_elapsed_s for r in results])),
                "multi_round_rate": float(len(multi_round) / max(len(results), 1)),
            }
        return summary

    def trace_refine_demo(self, problem: SubsetProblem) -> list[VCIRoundRecord]:
        """
        Explicit two-step VCI narrative for visualization:

        1) Greedy decode (often violates IF) → compute F → q-step refine
        2) p-bit s-step on refined q → feasible decode

        Use when a single s-step already satisfies constraints but we need
        to illustrate q ↔ s co-inference for documentation.
        """
        from qihc.orchestrator.types import RoutingContext

        q0 = np.asarray(problem.logits, dtype=float).ravel()
        ctx_logits = q0.copy()
        rounds: list[VCIRoundRecord] = []

        mask0 = greedy_top_k(ctx_logits, problem.top_k)
        fe0 = compute_free_energy(
            ctx_logits,
            q0,
            mask0,
            exclusion_pairs=problem.exclusion_pairs,
            beta=self.config.beta,
            kl_weight=self.config.kl_weight,
            violation_weight=self.config.violation_weight,
            refine_penalty=self.config.refine_penalty,
        )
        rounds.append(
            VCIRoundRecord(
                round_index=0,
                mask=mask0.copy(),
                expert_indices=list(np.flatnonzero(mask0)),
                ising_energy=0.0,
                elapsed_s=0.0,
                free_energy=fe0,
                feasible=is_feasible(mask0, problem.top_k, problem.exclusion_pairs),
            )
        )

        ctx = RoutingContext(text=problem.text, logits=ctx_logits)
        refined = self.frontend.refine(ctx, mask0, fe0.feedback, q0)
        ctx_logits = refined.logits

        weight, field = subset_to_ising(
            ctx_logits,
            top_k=problem.top_k,
            cardinality_penalty=self.config.cardinality_penalty,
            exclusion_pairs=problem.exclusion_pairs,
            exclusion_penalty=self.config.exclusion_penalty,
        )
        mask1, energy, elapsed = self.backend.solve(weight, field)
        mask1 = refine_mask_to_top_k(ctx_logits, mask1, problem.top_k)
        fe1 = compute_free_energy(
            ctx_logits,
            q0,
            mask1,
            ising_energy=energy,
            exclusion_pairs=problem.exclusion_pairs,
            beta=self.config.beta,
            kl_weight=self.config.kl_weight,
            violation_weight=self.config.violation_weight,
            refine_penalty=self.config.refine_penalty,
        )
        rounds.append(
            VCIRoundRecord(
                round_index=1,
                mask=mask1.copy(),
                expert_indices=list(np.flatnonzero(mask1)),
                ising_energy=float(energy),
                elapsed_s=float(elapsed),
                free_energy=fe1,
                feasible=is_feasible(mask1, problem.top_k, problem.exclusion_pairs),
            )
        )
        return rounds

    @staticmethod
    def _resolve_max_rounds(mode: VCIMode) -> int:
        if mode == "vci-1":
            return 1
        if mode == "vci-2":
            return 2
        if mode == "vci-full":
            return 4
        return 1
