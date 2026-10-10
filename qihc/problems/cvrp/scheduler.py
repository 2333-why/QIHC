"""QIHC-LNS closed loop with deterministic do-no-harm acceptance."""

from __future__ import annotations

import time
import itertools
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from qihc.ising.batched import NumpyPBitSampler, PBitSampleBatch, TorchPBitSampler
from qihc.problems.cvrp.instance import CVRPInstance, RouteSolution
from qihc.problems.cvrp.neighborhood import KNNNeighborhoodSelector, NeighborhoodSelector, safely_expand_proposal, canonicalize_proposal
from qihc.problems.cvrp.qubo import build_assignment_qubo, decode_assignment
from qihc.problems.cvrp.verifier import VerificationResult, greedy_initial_solution, verify_solution
from qihc.problems.cvrp.verifier import best_insertion
from qihc.problems.cvrp.logit_feedback import PBitLogitFeedback


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
    proposal_bias: float = 1.0
    logit_feedback_rate: float = 0.8
    enable_logit_feedback: bool = True
    safe_candidate_expansion: bool = True
    safe_heuristic_routes: int = 1
    safe_random_routes: int = 1
    feedback_elite_fraction: float = 0.25
    feedback_negative_weight: float = 0.5
    initialization: str = "incumbent"
    cold_batch_size: int = 6
    cold_routes_per_customer: int = 4
    cold_selector_mode: str = "selector"
    cold_proposal_bias: float = 0.25
    cold_confidence_threshold: float = 0.55
    cold_max_abs_logit: float = 1.0
    cold_quality_attempts: int = 1
    search_backend: str = "pbit"
    feedback_mode: str = "legacy"
    feedback_source: str = "population"
    feedback_population_limit: int = 128
    feedback_kl_budget: float = 0.05
    feedback_trust_region: bool = True
    feedback_pairs: bool = True
    feedback_prompt: bool = True
    feedback_token: bool = True
    feedback_energy: bool = True
    search_time_limit_s: float = 0.0
    backend_time_limit_s: float = 5.0
    block_updates: bool = False
    constraint_aware_decode: bool = False
    temperature_ladder: bool = False
    normalize_objective: bool = False
    time_budget_only: bool = False
    strict_assignment_decode: bool = False
    shuffle_feedback_quality: bool = False

    def __post_init__(self):
        if self.iterations < 0 or self.patience < 0 or self.search_time_limit_s < 0:
            raise ValueError("iterations, patience and time limit must be nonnegative")
        if self.time_budget_only and self.search_time_limit_s <= 0:
            raise ValueError("time-budget-only requires a positive search time limit")
        if self.num_chains <= 0 or self.sampling_steps <= 0 or not 0 < self.top_samples <= self.num_chains:
            raise ValueError("Require positive chains/steps and 0 < top_samples <= num_chains")
        if self.feedback_population_limit <= 0 or self.feedback_kl_budget < 0:
            raise ValueError("feedback population must be positive and KL budget nonnegative")
        if not 1 <= self.cold_quality_attempts <= 4 or self.cold_batch_size <= 0 or self.cold_routes_per_customer <= 0:
            raise ValueError("Cold construction requires positive domain sizes and 1..4 quality attempts")
        if self.search_backend not in {"pbit", "pbit-qubo", "cp-sat", "metropolis", "uniform"}:
            raise ValueError("Unknown search backend")
        if self.feedback_mode not in {"legacy", "population", "best"}:
            raise ValueError("Unknown feedback mode")


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
    candidate_compression: float
    violations: list[dict] = field(default_factory=list)
    objective_improvement: float = 0.0
    pbit_logit_updates: dict[int, dict[int, float]] = field(default_factory=dict)
    proposal_payload: dict[str, Any] = field(default_factory=dict)
    sampler_backend: str = "ising-qubo"
    feedback_diagnostics: dict = field(default_factory=dict)
    teacher_payload: dict = field(default_factory=dict)
    incumbent_routes_before: list[list[int]] = field(default_factory=list)
    incumbent_routes_after: list[list[int]] = field(default_factory=list)
    elapsed_s: float = 0.0
    sampler_metadata: dict = field(default_factory=dict)
    sample_decode_diagnostics: dict = field(default_factory=dict)
    energy_feedback_context: dict = field(default_factory=dict)


@dataclass
class LNSResult:
    instance_name: str
    solution: RouteSolution
    verification: VerificationResult
    initial_verification: VerificationResult
    records: list[LNSIterationRecord]
    total_elapsed_s: float
    config: dict[str, Any]
    construction_pbit_s: float = 0.0
    construction_batches: int = 0
    search_elapsed_s: float = 0.0
    stop_reason: str = "iteration_limit"
    initial_solution: RouteSolution | None = None

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
            "search_elapsed_s": self.search_elapsed_s,
            "stop_reason": self.stop_reason,
            "pbit_elapsed_s": sum(record.pbit_elapsed_s for record in self.records),
            "construction_pbit_s": self.construction_pbit_s,
            "construction_batches": self.construction_batches,
            "construction_guarded_batches": int(
                self.solution.metadata.get("guarded_batches", 0)
            ),
            "construction_attempts": int(
                self.solution.metadata.get("construction_attempts", 1)
            ),
            "successful_construction_attempts": int(
                self.solution.metadata.get("successful_construction_attempts", 1)
            ),
            "cold_route_limit": self.solution.metadata.get("cold_route_limit"),
            "mean_qubo_variables": float(np.mean([r.qubo_variables for r in self.records])) if self.records else 0.0,
            "mean_candidate_route_recall": float(
                np.mean([r.candidate_route_recall for r in self.records])
            ) if self.records else 0.0,
            "mean_candidate_compression": float(
                np.mean([r.candidate_compression for r in self.records])
            ) if self.records else 0.0,
            "total_pbit_improvement": float(sum(r.objective_improvement for r in self.records)),
            "llm_fallback_rate": float(
                np.mean(["llm_fallback_knn" in r.selector_source for r in self.records])
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
        self.logit_feedback = PBitLogitFeedback(
            learning_rate=self.config.logit_feedback_rate,
            elite_fraction=self.config.feedback_elite_fraction,
            negative_weight=self.config.feedback_negative_weight,
        )
        if self.config.feedback_mode != "legacy":
            from qihc.problems.cvrp.distribution_feedback import DistributionalFeedback
            self.logit_feedback = DistributionalFeedback(
                learning_rate=self.config.logit_feedback_rate,
                kl_budget=self.config.feedback_kl_budget,
                pair_limit=8 if self.config.feedback_pairs else 0,
                use_trust_region=self.config.feedback_trust_region,
                mode=self.config.feedback_mode,
            )

    def _sampler(self, seed: int, time_limit_s: float | None = None):
        cfg = self.config
        common = dict(
            num_chains=cfg.num_chains,
            steps=cfg.sampling_steps,
            temperature_start=cfg.temperature_start,
            temperature_end=cfg.temperature_end,
            update_fraction=cfg.update_fraction,
            top_k=cfg.top_samples,
            seed=seed,
            time_limit_s=time_limit_s,
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
        self.logit_feedback.residuals.clear()
        if hasattr(self.logit_feedback, "diagnostics"):
            self.logit_feedback.diagnostics = {}
        if hasattr(self.selector, "reset_search_state"):
            self.selector.reset_search_state()
        construction_pbit_s = 0.0
        construction_batches = 0
        if initial is not None:
            incumbent = initial.copy()
        elif self.config.initialization == "pbit-cold":
            from qihc.problems.cvrp.cold_start import construct_with_pbit
            if hasattr(self.selector, "set_pbit_logit_feedback"):
                self.selector.set_pbit_logit_feedback({})
            cold = construct_with_pbit(
                instance, self._sampler, batch_size=self.config.cold_batch_size,
                routes_per_customer=self.config.cold_routes_per_customer,
                seed=self.config.seed,
                selector=(self.selector if self.config.cold_selector_mode == "selector" else None),
                proposal_bias=self.config.cold_proposal_bias,
                confidence_threshold=self.config.cold_confidence_threshold,
                max_abs_logit=self.config.cold_max_abs_logit,
                quality_attempts=self.config.cold_quality_attempts,
            )
            incumbent, construction_pbit_s, construction_batches = cold.solution, cold.pbit_elapsed_s, cold.batches
        else:
            incumbent = greedy_initial_solution(instance)
        incumbent_result = verify_solution(instance, incumbent)
        if not incumbent_result.feasible:
            raise ValueError(f"Initial solution is infeasible: {incumbent_result.violations}")
        initial_result = incumbent_result
        initial_solution = incumbent.copy()
        initial_metadata = dict(incumbent.metadata)
        records: list[LNSIterationRecord] = []
        stale = 0
        search_started = time.perf_counter()
        stop_reason = "iteration_limit"
        for iteration in itertools.count():
            if not self.config.time_budget_only and iteration >= self.config.iterations:
                break
            if self.config.search_time_limit_s and time.perf_counter() - search_started >= self.config.search_time_limit_s:
                stop_reason = "time_limit"
                break
            before_routes = [list(route) for route in incumbent.routes]
            energy_feedback_context = self.logit_feedback.prompt_context(instance.customer_ids) if self.config.enable_logit_feedback and self.config.feedback_energy else {}
            if self.config.enable_logit_feedback and hasattr(self.selector, "set_pbit_logit_feedback"):
                feedback = self.logit_feedback.prompt_context(instance.customer_ids)
                if hasattr(self.selector, "set_feedback_channels"):
                    self.selector.set_feedback_channels(
                        feedback if self.config.feedback_prompt else {},
                        feedback if self.config.feedback_token else {},
                        getattr(self.logit_feedback, "diagnostics", {}) if self.config.feedback_prompt else {},
                    )
                else:
                    self.selector.set_pbit_logit_feedback(feedback if self.config.feedback_prompt else {})
            proposal = self.selector.propose(
                instance, incumbent, iteration, self.config.seed
            )
            if self.config.safe_candidate_expansion:
                proposal = safely_expand_proposal(
                    instance, incumbent, proposal,
                    heuristic_routes=self.config.safe_heuristic_routes,
                    random_routes=self.config.safe_random_routes,
                    seed=self.config.seed + 3571 * iteration,
                )
            proposal = canonicalize_proposal(proposal)
            for customer in proposal.destroy_customers:
                supplied = proposal.candidate_route_logits.get(customer, {})
                base = {route: float(supplied.get(route, 0.0)) for route in proposal.candidate_routes[customer]}
                proposal.candidate_route_logits[customer] = (
                    self.logit_feedback.adjusted_logits(customer, base)
                    if self.config.enable_logit_feedback and self.config.feedback_energy else base
                )
            candidate_route_recall = self._candidate_route_recall(instance, incumbent, proposal)
            candidate_count = sum(len(routes) for routes in proposal.candidate_routes.values())
            full_count = len(proposal.destroy_customers) * instance.vehicle_count
            candidate_compression = 1.0 - candidate_count / max(full_count, 1)
            problem = build_assignment_qubo(
                instance,
                incumbent,
                proposal,
                assignment_penalty=self.config.assignment_penalty,
                capacity_penalty=self.config.capacity_penalty,
                semantic_penalty=self.config.semantic_penalty,
                proposal_bias=self.config.proposal_bias,
                normalize_objective=self.config.normalize_objective,
            )
            compilation = instance.metadata.get("constraint_compilation") or {}
            representations = {
                item.get("representation") for item in compilation.get("constraints", [])
                if item.get("execution") != "verifier_only"
            }
            remaining_s = None
            if self.config.search_time_limit_s:
                remaining_s = self.config.search_time_limit_s - (time.perf_counter() - search_started)
                if remaining_s <= 0:
                    stop_reason = "time_limit"
                    break
            if self.config.search_backend in {"cp-sat", "metropolis", "uniform"}:
                from qihc.problems.cvrp.classical_assignment import solve_classical_assignment
                backend_limit = self.config.backend_time_limit_s
                if self.config.search_time_limit_s:
                    backend_limit = min(backend_limit, max(0.001, self.config.search_time_limit_s - (time.perf_counter() - search_started)))
                sampled = solve_classical_assignment(
                    problem, self.config.search_backend,
                    seed=self.config.seed + 7919 * iteration,
                    num_chains=self.config.num_chains, steps=self.config.sampling_steps,
                    top_k=self.config.top_samples, time_limit_s=backend_limit,
                )
                sampler_backend = self.config.search_backend
            elif self.config.search_backend == "pbit" and representations & {"pdit", "mfc"}:
                from qihc.s2e.hybrid_sampler import PDitMFCSampler, TorchPDitMFCSampler
                sampler_class = TorchPDitMFCSampler if self.config.sampler == "torch" else PDitMFCSampler
                kwargs = dict(
                    num_chains=self.config.num_chains, steps=self.config.sampling_steps,
                    top_k=self.config.top_samples, seed=self.config.seed + 7919 * iteration,
                    semantic_penalty=self.config.semantic_penalty,
                    proposal_bias=self.config.proposal_bias,
                    time_limit_s=remaining_s,
                    block_updates=self.config.block_updates,
                    temperature_ladder=self.config.temperature_ladder,
                    normalize_objective=self.config.normalize_objective,
                )
                if sampler_class is TorchPDitMFCSampler:
                    kwargs["device"] = self.config.device
                categorical = sampler_class(**kwargs).solve(instance, incumbent, proposal)
                sampled_bits = np.zeros((len(categorical.assignments), problem.num_variables), dtype=np.int8)
                for sample_idx, assignment in enumerate(categorical.assignments):
                    for customer, route in assignment.items():
                        variable_idx = problem.model.index.get(("assign", customer, route))
                        if variable_idx is not None:
                            sampled_bits[sample_idx, variable_idx] = 1
                population_bits = np.zeros((len(categorical.population_assignments), problem.num_variables), dtype=np.int8)
                for sample_idx, assignment in enumerate(categorical.population_assignments):
                    for customer, route in assignment.items():
                        idx = problem.model.index.get(("assign", customer, route))
                        if idx is not None:
                            population_bits[sample_idx, idx] = 1
                sampled = PBitSampleBatch(sampled_bits, categorical.energies, categorical.elapsed_s,
                                          categorical.metadata, population_bits)
                sampler_backend = categorical.metadata["backend"]
            else:
                weight, field, _ = problem.model.to_ising()
                sampled = self._sampler(self.config.seed + 7919 * iteration, time_limit_s=remaining_s).solve(
                    weight, field, problem.initial_bits
                )
                sampler_backend = "ising-qubo"
            candidate_solution: RouteSolution | None = None
            candidate_result: VerificationResult | None = None
            feasible_samples = 0
            last_violations: list[dict] = []
            sample_bits = [problem.initial_bits, *list(sampled.bits)]
            feedback_feasible = []
            feedback_objectives = []
            decode_changes = 0
            for bits in sample_bits:
                try:
                    decoded = decode_assignment(instance, problem, bits, strict=self.config.strict_assignment_decode)
                    raw_locations = {c: r for r, route in enumerate(decoded.routes) for c in route}
                    if self.config.constraint_aware_decode:
                        from qihc.problems.cvrp.verifier import repair_supported_constraints
                        decoded = repair_supported_constraints(instance, decoded)
                    repaired_locations = {c: r for r, route in enumerate(decoded.routes) for c in route}
                    decode_changes += int(repaired_locations != raw_locations)
                    verified = verify_solution(instance, decoded)
                except ValueError as exc:
                    last_violations = [{"type": "decode", "message": str(exc)}]
                    feedback_feasible.append(False); feedback_objectives.append(float("inf"))
                    continue
                if not verified.feasible:
                    last_violations = verified.violations
                    feedback_feasible.append(False); feedback_objectives.append(float("inf"))
                    continue
                feedback_feasible.append(True); feedback_objectives.append(verified.objective)
                feasible_samples += 1
                if candidate_result is None or verified.objective < candidate_result.objective:
                    candidate_solution, candidate_result = decoded, verified

            before = incumbent_result.objective
            accepted = bool(candidate_result and candidate_result.objective < before - 1e-9)
            improvement = max(0.0, before - candidate_result.objective) if candidate_result else 0.0
            feedback_bits = np.asarray(sample_bits, dtype=np.int8)
            if self.config.feedback_mode != "legacy":
                population = sampled.population_bits if self.config.feedback_source == "population" else sampled.bits
                if population is None:
                    population = sampled.bits
                # Select chain indices before decoding, independently of their energies.
                population_rng = np.random.default_rng(self.config.seed + 13007 * iteration)
                indices = population_rng.permutation(len(population))[:self.config.feedback_population_limit]
                feedback_bits = np.zeros((len(indices), problem.num_variables), dtype=np.int8)
                feedback_feasible, feedback_objectives = [], []
                for row_idx, sample_idx in enumerate(indices):
                    if self.config.search_time_limit_s and time.perf_counter() - search_started >= self.config.search_time_limit_s:
                        feedback_feasible.extend([False] * (len(indices) - row_idx))
                        feedback_objectives.extend([float("inf")] * (len(indices) - row_idx))
                        break
                    try:
                        decoded = decode_assignment(instance, problem, population[sample_idx], strict=self.config.strict_assignment_decode)
                        if self.config.constraint_aware_decode:
                            from qihc.problems.cvrp.verifier import repair_supported_constraints
                            decoded = repair_supported_constraints(instance, decoded)
                        verified = verify_solution(instance, decoded)
                        locations = {c: r for r, route in enumerate(decoded.routes) for c in route}
                        in_domain = all(locations[c] in proposal.candidate_routes[c] for c in proposal.destroy_customers)
                        for customer in proposal.destroy_customers:
                            variable_idx = problem.model.index.get(("assign", customer, locations[customer]))
                            if variable_idx is not None:
                                feedback_bits[row_idx, variable_idx] = 1
                        feedback_feasible.append(verified.feasible and in_domain)
                        feedback_objectives.append(verified.objective if verified.feasible and in_domain else float("inf"))
                    except ValueError:
                        feedback_feasible.append(False); feedback_objectives.append(float("inf"))
            if self.config.shuffle_feedback_quality:
                mask = np.asarray(feedback_feasible, dtype=bool)
                valid_indices = np.flatnonzero(mask)
                scores = np.asarray(feedback_objectives, dtype=float)
                rng = np.random.default_rng(self.config.seed + 19001 * iteration)
                scores[valid_indices] = rng.permutation(scores[valid_indices])
                feedback_objectives = scores.tolist()
            logit_updates = (
                self.logit_feedback.observe(
                    proposal, problem.model.variables, feedback_bits, accepted, improvement,
                    feasible_mask=np.asarray(feedback_feasible, dtype=bool),
                    objectives=np.asarray(feedback_objectives, dtype=float),
                )
                if self.config.enable_logit_feedback else {}
            )
            teacher = {}
            if candidate_solution is not None:
                locations = {c: r for r, route in enumerate(candidate_solution.routes) for c in route}
                teacher = {
                    "destroy_customers": list(proposal.destroy_customers),
                    "candidate_routes": {str(c): list(proposal.candidate_routes[c]) for c in proposal.destroy_customers},
                    "candidate_route_logits": {
                        str(c): {str(r): (2 if r == locations[c] else -2) for r in proposal.candidate_routes[c]}
                        for c in proposal.destroy_customers
                    }, "confidence": 1.0,
                }
                marginals = getattr(self.logit_feedback, "diagnostics", {}).get("marginals", {})
                for customer in proposal.destroy_customers:
                    marginal = marginals.get(str(customer))
                    if marginal:
                        values = np.log(np.clip(np.asarray(marginal["target"]), 1e-8, 1.0))
                        values -= values.mean()
                        teacher["candidate_route_logits"][str(customer)] = {
                            str(route): int(np.clip(round(value), -2, 2))
                            for route, value in zip(marginal["routes"], values)
                        }
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
                    candidate_compression=candidate_compression,
                    violations=last_violations,
                    objective_improvement=improvement,
                    pbit_logit_updates=logit_updates,
                    proposal_payload=dict(proposal.raw),
                    sampler_backend=sampler_backend,
                    feedback_diagnostics=dict(getattr(self.logit_feedback, "diagnostics", {})),
                    teacher_payload=teacher,
                    incumbent_routes_before=before_routes,
                    incumbent_routes_after=[list(route) for route in incumbent.routes],
                    elapsed_s=time.perf_counter() - search_started,
                    sampler_metadata=dict(sampled.metadata),
                    sample_decode_diagnostics={"semantic_repair_changed_assignment": decode_changes,
                                               "strict_assignment_decode": self.config.strict_assignment_decode,
                                               "quality_shuffled": self.config.shuffle_feedback_quality},
                    energy_feedback_context=energy_feedback_context,
                )
            )
            if self.config.patience and stale >= self.config.patience:
                stop_reason = "patience"
                break
        incumbent.metadata = {**initial_metadata, **incumbent.metadata}
        result = LNSResult(
            instance_name=instance.name,
            solution=incumbent,
            verification=incumbent_result,
            initial_verification=initial_result,
            records=records,
            total_elapsed_s=time.perf_counter() - t0,
            config=asdict(self.config),
            construction_pbit_s=construction_pbit_s,
            construction_batches=construction_batches,
            search_elapsed_s=time.perf_counter() - search_started,
            stop_reason=stop_reason,
            initial_solution=initial_solution,
        )
        summary = result.to_summary()
        result.solution.metadata["search_elapsed_s"] = time.perf_counter() - search_started
        if instance.best_known_cost:
            summary["best_known_cost"] = instance.best_known_cost
            summary["optimality_gap"] = (
                result.verification.cost - instance.best_known_cost
            ) / instance.best_known_cost
        result.solution.metadata["summary"] = summary
        return result
