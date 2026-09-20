"""Multi-representation compiler planning for QUBO, p-dit and MFC."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

from qihc.problems.cvrp.instance import CVRPInstance
from qihc.s2e.cpp import ConstraintProgramPackage
from qihc.s2e.validator import CPPValidationReport


class Representation(str, Enum):
    QUBO = "qubo"
    PDIT = "pdit"
    MFC = "mfc"


@dataclass
class RepresentationCost:
    representation: Representation
    auxiliary_variables: float
    graph_density: float
    log_dynamic_range: float
    predicted_solve_cost: float
    predicted_violation: float

    @property
    def total(self) -> float:
        return self.auxiliary_variables + self.graph_density + self.log_dynamic_range + self.predicted_solve_cost + 2.0 * self.predicted_violation


@dataclass
class ConstraintCompilation:
    program_id: str
    constraint_type: str
    representation: Representation
    scores: list[RepresentationCost]
    rationale: str
    execution: str = "verifier_only"


@dataclass
class CompilationPlan:
    cpp_checksum: str
    constraints: list[ConstraintCompilation]
    estimated_binary_variables: int
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"cpp_checksum": self.cpp_checksum, "constraints": [{**asdict(c), "representation": c.representation.value, "scores": [{**asdict(s), "representation": s.representation.value} for s in c.scores]} for c in self.constraints], "estimated_binary_variables": self.estimated_binary_variables, "metadata": self.metadata}


class RepresentationSelector:
    """Transparent cost model; coefficients can later be replaced by a learned model."""

    def score(self, kind: str, representation: Representation, instance: CVRPInstance) -> RepresentationCost:
        n, k = len(instance.customer_ids), instance.vehicle_count
        multi = kind in {"same_resource", "same_vehicle", "mutual_exclusion", "capacity"}
        global_c = kind in {"capacity", "max_route_distance", "precedence"}
        if representation == Representation.QUBO:
            aux = float(k if kind == "capacity" else 0); density = min(1.0, (k / max(n, 1)) if multi else 0.2); dynamic = 2.0 if global_c else 0.7; solve = 0.4 + density; violation = 0.18 if global_c else 0.05
        elif representation == Representation.PDIT:
            aux = 0.0; density = 0.25 if multi else 0.6; dynamic = 0.4; solve = 0.55; violation = 0.07 if multi else 0.2
        else:
            aux = 0.0; density = 0.1; dynamic = 0.25; solve = 0.8 if global_c else 1.2; violation = 0.06 if global_c else 0.14
        return RepresentationCost(representation, aux, density, dynamic, solve, violation)

    def choose(self, kind: str, allowed: tuple[str, ...], instance: CVRPInstance) -> tuple[Representation, list[RepresentationCost]]:
        candidates = [Representation(x) for x in allowed if x in {r.value for r in Representation}]
        if not candidates: raise ValueError(f"No supported representation for {kind}")
        scores = [self.score(kind, r, instance) for r in candidates]
        selected = min(scores, key=lambda item: (item.total, item.representation.value))
        return selected.representation, scores


def compile_cpp(cpp: ConstraintProgramPackage, instance: CVRPInstance, validation: CPPValidationReport, selector: RepresentationSelector | None = None) -> CompilationPlan:
    if not validation.passed or validation.highest_stage != "A4_ENCODING":
        raise ValueError("CPP must pass through A4_ENCODING before compilation")
    selector = selector or RepresentationSelector()
    compiled = []
    for p in cpp.programs:
        chosen, scores = selector.choose(p.type, p.candidate_encodings, instance)
        if p.type in {"same_resource", "same_vehicle", "mutual_exclusion"}:
            execution = "qubo_pair_penalty" if chosen == Representation.QUBO else "categorical_pair_penalty"
        elif p.type == "precedence":
            execution = "route_colocation_penalty_plus_order_verifier"
        elif p.type == "capacity":
            execution = "qubo_capacity_penalty" if chosen == Representation.QUBO else "multiplier_capacity_penalty"
        else:
            # These constraints are enforced by the exact solution verifier,
            # but are not yet lowered into a local assignment energy.
            execution = "verifier_only"
        compiled.append(ConstraintCompilation(p.id, p.type, chosen, scores, f"minimum transparent cost={min(x.total for x in scores):.4f}", execution))
    counts = {r.value: sum(c.representation == r for c in compiled) for r in Representation}
    return CompilationPlan(cpp.checksum, compiled, len(instance.customer_ids) * instance.vehicle_count, {
        "representation_counts": counts, "selector": "transparent-v1",
        "verifier_only_types": [c.constraint_type for c in compiled if c.execution == "verifier_only"],
    })
