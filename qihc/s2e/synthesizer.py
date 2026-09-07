"""Constraint synthesis backends with deterministic structured fallback."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Protocol

from qihc.problems.cvrp.instance import ConstraintSpec, CVRPInstance
from qihc.s2e.cpp import ConstraintProgramPackage


class ConstraintBackend(Protocol):
    def generate(self, description: str, customer_ids: list[int]) -> tuple[list[ConstraintSpec], dict]: ...


@dataclass
class HeuristicConstraintBackend:
    """Offline deterministic parser used for tests and as a fail-closed baseline."""

    def generate(self, description: str, customer_ids: list[int]) -> tuple[list[ConstraintSpec], dict]:
        valid = set(customer_ids)
        specs: list[ConstraintSpec] = []
        clauses = [x.strip() for x in re.split(r"[。；;\n]+", description) if x.strip()]
        for clause in clauses:
            ids = [int(x) for x in re.findall(r"(?:客户|订单)?\s*(\d+)", clause) if int(x) in valid]
            spec = None
            if len(ids) >= 2 and any(k in clause for k in ("不能由同一", "不得同车", "不能同车", "different vehicle")):
                spec = ConstraintSpec("mutual_exclusion", params={"entities": ids[:2]}, source_text=clause)
            elif len(ids) >= 2 and any(k in clause for k in ("同一辆车", "同车", "same vehicle")):
                spec = ConstraintSpec("same_resource", params={"entities": ids[:2]}, source_text=clause)
            elif len(ids) >= 2 and any(k in clause for k in ("先于", "之前", "before")):
                spec = ConstraintSpec("precedence", params={"before": ids[0], "after": ids[1]}, source_text=clause)
            if spec:
                specs.append(spec)
        return specs, {"backend": "heuristic", "clauses": clauses}


class ConstraintSynthesizer:
    def __init__(self, backend: ConstraintBackend, max_repairs: int = 2):
        self.backend = backend
        self.max_repairs = max_repairs

    def synthesize(self, instance: CVRPInstance) -> ConstraintProgramPackage:
        specs, audit = self.backend.generate(instance.description, instance.customer_ids)
        return ConstraintProgramPackage.from_specs(instance.name, instance.description, instance.customer_ids, specs, audit)

    def synthesize_verified(self, instance: CVRPInstance, validator):
        from qihc.s2e.repair import repair_cpp
        cpp = self.synthesize(instance)
        reports = []
        for _ in range(self.max_repairs + 1):
            report = validator.validate(cpp, instance); reports.append(report)
            if report.passed: return cpp, reports
            cpp = repair_cpp(cpp, report)
        return cpp, reports


class LocalLLMConstraintBackend:
    """Adapter around the existing local-only Transformers frontend."""

    def __init__(self, model_path: str, device: str = "cuda:0", temperature: float = 0.0):
        from qihc.problems.cvrp.llm_selector import LocalLLMNeighborhoodSelector
        self.frontend = LocalLLMNeighborhoodSelector(model_path, destroy_size=1, routes_per_customer=1, device=device, temperature=temperature)

    def generate(self, description: str, customer_ids: list[int]) -> tuple[list[ConstraintSpec], dict]:
        return self.frontend.parse_constraint_ir(description, customer_ids)
