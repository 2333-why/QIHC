"""A0-A4 deterministic validation stack and counterexample production."""

from __future__ import annotations

import itertools
from dataclasses import asdict, dataclass, field
from enum import IntEnum
from typing import Any

from qihc.problems.cvrp.instance import CVRPInstance, RouteSolution
from qihc.problems.cvrp.verifier import verify_solution
from qihc.s2e.cpp import CPP_SCHEMA_VERSION, SUPPORTED_CONSTRAINTS, ConstraintProgram, ConstraintProgramPackage


class ValidationStage(IntEnum):
    A0_SCHEMA = 0
    A1_STATIC = 1
    A2_TESTS = 2
    A3_EXACT = 3
    A4_ENCODING = 4


@dataclass
class StageResult:
    stage: str
    passed: bool
    errors: list[dict[str, Any]] = field(default_factory=list)
    checked: int = 0


@dataclass
class CPPValidationReport:
    passed: bool
    highest_stage: str
    stages: list[StageResult]
    counterexamples: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"passed": self.passed, "highest_stage": self.highest_stage, "stages": [asdict(x) for x in self.stages], "counterexamples": self.counterexamples}


class CPPValidator:
    def __init__(self, exact_customer_limit: int = 8, max_exact_assignments: int = 20000):
        self.exact_customer_limit = exact_customer_limit
        self.max_exact_assignments = max_exact_assignments

    def _a0(self, cpp: ConstraintProgramPackage) -> StageResult:
        errors = []
        if cpp.schema_version != CPP_SCHEMA_VERSION: errors.append({"reason": "schema_version", "value": cpp.schema_version})
        ids = [p.id for p in cpp.programs]
        if len(ids) != len(set(ids)): errors.append({"reason": "duplicate_program_id"})
        for p in cpp.programs:
            if p.type not in SUPPORTED_CONSTRAINTS: errors.append({"program": p.id, "reason": "unsupported_type", "type": p.type})
            if not 0 <= p.confidence <= 1: errors.append({"program": p.id, "reason": "confidence_range"})
            if p.weight < 0: errors.append({"program": p.id, "reason": "negative_weight"})
        return StageResult("A0_SCHEMA", not errors, errors, len(cpp.programs))

    def _mentioned(self, p: ConstraintProgram) -> list[int]:
        if p.type in {"same_resource", "same_vehicle", "mutual_exclusion"}: return [int(x) for x in p.params.get("entities", [])]
        if p.type == "precedence": return [int(p.params.get("before", -1)), int(p.params.get("after", -1))]
        if p.type in {"time_window", "preferred_time"}: return [int(p.params.get("entity", p.params.get("customer", -1)))]
        return []

    def _a1(self, cpp: ConstraintProgramPackage) -> StageResult:
        valid, errors = set(cpp.customer_ids), []
        for p in cpp.programs:
            mentioned = self._mentioned(p)
            if mentioned and any(x not in valid for x in mentioned): errors.append({"program": p.id, "reason": "unknown_customer", "entities": mentioned})
            if p.type in {"same_resource", "same_vehicle", "mutual_exclusion"} and len(mentioned) < 2: errors.append({"program": p.id, "reason": "arity"})
            if p.type == "precedence" and (len(mentioned) != 2 or mentioned[0] == mentioned[1]): errors.append({"program": p.id, "reason": "precedence_domain"})
        return StageResult("A1_STATIC", not errors, errors, len(cpp.programs))

    def _a2(self, cpp: ConstraintProgramPackage) -> StageResult:
        errors, checked = [], 0
        for p in cpp.programs:
            for test in p.tests:
                checked += 1
                if "expected" not in test or "routes" not in test: errors.append({"program": p.id, "reason": "malformed_test", "test": test})
        return StageResult("A2_TESTS", not errors, errors, checked)

    def _assignment_satisfies(self, p: ConstraintProgram, assignment: dict[int, int]) -> bool:
        if p.type in {"same_resource", "same_vehicle"}:
            e = self._mentioned(p); return len(e) >= 2 and assignment[e[0]] == assignment[e[1]]
        if p.type == "mutual_exclusion":
            e = self._mentioned(p); return len(e) >= 2 and assignment[e[0]] != assignment[e[1]]
        return True

    def _a3(self, cpp: ConstraintProgramPackage, instance: CVRPInstance | None) -> tuple[StageResult, list[dict[str, Any]]]:
        if instance is None or len(instance.customer_ids) > self.exact_customer_limit:
            return StageResult("A3_EXACT", True, [], 0), []
        relevant = [p for p in cpp.programs if p.type in {"same_resource", "same_vehicle", "mutual_exclusion"}]
        checked, counterexamples = 0, []
        for values in itertools.product(range(instance.vehicle_count), repeat=len(instance.customer_ids)):
            checked += 1
            if checked > self.max_exact_assignments: break
            assignment = dict(zip(instance.customer_ids, values))
            for p in relevant:
                expected = self._assignment_satisfies(p, assignment)
                routes = [[c for c in instance.customer_ids if assignment[c] == k] for k in range(instance.vehicle_count)]
                isolated = CVRPInstance(instance.name, instance.depot, instance.customers, instance.vehicle_count, 10**9, [p.to_spec()])
                observed = verify_solution(isolated, RouteSolution(routes)).feasible
                if expected != observed:
                    counterexamples.append({"program": p.id, "assignment": assignment, "expected": expected, "observed": observed})
                    return StageResult("A3_EXACT", False, [{"reason": "checker_mismatch", "program": p.id}], checked), counterexamples
        return StageResult("A3_EXACT", True, [], checked), counterexamples

    def _a4(self, cpp: ConstraintProgramPackage) -> StageResult:
        errors = [{"program": p.id, "reason": "no_candidate_encoding"} for p in cpp.programs if not p.candidate_encodings]
        return StageResult("A4_ENCODING", not errors, errors, len(cpp.programs))

    def validate(self, cpp: ConstraintProgramPackage, instance: CVRPInstance | None = None, through: ValidationStage = ValidationStage.A4_ENCODING) -> CPPValidationReport:
        stages, counterexamples = [], []
        for idx in range(int(through) + 1):
            if idx == 0: result = self._a0(cpp)
            elif idx == 1: result = self._a1(cpp)
            elif idx == 2: result = self._a2(cpp)
            elif idx == 3:
                result, ces = self._a3(cpp, instance); counterexamples.extend(ces)
            else: result = self._a4(cpp)
            stages.append(result)
            if not result.passed: break
        return CPPValidationReport(all(x.passed for x in stages), stages[-1].stage if stages else "NONE", stages, counterexamples)
