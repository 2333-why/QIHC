"""Versioned Constraint Program Package (CPP) intermediate representation."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any

from qihc.problems.cvrp.instance import ConstraintSpec


CPP_SCHEMA_VERSION = "qihc.cpp.v1"
SUPPORTED_CONSTRAINTS = {
    "same_resource", "same_vehicle", "mutual_exclusion", "precedence",
    "time_window", "preferred_time", "max_route_distance", "capacity",
}


@dataclass(frozen=True)
class ConstraintProgram:
    id: str
    type: str
    hard: bool
    weight: float
    params: dict[str, Any]
    source_text: str
    scope: str = "route"
    checker: str = "builtin"
    candidate_encodings: tuple[str, ...] = ("qubo", "pdit", "mfc")
    tests: tuple[dict[str, Any], ...] = ()
    confidence: float = 1.0

    @classmethod
    def from_spec(cls, spec: ConstraintSpec, index: int = 0) -> "ConstraintProgram":
        payload = json.dumps({"type": spec.type, "params": spec.params}, sort_keys=True, ensure_ascii=False)
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]
        return cls(
            id=f"c{index:03d}-{digest}", type=spec.type, hard=spec.hard,
            weight=spec.weight, params=dict(spec.params), source_text=spec.source_text,
            confidence=spec.confidence,
        )

    def to_spec(self) -> ConstraintSpec:
        return ConstraintSpec(self.type, self.hard, self.weight, dict(self.params), self.source_text, self.confidence)


@dataclass
class ConstraintProgramPackage:
    instance_name: str
    description: str
    customer_ids: list[int]
    programs: list[ConstraintProgram]
    schema_version: str = CPP_SCHEMA_VERSION
    generator: dict[str, Any] = field(default_factory=dict)
    repair_history: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def from_specs(cls, instance_name: str, description: str, customer_ids: list[int], specs: list[ConstraintSpec], generator: dict[str, Any] | None = None) -> "ConstraintProgramPackage":
        return cls(instance_name, description, list(customer_ids), [ConstraintProgram.from_spec(s, i) for i, s in enumerate(specs)], generator=generator or {})

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ConstraintProgramPackage":
        return cls(
            instance_name=str(value["instance_name"]), description=str(value.get("description", "")),
            customer_ids=[int(x) for x in value["customer_ids"]],
            programs=[ConstraintProgram(**{**p, "candidate_encodings": tuple(p.get("candidate_encodings", ("qubo", "pdit", "mfc"))), "tests": tuple(p.get("tests", ()))}) for p in value["programs"]],
            schema_version=str(value.get("schema_version", CPP_SCHEMA_VERSION)),
            generator=dict(value.get("generator", {})), repair_history=list(value.get("repair_history", [])),
        )

    def canonical_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, ensure_ascii=False, separators=(",", ":"))

    @property
    def checksum(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()

    def specs(self) -> list[ConstraintSpec]:
        return [program.to_spec() for program in self.programs]
