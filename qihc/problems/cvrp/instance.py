"""Data model and loaders for CVRP/NL-CVRP instances."""

from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np


@dataclass(frozen=True)
class Customer:
    id: int
    x: float
    y: float
    demand: int
    ready_time: float = 0.0
    due_time: float = math.inf
    service_time: float = 0.0
    priority: float = 0.0
    text: str = ""


@dataclass(frozen=True)
class ConstraintSpec:
    """Normalized constraint emitted by an LLM or supplied by a dataset."""

    type: str
    hard: bool = True
    weight: float = 1.0
    params: dict[str, Any] = field(default_factory=dict)
    source_text: str = ""
    confidence: float = 1.0

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ConstraintSpec":
        known = {"type", "hard", "weight", "params", "source_text", "confidence"}
        params = dict(value.get("params", {}))
        params.update({k: v for k, v in value.items() if k not in known})
        return cls(
            type=str(value["type"]),
            hard=bool(value.get("hard", True)),
            weight=float(value.get("weight", 1.0)),
            params=params,
            source_text=str(value.get("source_text", "")),
            confidence=float(value.get("confidence", 1.0)),
        )


@dataclass
class CVRPInstance:
    name: str
    depot: Customer
    customers: dict[int, Customer]
    vehicle_count: int
    vehicle_capacity: int
    constraints: list[ConstraintSpec] = field(default_factory=list)
    description: str = ""
    best_known_cost: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.depot.id in self.customers:
            self.customers = dict(self.customers)
            self.customers.pop(self.depot.id)
        if self.vehicle_count <= 0:
            raise ValueError("vehicle_count must be positive")
        if self.vehicle_capacity <= 0:
            raise ValueError("vehicle_capacity must be positive")
        if any(c.demand < 0 for c in self.customers.values()):
            raise ValueError("customer demand cannot be negative")

    @property
    def customer_ids(self) -> list[int]:
        return sorted(self.customers)

    def point(self, customer_id: int) -> Customer:
        return self.depot if customer_id == self.depot.id else self.customers[customer_id]

    def distance(self, left: int, right: int) -> float:
        a, b = self.point(left), self.point(right)
        distance = float(math.hypot(a.x - b.x, a.y - b.y))
        edge_type = str(self.metadata.get("edge_weight_type", "CONTINUOUS")).upper()
        if edge_type == "EUC_2D":
            return float(math.floor(distance + 0.5))
        if edge_type == "CEIL_2D":
            return float(math.ceil(distance))
        return distance

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "depot": asdict(self.depot),
            "customers": [asdict(self.customers[i]) for i in self.customer_ids],
            "vehicle_count": self.vehicle_count,
            "vehicle_capacity": self.vehicle_capacity,
            "constraints": [asdict(c) for c in self.constraints],
            "description": self.description,
            "best_known_cost": self.best_known_cost,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "CVRPInstance":
        depot = Customer(**value["depot"])
        customers = {int(c["id"]): Customer(**c) for c in value["customers"]}
        constraints = [ConstraintSpec.from_dict(c) for c in value.get("constraints", [])]
        return cls(
            name=str(value["name"]),
            depot=depot,
            customers=customers,
            vehicle_count=int(value["vehicle_count"]),
            vehicle_capacity=int(value["vehicle_capacity"]),
            constraints=constraints,
            description=str(value.get("description", "")),
            best_known_cost=value.get("best_known_cost"),
            metadata=dict(value.get("metadata", {})),
        )

    @classmethod
    def load_json(cls, path: str | Path) -> "CVRPInstance":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


@dataclass
class RouteSolution:
    routes: list[list[int]]
    source: str = "unknown"
    metadata: dict[str, Any] = field(default_factory=dict)

    def copy(self) -> "RouteSolution":
        return RouteSolution([list(route) for route in self.routes], self.source, dict(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        return {"routes": self.routes, "source": self.source, "metadata": self.metadata}


def _parse_header_value(line: str) -> tuple[str, str] | None:
    if ":" not in line:
        return None
    key, value = line.split(":", 1)
    return key.strip().upper(), value.strip()


def load_cvrplib(path: str | Path, best_known_cost: float | None = None) -> CVRPInstance:
    """Load common TSPLIB/CVRPLIB ``.vrp`` files with EUC_2D coordinates."""

    source = Path(path)
    lines = source.read_text(encoding="utf-8", errors="replace").splitlines()
    headers: dict[str, str] = {}
    coords: dict[int, tuple[float, float]] = {}
    demands: dict[int, int] = {}
    depot_ids: list[int] = []
    section: str | None = None

    for raw in lines:
        line = raw.strip()
        if not line or line.upper() == "EOF":
            continue
        upper = line.upper()
        if upper in {"NODE_COORD_SECTION", "DEMAND_SECTION", "DEPOT_SECTION"}:
            section = upper
            continue
        header = _parse_header_value(line) if section is None else None
        if header:
            headers[header[0]] = header[1]
            continue
        parts = line.split()
        if section == "NODE_COORD_SECTION" and len(parts) >= 3:
            coords[int(parts[0])] = (float(parts[1]), float(parts[2]))
        elif section == "DEMAND_SECTION" and len(parts) >= 2:
            demands[int(parts[0])] = int(float(parts[1]))
        elif section == "DEPOT_SECTION" and parts:
            node = int(parts[0])
            if node >= 0:
                depot_ids.append(node)

    if not coords or not demands:
        raise ValueError(f"Missing NODE_COORD_SECTION or DEMAND_SECTION in {source}")
    depot_id = depot_ids[0] if depot_ids else min(coords)
    if depot_id not in coords:
        raise ValueError(f"Depot {depot_id} has no coordinates")
    capacity = int(headers.get("CAPACITY", "0"))
    name = headers.get("NAME", source.stem)
    if best_known_cost is None:
        comment = headers.get("COMMENT", "")
        bks_match = re.search(
            r"(?:optimal\s+value|best\s+known(?:\s+value|\s+solution)?|bks)\s*[:=]?\s*([0-9]+(?:\.[0-9]+)?)",
            comment,
            re.IGNORECASE,
        )
        if bks_match:
            best_known_cost = float(bks_match.group(1))
    match = re.search(r"-k(\d+)", name, re.IGNORECASE)
    vehicle_count = int(match.group(1)) if match else 0
    if vehicle_count <= 0:
        comment = headers.get("COMMENT", "")
        match = re.search(r"(?:trucks?|vehicles?)\s*[:=]?\s*(\d+)", comment, re.IGNORECASE)
        vehicle_count = int(match.group(1)) if match else max(1, math.ceil(sum(demands.values()) / capacity))

    depot_xy = coords[depot_id]
    depot = Customer(depot_id, depot_xy[0], depot_xy[1], 0)
    customers = {
        node: Customer(node, xy[0], xy[1], int(demands.get(node, 0)))
        for node, xy in coords.items()
        if node != depot_id
    }
    return CVRPInstance(
        name=name,
        depot=depot,
        customers=customers,
        vehicle_count=vehicle_count,
        vehicle_capacity=capacity,
        best_known_cost=best_known_cost,
        metadata={
            "source": str(source),
            "headers": headers,
            "edge_weight_type": headers.get("EDGE_WEIGHT_TYPE", "EUC_2D"),
        },
    )


def generate_synthetic_instance(
    n_customers: int = 40,
    vehicle_count: int = 6,
    vehicle_capacity: int = 50,
    seed: int = 0,
    semantic_constraints: bool = True,
) -> CVRPInstance:
    """Create deterministic offline CVRP instances for smoke and scaling tests."""

    rng = np.random.default_rng(seed)
    coords = rng.uniform(0.0, 100.0, size=(n_customers, 2))
    max_total = max(vehicle_count * vehicle_capacity - n_customers, n_customers)
    max_demand = max(2, min(10, max_total // max(n_customers, 1)))
    demands = rng.integers(1, max_demand + 1, size=n_customers)
    while int(demands.sum()) > vehicle_count * vehicle_capacity:
        idx = int(np.argmax(demands))
        demands[idx] = max(1, demands[idx] - 1)
    customers = {
        i + 1: Customer(i + 1, float(coords[i, 0]), float(coords[i, 1]), int(demands[i]))
        for i in range(n_customers)
    }
    constraints: list[ConstraintSpec] = []
    if semantic_constraints and n_customers >= 6:
        constraints = [
            ConstraintSpec("same_resource", params={"entities": [1, 2]}, source_text="客户1和2必须同车"),
            ConstraintSpec("mutual_exclusion", params={"entities": [3, 4]}, source_text="客户3和4不能同车"),
            ConstraintSpec("precedence", params={"before": 5, "after": 6}, source_text="客户5先于客户6"),
        ]
    return CVRPInstance(
        name=f"synthetic-n{n_customers}-k{vehicle_count}-s{seed}",
        depot=Customer(0, 50.0, 50.0, 0),
        customers=customers,
        vehicle_count=vehicle_count,
        vehicle_capacity=vehicle_capacity,
        constraints=constraints,
        description="Synthetic NL-CVRP with capacity and normalized semantic constraints.",
        metadata={"seed": seed, "generator": "qihc.synthetic.v1"},
    )


def load_jsonl(path: str | Path) -> list[CVRPInstance]:
    instances: list[CVRPInstance] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                instances.append(CVRPInstance.from_dict(json.loads(line)))
    return instances


def save_jsonl(instances: Iterable[CVRPInstance], path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for instance in instances:
            handle.write(json.dumps(instance.to_dict(), ensure_ascii=False) + "\n")
