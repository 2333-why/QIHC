#!/usr/bin/env python3
"""Prepare grouped train/validation/test splits, including size-transfer tests."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.prepare_hard_nlcvrp import add_constraints
from qihc.problems.cvrp.instance import RouteSolution, generate_synthetic_instance, load_jsonl, save_jsonl
from qihc.problems.cvrp.verifier import greedy_initial_solution
from qihc.s2e.cpp import ConstraintProgramPackage


def geometry_key(instance) -> str:
    # Different constraint variants/paraphrases of one numerical instance stay together.
    value = [
        [instance.depot.x, instance.depot.y],
        sorted([c.x, c.y, c.demand] for c in instance.customers.values()),
    ]
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def prepare(output: Path, inputs: list[Path], sizes: list[int], seeds: int, clauses: int) -> dict:
    instances = []
    for size in sizes:
        for seed in range(seeds):
            base = generate_synthetic_instance(size, max(2, (size + 7) // 8), 60, seed + 1000003 * size, semantic_constraints=False)
            routes = [[] for _ in range(base.vehicle_count)]
            loads = [0] * base.vehicle_count
            for customer in sorted(base.customer_ids, key=lambda c: -base.customers[c].demand):
                feasible = [r for r in range(base.vehicle_count) if loads[r] + base.customers[customer].demand <= base.vehicle_capacity]
                if not feasible:
                    raise ValueError("Synthetic witness packing failed")
                route = min(feasible, key=lambda r: (loads[r], len(routes[r]), r))
                routes[route].append(customer); loads[route] += base.customers[customer].demand
            witness = RouteSolution(routes)
            instance = add_constraints(base, witness, 2026 + seed, clauses)
            instance.name = f"research-n{size}-s{seed}"
            instance.metadata["research_source"] = "synthetic_witness_constrained"
            instances.append(instance)
    for source in inputs:
        instances.extend(load_jsonl(source))
    by_name = {}
    for instance in instances:
        if instance.name in by_name:
            if instance.to_dict() != by_name[instance.name].to_dict():
                raise ValueError(f"Conflicting duplicate instance: {instance.name}")
            continue
        instance.metadata.pop("known_feasible_routes", None)
        by_name[instance.name] = instance
    groups = {}
    for instance in by_name.values():
        groups.setdefault(geometry_key(instance), []).append(instance)
    small = sorted(key for key, values in groups.items() if len(values[0].customers) <= max(sizes))
    if len(small) < 3:
        raise ValueError("At least three distinct small numerical instances are required")
    split_for = {key: "train" for key in small}
    for size in sorted({len(groups[key][0].customers) for key in small}):
        size_keys = [key for key in small if len(groups[key][0].customers) == size]
        if len(size_keys) < 3:
            raise ValueError(f"At least three distinct numerical instances required at size={size}")
        validation_n = max(1, len(size_keys) // 6)
        test_n = max(1, len(size_keys) // 6)
        split_for.update({key: "validation" for key in size_keys[:validation_n]})
        split_for.update({key: "test" for key in size_keys[validation_n:validation_n + test_n]})
    for key, values in groups.items():
        if key not in split_for:
            split_for[key] = "test" if len(values[0].customers) >= 400 else "validation"
    splits = {name: [] for name in ("train", "validation", "test")}
    for key, values in groups.items():
        splits[split_for[key]].extend(values)
    output.mkdir(parents=True, exist_ok=True)
    for name, values in splits.items():
        save_jsonl(values, output / f"{name}.jsonl")
    variants = []
    for instance in splits["test"]:
        for language in ("zh", "en"):
            variant = copy.deepcopy(instance)
            clauses = []
            for spec in variant.constraints:
                if spec.type in {"same_resource", "same_vehicle", "mutual_exclusion"}:
                    a, b = spec.params["entities"][:2]
                    if spec.type in {"same_resource", "same_vehicle"}:
                        clause = (f"配送客户{a}与客户{b}时，必须安排同一台车负责。" if language == "zh"
                                  else f"Customers {a} and {b} must be assigned to the same vehicle.")
                    else:
                        clause = (f"不要把客户{a}、客户{b}安排在同一台车上。" if language == "zh"
                                  else f"Customers {a} and {b} must never share a vehicle.")
                elif spec.type == "precedence":
                    a, b = spec.params["before"], spec.params["after"]
                    clause = (f"客户{b}的服务必须在客户{a}之后完成；这两位客户由同一车辆负责。" if language == "zh"
                              else f"Serve customer {b} only after customer {a}, and assign both to the same vehicle.")
                else:
                    clause = spec.source_text or instance.description
                if not spec.hard:
                    # Avoid silently strengthening soft preferences during a paraphrase.
                    clause = spec.source_text or instance.description
                clauses.append(clause)
            variant.name += f"-paraphrase-{language}"
            variant.description = " ".join(clauses)
            variant.metadata["semantic_parent"] = instance.name
            variants.append(variant)
    save_jsonl(variants, output / "semantic_test.jsonl")
    save_jsonl(by_name.values(), output / "all.jsonl")
    with (output / "oracle_cpp.jsonl").open("w", encoding="utf-8") as handle:
        for instance in by_name.values():
            cpp = ConstraintProgramPackage.from_specs(
                instance.name, instance.description, instance.customer_ids, instance.constraints
            )
            handle.write(json.dumps({"instance": instance.name, "status": "ok", "cpp": cpp.to_dict()}, ensure_ascii=False) + "\n")
    summary = {
        "splits": {name: [x.name for x in values] for name, values in splits.items()},
        "geometry_groups": {key: {"split": split_for[key], "instances": [x.name for x in values]}
                            for key, values in groups.items()},
        "size_transfer_threshold": max(sizes),
        "large_test_threshold": 400,
        "witness_routes_exported": False,
        "protocol": "small-instance training; disjoint-geometry validation; held-out small plus large test",
    }
    (output / "split_manifest.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, nargs="*", default=[])
    parser.add_argument("--sizes", type=int, nargs="+", default=[20, 50, 100])
    parser.add_argument("--instances-per-size", type=int, default=40)
    parser.add_argument("--clauses-per-type", type=int, default=2)
    args = parser.parse_args()
    summary = prepare(args.output, args.inputs, args.sizes, args.instances_per_size, args.clauses_per_type)
    print(json.dumps({name: len(values) for name, values in summary["splits"].items()}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
