#!/usr/bin/env python3
"""Build harder NL-CVRP cases from public .vrp/.sol pairs without route leakage.

The published solution is used *only* as a feasibility witness for generating
natural-language constraints. It is never stored in the output JSONL.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qihc.problems.cvrp.instance import ConstraintSpec, RouteSolution, load_cvrplib, save_jsonl
from qihc.problems.cvrp.verifier import verify_solution


def read_solution(path: Path) -> RouteSolution:
    routes = []
    reported_cost = None
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = re.match(r"\s*Route\s*#?\d+\s*:\s*(.*)", line, re.IGNORECASE)
        if match:
            routes.append([int(token) for token in match.group(1).split()])
        cost_match = re.match(r"\s*Cost\s*[:=]?\s*([0-9]+(?:\.[0-9]+)?)", line, re.IGNORECASE)
        if cost_match:
            reported_cost = float(cost_match.group(1))
    if not routes:
        raise ValueError(f"No CVRPLIB Route lines in {path}")
    return RouteSolution(routes, source="witness_not_exported", metadata={"reported_cost": reported_cost})


def align_solution_ids(instance, witness: RouteSolution) -> RouteSolution:
    """CVRPLIB .sol routes commonly number customers 1..n-1, not VRP node IDs."""
    raw = {c for route in witness.routes for c in route}
    expected = set(instance.customer_ids)
    for offset in (0, 1, -1):
        if {c + offset for c in raw} == expected:
            return RouteSolution(
                [[c + offset for c in route] for route in witness.routes],
                source=witness.source,
                metadata={**witness.metadata, "solution_id_offset": offset},
            )
    raise ValueError("published solution customer IDs do not match the instance")


def add_constraints(instance, witness: RouteSolution, seed: int, count: int):
    rng = random.Random(seed)
    routes = [route for route in witness.routes if len(route) >= 3]
    if len(routes) < 2:
        raise ValueError("Need at least two nontrivial witness routes")
    rng.shuffle(routes)
    chosen = []
    used = set()
    for route in routes:
        available = [c for c in route if c not in used]
        if len(available) >= 2:
            a, b = rng.sample(available, 2)
            chosen.append(ConstraintSpec("same_resource", params={"entities": [a, b]},
                source_text=f"客户 {a} 与客户 {b} 必须由同一辆车配送。"))
            used.update((a, b))
            if len(chosen) >= count:
                break
    for route in routes:
        available = [c for c in route if c not in used]
        if len(available) >= 2:
            a, b = sorted(rng.sample(available, 2), key=route.index)
            chosen.append(ConstraintSpec("precedence", params={"before": a, "after": b},
                source_text=f"客户 {a} 必须先于客户 {b} 完成配送，且两者在同一辆车上。"))
            used.update((a, b))
            if len(chosen) >= 2 * count:
                break
    for i, route in enumerate(routes):
        other = routes[(i + 1) % len(routes)]
        left = next((c for c in route if c not in used), None)
        right = next((c for c in other if c not in used), None)
        if left is not None and right is not None:
            chosen.append(ConstraintSpec("mutual_exclusion", params={"entities": [left, right]},
                source_text=f"客户 {left} 与客户 {right} 不得由同一辆车配送。"))
            used.update((left, right))
            if len(chosen) >= 3 * count:
                break
    if len(chosen) != 3 * count:
        raise ValueError("Insufficient disjoint witness customers for requested clauses")
    instance.constraints = chosen
    instance.description = " ".join(spec.source_text for spec in chosen)
    instance.name += "-nl-hard"
    instance.metadata.pop("known_feasible_routes", None)
    instance.metadata["constraint_generation"] = "published_solution_witness_not_exported"
    if not verify_solution(instance, witness).feasible:
        raise ValueError("Generated constraints do not admit the witness")
    return instance


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("output_jsonl", type=Path)
    parser.add_argument("--min-customers", type=int, default=200)
    parser.add_argument("--max-customers", type=int, default=1000)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--clauses-per-type", type=int, default=2)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    selected, skipped = [], []
    for path in sorted(args.input_dir.rglob("*.vrp")):
        if args.limit and len(selected) >= args.limit:
            break
        solution_path = path.with_suffix(".sol")
        if not solution_path.exists():
            skipped.append({"name": path.stem, "reason": "matching .sol missing"})
            continue
        try:
            instance = load_cvrplib(path)
            if not args.min_customers <= len(instance.customers) <= args.max_customers:
                continue
            witness = align_solution_ids(instance, read_solution(solution_path))
            if not verify_solution(instance, witness).feasible:
                raise ValueError("published solution failed the local verifier")
            if instance.best_known_cost is None:
                instance.best_known_cost = witness.metadata["reported_cost"]
            selected.append(add_constraints(instance, witness, args.seed + len(selected), args.clauses_per_type))
        except (OSError, ValueError) as exc:
            skipped.append({"name": path.stem, "reason": str(exc)})
    save_jsonl(selected, args.output_jsonl)
    manifest = {"selected": [x.name for x in selected], "skipped": skipped,
                "min_customers": args.min_customers, "max_customers": args.max_customers,
                "seed": args.seed, "witness_routes_exported": False}
    args.output_jsonl.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"written": len(selected), "skipped": len(skipped)}, ensure_ascii=False))
    return 0 if selected else 1


if __name__ == "__main__":
    raise SystemExit(main())
