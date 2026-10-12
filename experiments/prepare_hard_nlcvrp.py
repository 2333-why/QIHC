#!/usr/bin/env python3
"""Build harder NL-CVRP cases from public .vrp/.sol pairs without route leakage.

The published solution is used *only* as a feasibility witness for generating
natural-language constraints. It is never stored in the output JSONL.
"""

from __future__ import annotations

import argparse
import hashlib
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


def prepare_cases(input_dir: Path, output_jsonl: Path, *, min_customers=200, max_customers=1000,
                  limit=20, clauses_per_type=2, seed=2026, selection="smallest", input_manifest: Path | None = None) -> dict:
    """Validate the whole declared pool, then select without relaxing the fleet."""
    if limit < 0 or clauses_per_type < 1 or selection not in {"smallest", "spread"}:
        raise ValueError("Invalid case count, clause count or selection")
    eligible, skipped, pool_checksums, primary_names, qualified = [], [], {}, [], {}
    def size_key(path: Path) -> tuple[int, str]:
        match = re.search(r"-n(\d+)-k\d+", path.stem, re.IGNORECASE)
        return (int(match.group(1)) - 1 if match else -1, path.stem)

    if input_manifest:
        entries = json.loads(input_manifest.read_text(encoding="utf-8"))
        names = [row["name"] for row in entries]
        primary_names = [row["name"] for row in entries if row.get("candidate_role") == "primary"]
        if len(names) != len(set(names)) or any(not re.fullmatch(r"X-n\d+-k\d+", name) for name in names):
            raise ValueError("Invalid/duplicate candidate name in download manifest")
        paths = [input_dir / f"{name}.vrp" for name in names]
        pool_checksums = {row["name"]: row.get("files", {}) for row in entries}
    else:
        paths = list(input_dir.rglob("*.vrp"))
    paths = sorted([p for p in paths if min_customers <= size_key(p)[0] <= max_customers],
                   key=lambda p: (0 if p.stem in primary_names else 1, size_key(p)))
    for path in paths:
        solution_path = path.with_suffix(".sol")
        if not solution_path.exists():
            skipped.append({"name": path.stem, "reason": "matching .sol missing"})
            continue
        try:
            for suffix, expected in pool_checksums.get(path.stem, {}).items():
                if suffix not in {"vrp", "sol"}:
                    raise ValueError("Unexpected file suffix in download manifest")
                if expected.get("sha256") and hashlib.sha256(path.with_suffix('.' + suffix).read_bytes()).hexdigest() != expected["sha256"]:
                    raise ValueError(f"download manifest checksum mismatch: {suffix}")
            instance = load_cvrplib(path)
            if not min_customers <= len(instance.customers) <= max_customers:
                continue
            witness = align_solution_ids(instance, read_solution(solution_path))
            verification = verify_solution(instance, witness)
            if not verification.feasible:
                skipped.append({"name": path.stem, "reason": "published solution failed the local verifier",
                                "violations": verification.violations,
                                "vehicle_limit": instance.vehicle_count, "witness_routes": len(witness.routes)})
                continue
            if instance.best_known_cost is None:
                instance.best_known_cost = witness.metadata["reported_cost"]
            prepared = add_constraints(instance, witness, seed + len(eligible), clauses_per_type)
            eligible.append(prepared)
            qualified[path.stem] = prepared
        except (OSError, ValueError) as exc:
            skipped.append({"name": path.stem, "reason": str(exc)})
    replacements = []
    if primary_names and limit and len(primary_names) == limit:
        selected = [qualified[name] for name in primary_names if name in qualified]
        reserve_names = [name for name in qualified if name not in primary_names]
        for failed_name in primary_names:
            if failed_name in qualified or not reserve_names:
                continue
            target_size = size_key(input_dir / f"{failed_name}.vrp")[0]
            chosen_name = min(reserve_names, key=lambda name: (
                abs(len(qualified[name].customers) - target_size), size_key(input_dir / f"{name}.vrp")))
            reserve_names.remove(chosen_name)
            selected.append(qualified[chosen_name])
            replacements.append({"excluded_primary": failed_name, "replacement": chosen_name})
        selected.sort(key=lambda instance: (len(instance.customers), instance.name))
    elif limit and len(eligible) > limit and selection == "spread":
        indices = [0] if limit == 1 else [round(i * (len(eligible) - 1) / (limit - 1)) for i in range(limit)]
        selected = [eligible[index] for index in indices]
    else:
        selected = eligible[:limit or None]
    save_jsonl(selected, output_jsonl)
    manifest = {"selected": [x.name for x in selected], "skipped": skipped,
                "min_customers": min_customers, "max_customers": max_customers,
                "seed": seed, "witness_routes_exported": False, "selection": selection,
                "selection_stage": "after_reference_and_generated_constraint_verification",
                "vehicle_policy": "preserve_original_limit", "requested": limit,
                "candidate_pool": [p.stem for p in paths], "valid_candidates": [x.name for x in eligible],
                "primary_candidates": primary_names, "replacements": replacements,
                "input_manifest": str(input_manifest) if input_manifest else None,
                "input_manifest_sha256": hashlib.sha256(input_manifest.read_bytes()).hexdigest() if input_manifest else None}
    output_jsonl.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"written": len(selected), "skipped": len(skipped)}, ensure_ascii=False))
    if limit and len(selected) != limit:
        raise RuntimeError(f"Requested {limit} benchmark cases but prepared only {len(selected)}; enlarge candidate pool; inspect {output_jsonl.with_suffix('.manifest.json')}")
    if not selected:
        raise RuntimeError("No verified benchmark cases")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("output_jsonl", type=Path)
    parser.add_argument("--min-customers", type=int, default=200)
    parser.add_argument("--max-customers", type=int, default=1000)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--clauses-per-type", type=int, default=2)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--selection", choices=["smallest", "spread"], default="smallest")
    parser.add_argument("--input-manifest", type=Path, help="Only use the verified downloader's declared candidate pool")
    args = parser.parse_args()
    prepare_cases(args.input_dir, args.output_jsonl, min_customers=args.min_customers,
                  max_customers=args.max_customers, limit=args.limit, clauses_per_type=args.clauses_per_type,
                  seed=args.seed, selection=args.selection, input_manifest=args.input_manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
