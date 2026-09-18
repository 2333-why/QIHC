#!/usr/bin/env python3
"""Create a controlled natural-language CVRP benchmark from CVRPLIB instances."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from qihc.problems.cvrp.instance import ConstraintSpec, load_cvrplib, save_jsonl  # noqa: E402
from qihc.problems.cvrp.verifier import greedy_initial_solution, verify_solution  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("output_jsonl", type=Path)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    instances = []
    skipped = []
    for path in sorted(args.input_dir.rglob("*.vrp")):
        if args.limit and len(instances) >= args.limit:
            break
        try:
            instance = load_cvrplib(path)
            incumbent = greedy_initial_solution(instance)
        except (OSError, ValueError) as exc:
            skipped.append({"path": str(path), "reason": repr(exc)})
            print(f"skipping {path.name}: {exc}", file=sys.stderr, flush=True)
            continue
        nonempty = [route for route in incumbent.routes if len(route) >= 2]
        if not nonempty:
            skipped.append({"path": str(path), "reason": "no route contains two customers"})
            continue
        same_route = max(nonempty, key=len)
        before, after = same_route[0], same_route[-1]
        same_left, same_right = same_route[:2]
        different = None
        for left_idx, left_route in enumerate(incumbent.routes):
            for right_idx, right_route in enumerate(incumbent.routes):
                if left_idx != right_idx and left_route and right_route:
                    different = (left_route[0], right_route[0])
                    break
            if different:
                break
        constraints = [
            ConstraintSpec(
                "same_resource",
                params={"entities": [same_left, same_right]},
                source_text=f"客户 {same_left} 和客户 {same_right} 必须由同一辆车配送。",
            ),
            ConstraintSpec(
                "precedence",
                params={"before": before, "after": after},
                source_text=f"客户 {before} 必须先于客户 {after} 完成配送。",
            ),
        ]
        descriptions = [constraint.source_text for constraint in constraints]
        if different:
            constraints.append(
                ConstraintSpec(
                    "mutual_exclusion",
                    params={"entities": list(different)},
                    source_text=f"客户 {different[0]} 和客户 {different[1]} 不能由同一辆车配送。",
                )
            )
            descriptions.append(constraints[-1].source_text)
        instance.constraints = constraints
        instance.description = " ".join(descriptions)
        instance.name += "-nl"
        instance.metadata["known_feasible_routes"] = incumbent.routes
        checked = verify_solution(instance, incumbent)
        if checked.feasible:
            instances.append(instance)
        else:
            skipped.append({
                "path": str(path),
                "reason": "generated semantic constraints invalidate the known incumbent",
                "violations": checked.violations,
            })
    save_jsonl(instances, args.output_jsonl)
    manifest = {
        "input_dir": str(args.input_dir),
        "requested_limit": args.limit,
        "written": len(instances),
        "skipped": skipped,
    }
    manifest_path = args.output_jsonl.with_suffix(".manifest.json")
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(
        f"wrote {len(instances)} instances to {args.output_jsonl}; "
        f"skipped {len(skipped)} (manifest: {manifest_path})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
