#!/usr/bin/env python3
"""Export one verified neutral initialization per instance/seed for all arms."""

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = json.loads(args.results.read_text(encoding="utf-8"))
    selected = []
    for row in rows:
        if row.get("reference_feasible", row.get("feasible", False)):
            selected.append({"instance": row["instance"], "search_seed": row["search_seed"],
                             "solution": {"routes": row["solution"]["routes"]},
                             "construction_elapsed_s": row["total_elapsed_s"]})
    if not selected:
        raise SystemExit("No verified shared initial solutions")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in selected:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({"shared_initials": len(selected), "attempted_success_rows": len(rows)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
