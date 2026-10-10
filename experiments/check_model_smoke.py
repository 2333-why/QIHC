#!/usr/bin/env python3
"""Do not let an entirely falling-back LLM arm pass as successful inference."""

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    rows = json.loads((args.root / "results.json").read_text(encoding="utf-8"))
    audits = []
    for path in args.root.glob("llm_audit_rank*.jsonl"):
        audits.extend(json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line)
    if not rows or not any(row.get("reference_feasible") for row in rows):
        raise RuntimeError("Actual-model smoke produced no strictly feasible job")
    successes = sum(bool(row.get("ok")) for row in audits)
    if successes == 0:
        raise RuntimeError("All actual LLM generations failed; inspect audits before expensive experiments")
    record_count = sum(len(row.get("records", [])) for row in rows)
    fallbacks = sum("fallback" in record.get("selector_source", "") for row in rows for record in row.get("records", []))
    if record_count and fallbacks == record_count:
        raise RuntimeError("Every search proposal fell back; this is not a functioning LLM arm")
    (args.root / "model_smoke_check.json").write_text(json.dumps({"successful_generations": successes,
        "audit_calls": len(audits), "search_records": record_count, "fallbacks": fallbacks, "status": "passed"}, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
