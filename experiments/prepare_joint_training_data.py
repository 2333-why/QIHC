#!/usr/bin/env python3
"""Build joint constraint/proposal SFT, DPO and recorded-reward GRPO data."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qihc.problems.cvrp.instance import load_jsonl
from qihc.s2e.cpp import ConstraintProgramPackage


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def proposal_prompt(row: dict) -> str:
    return (
        "Given this constrained combinatorial optimization instance, propose a candidate solution "
        "and a compact p-bit search neighborhood. Return JSON only; do not generate an energy function.\n"
        f"Problem: {row.get('problem_prompt', '')}\n"
        f"Constraints: {json.dumps(row.get('constraints', []), ensure_ascii=False)}"
    )


def load_llm_audits(path: Path | None) -> dict[tuple[str, int, int], dict]:
    if path is None:
        return {}
    files = sorted(path.glob("llm_audit_rank*.jsonl")) if path.is_dir() else [path]
    audits: dict[tuple[str, int, int], dict] = {}
    for source in files:
        if not source.exists():
            continue
        for row in read_jsonl(source):
            if (row.get("ok") and row.get("instance") is not None
                    and row.get("search_seed") is not None
                    and row.get("iteration") is not None):
                audits[(
                    str(row["instance"]), int(row["search_seed"]), int(row["iteration"])
                )] = row
    return audits


def pbit_teacher_proposal(proposal: dict, updates: dict) -> dict:
    """Distill p-bit posterior residuals into the LLM's route logits."""

    teacher = copy.deepcopy(proposal)
    routes_by_customer = teacher.get("candidate_routes", {})
    logits = teacher.setdefault("candidate_route_logits", {})
    for customer, routes in routes_by_customer.items():
        customer_key = str(customer)
        base = logits.get(customer_key, logits.get(customer, {}))
        residual = updates.get(customer_key, updates.get(customer, {}))
        corrected = {}
        for route in routes:
            route_key = str(route)
            base_value = base.get(route_key, base.get(route, 0.0))
            residual_value = residual.get(route_key, residual.get(route, 0.0))
            corrected[route_key] = round(float(base_value) + float(residual_value), 3)
        logits[customer_key] = corrected
    teacher["candidate_route_logits"] = logits
    teacher.pop("recovered_truncated_logits", None)
    return teacher


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--constraint-records", type=Path, required=True)
    parser.add_argument("--solver-results", type=Path, required=True)
    parser.add_argument("--llm-audits", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=True)

    instances = {instance.name: instance for instance in load_jsonl(args.data)}
    constraint_records = {row["instance"]: row for row in read_jsonl(args.constraint_records)}
    solver_rows = json.loads(args.solver_results.read_text(encoding="utf-8"))
    audits = load_llm_audits(args.llm_audits)
    sft: list[dict] = []; dpo: list[dict] = []; grpo: list[dict] = []

    for name, instance in instances.items():
        prompt = "Translate the natural-language constraints into verified CPP JSON.\n" + instance.description
        gold = ConstraintProgramPackage.from_specs(
            name, instance.description, instance.customer_ids, instance.constraints, {"source": "gold"}
        ).canonical_json()
        sft.append({"task": "constraint", "prompt": prompt, "completion": gold})
        grpo.append({"task": "constraint", "prompt": prompt, "gold_completion": gold, "recorded_reward": 2.0})
        predicted = constraint_records.get(name)
        if predicted and predicted.get("status") == "ok":
            rejected = json.dumps(predicted["cpp"], sort_keys=True, ensure_ascii=False, separators=(",", ":"))
            if rejected != gold:
                dpo.append({"task": "constraint", "prompt": prompt, "chosen": gold, "rejected": rejected})

    for row in solver_rows:
        if row.get("status") != "ok" or row.get("method") != "llm":
            continue
        fallback_prompt = proposal_prompt(row)
        candidates = [record for record in row.get("records", []) if record.get("proposal_payload")]
        if not candidates:
            continue
        if audits:
            search_seed = int(row.get("search_seed", -1))
            candidates = [
                record for record in candidates
                if (str(row["instance"]), search_seed, int(record["iteration"])) in audits
            ]
            for record in candidates:
                audit = audits[(
                    str(row["instance"]), search_seed, int(record["iteration"])
                )]
                prompt = audit["prompt"]
                rejected_payload = record["proposal_payload"]
                chosen_payload = pbit_teacher_proposal(
                    rejected_payload, record.get("pbit_logit_updates", {})
                )
                rejected = json.dumps(rejected_payload, ensure_ascii=False, sort_keys=True)
                chosen = json.dumps(chosen_payload, ensure_ascii=False, sort_keys=True)
                reward = (
                    2.0 * float(record.get("accepted", False))
                    + float(record.get("objective_improvement", 0.0))
                    + 0.5 * float(record.get("candidate_route_recall", 0.0))
                    + 0.5 * float(record.get("candidate_compression", 0.0))
                )
                sft.append({"task": "proposal", "prompt": prompt, "completion": chosen, "recorded_reward": reward})
                grpo.append({"task": "proposal", "prompt": prompt, "gold_completion": chosen, "recorded_reward": reward})
                if chosen != rejected:
                    dpo.append({"task": "proposal", "prompt": prompt, "chosen": chosen, "rejected": rejected})
            continue
        prompt = fallback_prompt
        ranked = sorted(
            candidates,
            key=lambda record: (
                bool(record.get("accepted")),
                float(record.get("objective_improvement", 0.0)),
                float(record.get("candidate_route_recall", 0.0)),
                float(record.get("candidate_compression", 0.0)),
            ),
            reverse=True,
        )
        chosen_record = ranked[0]
        chosen = json.dumps(chosen_record["proposal_payload"], ensure_ascii=False, sort_keys=True)
        reward = (
            2.0 * float(chosen_record.get("accepted", False))
            + float(chosen_record.get("objective_improvement", 0.0))
            + 0.5 * float(chosen_record.get("candidate_route_recall", 0.0))
            + 0.5 * float(chosen_record.get("candidate_compression", 0.0))
        )
        sft.append({"task": "proposal", "prompt": prompt, "completion": chosen, "recorded_reward": reward})
        grpo.append({"task": "proposal", "prompt": prompt, "gold_completion": chosen, "recorded_reward": reward})
        if len(ranked) > 1:
            rejected = json.dumps(ranked[-1]["proposal_payload"], ensure_ascii=False, sort_keys=True)
            if rejected != chosen:
                dpo.append({"task": "proposal", "prompt": prompt, "chosen": chosen, "rejected": rejected})

    write_jsonl(args.output / "sft.jsonl", sft)
    write_jsonl(args.output / "dpo.jsonl", dpo)
    write_jsonl(args.output / "grpo.jsonl", grpo)
    (args.output / "summary.json").write_text(
        json.dumps({
            "sft": len(sft), "dpo": len(dpo), "grpo": len(grpo),
            "constraint_sft": sum(row.get("task") == "constraint" for row in sft),
            "proposal_sft": sum(row.get("task") == "proposal" for row in sft),
            "used_exact_llm_audits": bool(audits),
        }, indent=2), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
