#!/usr/bin/env python3
"""Build joint constraint/proposal SFT, DPO and recorded-reward GRPO data."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qihc.problems.cvrp.instance import load_jsonl
from qihc.problems.cvrp.training_reward import semantic_signature, rollout_reward
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


def cold_start_prompt(record: dict) -> str:
    if record.get("prompt"):
        return str(record["prompt"])
    return (
        "Construct this CVRP batch using the trusted insertion costs, remaining capacity, "
        "future-demand lookahead, and semantic constraints. Return JSON containing "
        "destroy_customers, candidate_routes, candidate_route_scores, and confidence.\n"
        f"Numeric construction state: {json.dumps(record.get('context', {}), ensure_ascii=False, separators=(',', ':'))}"
    )


def append_cold_start_examples(row: dict, sft: list[dict], dpo: list[dict], grpo: list[dict]) -> int:
    records = row.get("solution", {}).get("metadata", {}).get("cold_start_records", [])
    added = 0
    for record in records:
        teacher = record.get("teacher_payload")
        if not isinstance(teacher, dict):
            continue
        prompt = cold_start_prompt(record)
        chosen = json.dumps(teacher, ensure_ascii=False, sort_keys=True)
        margin = max(0.0, float(record.get("objective_margin", 0.0)))
        reward = 2.0 + math.log1p(margin)
        if record.get("selected_attempt", True):
            sft.append({"task": "proposal", "phase": "cold_start", "prompt": prompt,
                        "completion": chosen, "recorded_reward": reward})
            grpo_row = {"task": "proposal", "phase": "cold_start", "prompt": prompt,
                        "gold_completion": chosen, "recorded_reward": reward}
            if row.get("training_instance") and record.get("incumbent_routes_before") is not None:
                grpo_row["reward_context"] = {
                    "instance": row["training_instance"],
                    "routes": record["incumbent_routes_before"],
                    "allowed_routes": teacher["candidate_routes"],
                    "partial": True,
                    "reference_objective": record.get("accepted_partial_objective") or 1.0,
                    "seed": record.get("seed", 0),
                    "cold_context": record.get("context", {}),
                    "solver_config": row.get("config", {}),
                }
            grpo.append(grpo_row)
        rejected_payload = record.get("rejected_payload") or record.get("proposal_payload")
        if isinstance(rejected_payload, dict) and margin > 1e-9:
            rejected = json.dumps(rejected_payload, ensure_ascii=False, sort_keys=True)
            if rejected != chosen:
                dpo.append({"task": "proposal", "phase": "cold_start", "prompt": prompt,
                            "chosen": chosen, "rejected": rejected,
                            "reward_context": grpo_row.get("reward_context") if record.get("selected_attempt", True) else None})
        added += int(record.get("selected_attempt", True))
    return added


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
                    and row.get("iteration") is not None
                    and row.get("phase", "lns") != "cold_start"):
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


def inference_domain(prompt: str, fallback: dict) -> dict:
    """Recover the shortlist supplied to the model, not the logged teacher's subset."""
    match = re.search(r"Objective-aware move features: ([^\n]+)", prompt)
    if not match:
        return {str(c): list(routes) for c, routes in fallback.items()}
    features = json.loads(match.group(1))
    return {str(row["customer"]): list(dict.fromkeys(
        [row["current_route"], *(move["route"] for move in row["moves"])]
    )) for row in features}


def align_teacher_domain(payload: dict, allowed: dict) -> dict | None:
    """Never supervise safe-expansion route IDs that were absent from the prompt."""
    teacher = copy.deepcopy(payload)
    for customer in teacher.get("destroy_customers", []):
        key = str(customer)
        if key not in allowed:
            return None
        selected = [int(r) for r in teacher["candidate_routes"].get(key, []) if int(r) in allowed[key]]
        if not selected:
            return None
        teacher["candidate_routes"][key] = selected
        logits = teacher.get("candidate_route_logits", {}).get(key, {})
        teacher.setdefault("candidate_route_logits", {})[key] = {str(r): logits.get(str(r), logits.get(r, 0.0)) for r in selected}
    return teacher


def canonical_proposal(payload: dict) -> str:
    """Train the inference schema, never debug metadata or alphabetic field order."""
    customers = list(dict.fromkeys(int(c) for c in payload["destroy_customers"]))
    if not customers:
        raise ValueError("Empty proposal")
    routes, logits = {}, {}
    for customer in customers:
        key = str(customer)
        routes[key] = list(dict.fromkeys(int(r) for r in payload["candidate_routes"][key]))
        if not routes[key]:
            raise ValueError("Empty route domain")
        source = payload.get("candidate_route_logits", {}).get(key, {})
        logits[key] = {}
        for route in routes[key]:
            value = float(source.get(str(route), source.get(route, 0)))
            if not math.isfinite(value):
                raise ValueError("Nonfinite logit")
            # Match the compact integer protocol requested at inference time.
            logits[key][str(route)] = max(-2, min(2, round(value)))
    confidence = float(payload.get("confidence", 1))
    if not math.isfinite(confidence):
        raise ValueError("Nonfinite confidence")
    return json.dumps({"destroy_customers": customers, "candidate_routes": routes,
                       "candidate_route_logits": logits, "confidence": round(max(0, min(1, confidence)), 3)},
                      ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--constraint-records", type=Path, required=True)
    parser.add_argument("--solver-results", type=Path, required=True)
    parser.add_argument("--llm-audits", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--holdout-modulus", type=int, default=0)
    parser.add_argument("--holdout-fold", type=int, default=0)
    parser.add_argument("--competitive-teacher", action="store_true")
    parser.add_argument("--preference-chains", type=int, default=64)
    parser.add_argument("--preference-steps", type=int, default=48)
    parser.add_argument("--preference-margin", type=float, default=0.001)
    parser.add_argument("--verify-preferences", action="store_true",
                        help="Keep proposal DPO pairs only after matched execution rewards establish a preference")
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    if args.competitive_teacher and (not args.verify_preferences or not args.llm_audits):
        parser.error("Competitive teachers require --verify-preferences and --llm-audits")
    if min(args.preference_chains, args.preference_steps) <= 0 or args.preference_margin <= 0:
        parser.error("Preference rollout budgets and margin must be positive")

    instances = {instance.name: instance for instance in load_jsonl(args.data)}
    if args.holdout_modulus < 0 or (
        args.holdout_modulus and not 0 <= args.holdout_fold < args.holdout_modulus
    ):
        raise ValueError("holdout fold must be within the positive modulus")
    holdout = {
        name for name in instances
        if args.holdout_modulus and int(hashlib.sha256(name.encode()).hexdigest(), 16) % args.holdout_modulus == args.holdout_fold
    }
    training_instances = {name: instance for name, instance in instances.items() if name not in holdout}
    constraint_records = {row["instance"]: row for row in read_jsonl(args.constraint_records)}
    solver_rows = json.loads(args.solver_results.read_text(encoding="utf-8"))
    audits = load_llm_audits(args.llm_audits)
    if args.competitive_teacher and not audits:
        raise ValueError("No successful exact inference audits for competitive supervision")
    sft: list[dict] = []; dpo: list[dict] = []; grpo: list[dict] = []
    cold_start_examples = 0
    competitions = []

    for name, instance in training_instances.items():
        predicted = constraint_records.get(name)
        generator = predicted.get("cpp", {}).get("generator", {}) if predicted else {}
        prompt = generator.get("prompt") or (
            "Convert these routing requirements to JSON Constraint IR with a constraints list. "
            "Each item must contain type, hard, weight and params.\n"
            f"Valid customer IDs: {instance.customer_ids}\nRequirements: {instance.description}"
        )
        gold_payload = {"constraints": [
            {"type": spec.type, "hard": spec.hard, "weight": spec.weight, "params": spec.params}
            for spec in instance.constraints
        ]}
        gold = json.dumps(gold_payload, sort_keys=True, ensure_ascii=False)
        sft.append({"task": "constraint", "prompt": prompt, "completion": gold})
        grpo.append({"task": "constraint", "prompt": prompt, "gold_completion": gold, "recorded_reward": 2.0})
        if predicted and predicted.get("status") == "ok":
            rejected_payload = {"constraints": [
                {"type": spec["type"], "hard": spec.get("hard", True), "weight": spec.get("weight", 1.0), "params": spec["params"]}
                for spec in predicted["cpp"].get("programs", [])
            ]}
            rejected = json.dumps(rejected_payload, sort_keys=True, ensure_ascii=False)
            if semantic_signature(rejected_payload) != semantic_signature(gold_payload):
                dpo.append({"task": "constraint", "prompt": prompt, "chosen": gold, "rejected": rejected})

    for row in solver_rows:
        if (row.get("status") != "ok" or row.get("method") != "llm" or row.get("reference_feasible") is False
                or row.get("instance") not in training_instances):
            continue
        cold_start_examples += append_cold_start_examples(row, sft, dpo, grpo)
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
                chosen_payload = record.get("teacher_payload") or pbit_teacher_proposal(
                    rejected_payload, record.get("pbit_logit_updates", {})
                )
                allowed_routes = inference_domain(prompt, chosen_payload.get("candidate_routes", {}))
                chosen_payload = align_teacher_domain(chosen_payload, allowed_routes)
                if chosen_payload is None:
                    continue
                try:
                    rejected = canonical_proposal(rejected_payload)
                    chosen = canonical_proposal(chosen_payload)
                except (ValueError, TypeError, KeyError):
                    continue
                reward = (
                    2.0 * float(record.get("accepted", False))
                    + float(record.get("objective_improvement", 0.0))
                    + 0.5 * float(record.get("candidate_route_recall", 0.0))
                    + 0.5 * float(record.get("candidate_compression", 0.0))
                )
                sft_row = {"task": "proposal", "prompt": prompt, "completion": chosen, "recorded_reward": reward}
                grpo_row = {"task": "proposal", "prompt": prompt, "gold_completion": chosen, "recorded_reward": reward}
                if row.get("training_instance") and record.get("incumbent_routes_before"):
                    grpo_row["reward_context"] = {
                        "instance": row["training_instance"], "routes": record["incumbent_routes_before"],
                        "allowed_routes": allowed_routes,
                        "required_destroy_count": len(chosen_payload["destroy_customers"]),
                        "reference_objective": record["incumbent_before"], "seed": search_seed,
                        "solver_config": row.get("config", {}),
                        "sampler_backend": record.get("sampler_backend", "pdit"),
                        "iteration": int(record["iteration"]),
                        "energy_feedback_context": record.get("energy_feedback_context", {}),
                    }
                if args.competitive_teacher:
                    from experiments.competitive_teacher import select_teacher
                    context = grpo_row.get("reward_context")
                    if context is None:
                        raise ValueError("Competitive supervision requires exported execution context")
                    competition = select_teacher(prompt, chosen_payload, rejected_payload, context,
                        chains=args.preference_chains, steps=args.preference_steps, margin=args.preference_margin)
                    competitions.append({"instance": row["instance"], "seed": search_seed,
                                         "iteration": record["iteration"], "competition": competition})
                    if not competition or not competition["winner"]:
                        continue
                    winner, loser = competition["winner"], competition["pair"]
                    chosen = winner["answer"]
                    sft_row.update(completion=chosen, teacher_source=winner["source"],
                                   recorded_reward=winner["selection_reward"])
                    grpo_row.update(gold_completion=chosen, recorded_reward=winner["selection_reward"])
                    sft.append(sft_row)
                    grpo.append(grpo_row)
                    if loser:
                        dpo.append({"task": "proposal", "prompt": prompt, "chosen": chosen,
                                    "rejected": loser["answer"], "reward_context": context,
                                    "competition_verified": True,
                                    "chosen_source": winner["source"], "rejected_source": loser["source"],
                                    "chosen_rewards": winner["rewards"], "rejected_rewards": loser["rewards"],
                                    "executed_reward_margin": min(a-b for a,b in zip(winner["rewards"], loser["rewards"]))})
                    continue
                sft.append(sft_row)
                grpo.append(grpo_row)
                if chosen != rejected and (args.verify_preferences or record.get("accepted", False)):
                    dpo.append({"task": "proposal", "prompt": prompt, "chosen": chosen, "rejected": rejected,
                                "reward_context": grpo_row.get("reward_context")})
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
        chosen = canonical_proposal(chosen_record["proposal_payload"])
        reward = (
            2.0 * float(chosen_record.get("accepted", False))
            + float(chosen_record.get("objective_improvement", 0.0))
            + 0.5 * float(chosen_record.get("candidate_route_recall", 0.0))
            + 0.5 * float(chosen_record.get("candidate_compression", 0.0))
        )
        sft.append({"task": "proposal", "prompt": prompt, "completion": chosen, "recorded_reward": reward})
        grpo.append({"task": "proposal", "prompt": prompt, "gold_completion": chosen, "recorded_reward": reward})
        if len(ranked) > 1:
            rejected = canonical_proposal(ranked[-1]["proposal_payload"])
            if rejected != chosen:
                dpo.append({"task": "proposal", "prompt": prompt, "chosen": chosen, "rejected": rejected})

    attempted_preferences = len(dpo)
    if args.verify_preferences:
        validated_pairs = []
        for pair in dpo:
            if args.competitive_teacher and pair.get("competition_verified"):
                validated_pairs.append(pair)
                continue
            if pair.get("task") == "constraint":
                validated_pairs.append(pair)
                continue
            context = pair.get("reward_context")
            if context is None:
                continue
            try:
                chosen_score = rollout_reward(json.loads(pair["chosen"]), context)
                rejected_score = rollout_reward(json.loads(pair["rejected"]), context)
            except (ValueError, KeyError, TypeError):
                continue
            if not all(math.isfinite(v) for v in (chosen_score, rejected_score)) or abs(chosen_score - rejected_score) <= 1e-9:
                continue
            if rejected_score > chosen_score:
                pair["chosen"], pair["rejected"] = pair["rejected"], pair["chosen"]
            pair["executed_reward_margin"] = abs(chosen_score - rejected_score)
            validated_pairs.append(pair)
        dpo = validated_pairs
    # Repeated logging must not amplify a preference's training weight.
    dpo = list({(r["prompt"], r["chosen"], r["rejected"]): r for r in dpo}.values())
    write_jsonl(args.output / "sft.jsonl", sft)
    write_jsonl(args.output / "dpo.jsonl", dpo)
    write_jsonl(args.output / "grpo.jsonl", grpo)
    write_jsonl(args.output / "teacher_competitions.jsonl", competitions)
    (args.output / "summary.json").write_text(
        json.dumps({
            "sft": len(sft), "dpo": len(dpo), "grpo": len(grpo),
            "constraint_sft": sum(row.get("task") == "constraint" for row in sft),
            "proposal_sft": sum(row.get("task") == "proposal" for row in sft),
            "cold_start_sft": cold_start_examples,
            "used_exact_llm_audits": bool(audits),
            "training_instances": len(training_instances),
            "holdout_instances": len(holdout),
            "preference_pairs_attempted": attempted_preferences,
            "preferences_execution_verified": args.verify_preferences,
            "competitive_teacher": args.competitive_teacher,
            "competition_states": len(competitions),
            "teacher_wins": {name: sum(r.get("teacher_source") == name for r in sft)
                             for name in ("knn", "pbit", "llm")},
            "preference_rollout": {"chains": args.preference_chains, "steps": args.preference_steps,
                                   "ranking_seeds": 2, "confirmation_seeds": 1, "minimum_reward_margin": args.preference_margin},
        }, indent=2), encoding="utf-8"
    )
    (args.output / "holdout_instances.json").write_text(
        json.dumps(sorted(holdout), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
