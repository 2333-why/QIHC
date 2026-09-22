#!/usr/bin/env python3
"""Distributed QIHC-S2E constraint synthesis, verification and compilation."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))

from qihc.problems.cvrp.instance import load_jsonl
from qihc.s2e import CPPValidator, ConstraintSynthesizer, FeedbackRecord, build_dpo_pairs, build_grpo_records, build_sft_records, compile_cpp
from qihc.s2e.synthesizer import HeuristicConstraintBackend, LocalLLMConstraintBackend


def canonical_constraint(kind: str, hard: bool, params: dict) -> str:
    kind = "same_resource" if kind == "same_vehicle" else kind
    normalized = dict(params)
    if kind in {"same_resource", "mutual_exclusion"}:
        entities = normalized.get("entities", normalized.get("customers", []))
        normalized = {"entities": sorted(int(value) for value in entities)}
    elif kind == "precedence":
        normalized = {
            "before": int(normalized["before"]),
            "after": int(normalized["after"]),
        }
    return json.dumps(
        {"type": kind, "hard": bool(hard), "params": normalized},
        sort_keys=True,
        ensure_ascii=False,
    )


def canonical(program) -> str:
    return canonical_constraint(program.type, program.hard, program.params)


def semantic_closure(items: set[str]) -> set[str]:
    """Canonical constraint set including implications used by the compiler.

    A precedence constraint necessarily places both customers on one route, so
    the compiler materializes a same_resource constraint as well.  Comparing
    raw sets therefore reports a false mismatch for a semantically exact CPP.
    """
    closed = set(items)
    for item in items:
        value = json.loads(item)
        if value.get("type") != "precedence":
            continue
        params = value.get("params", {})
        before, after = params.get("before"), params.get("after")
        if before is None or after is None:
            continue
        closed.add(json.dumps({
            "type": "same_resource",
            "hard": value.get("hard", True),
            "params": {"entities": [before, after]},
        }, sort_keys=True, ensure_ascii=False))
    return closed


def write_jsonl(path: Path, rows) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows: handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_checkpoint(path: Path) -> dict[str, dict]:
    """Recover complete per-instance records, ignoring a torn final write."""
    selected: dict[str, dict] = {}
    if not path.exists():
        return selected
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(item, dict):
            continue
        record, feedback = item.get("record"), item.get("feedback")
        if isinstance(record, dict) and isinstance(feedback, dict) and isinstance(record.get("instance"), str):
            selected[record["instance"]] = item
    return selected


def checkpoint_is_complete(item: dict | None) -> bool:
    """Only reuse records produced by the semantics-aware pipeline."""
    if not item:
        return False
    record = item.get("record", {})
    return record.get("status") == "ok" and record.get("semantic_match") is True


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, required=True); p.add_argument("--output", type=Path, required=True)
    p.add_argument("--backend", choices=["heuristic", "local-llm"], default="local-llm")
    p.add_argument("--model-path"); p.add_argument("--adapter-path"); p.add_argument("--limit", type=int, default=0); p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max-new-tokens", type=int, default=768)
    p.add_argument("--resume", action="store_true", help="Skip successful per-instance checkpoints")
    args = p.parse_args(); rank = int(os.environ.get("RANK", 0)); world = int(os.environ.get("WORLD_SIZE", 1)); local_rank = int(os.environ.get("LOCAL_RANK", rank))
    dist = None
    if world > 1:
        import torch.distributed as td
        td.init_process_group("gloo"); dist = td
    args.output.mkdir(parents=True, exist_ok=True)
    instances = load_jsonl(args.data)[: args.limit or None]
    backend = HeuristicConstraintBackend() if args.backend == "heuristic" else LocalLLMConstraintBackend(args.model_path, f"cuda:{local_rank}", args.temperature, args.max_new_tokens, args.adapter_path)
    synth = ConstraintSynthesizer(backend); validator = CPPValidator()
    shard = [instance for idx, instance in enumerate(instances) if idx % world == rank]
    checkpoint_path = args.output / f"checkpoint_rank{rank}.jsonl"
    selected: dict[str, dict] = {}
    if args.resume:
        # A resumed run may change from two GPUs to one (or the reverse).
        # Checkpoints are keyed by instance, not by the rank that produced them.
        for previous_path in sorted(args.output.glob("checkpoint_rank*.jsonl")):
            for name, item in read_checkpoint(previous_path).items():
                previous = selected.get(name)
                if previous is None or not checkpoint_is_complete(previous):
                    selected[name] = item
    print(json.dumps({"rank": rank, "resume": args.resume, "successful_checkpoints": sum(checkpoint_is_complete(item) for item in selected.values()), "remaining_instances": sum(not checkpoint_is_complete(selected.get(instance.name)) for instance in shard)}), flush=True)
    if args.resume and checkpoint_path.exists() and checkpoint_path.stat().st_size:
        with checkpoint_path.open("rb+") as existing:
            existing.seek(-1, os.SEEK_END)
            if existing.read(1) != b"\n":
                existing.write(b"\n")
    with checkpoint_path.open("a" if args.resume else "w", encoding="utf-8") as journal:
        for instance in shard:
            if checkpoint_is_complete(selected.get(instance.name)):
                continue
            try:
                cpp, repair_reports = synth.synthesize_verified(instance, validator); report = repair_reports[-1]; plan = compile_cpp(cpp, instance, report) if report.passed and report.highest_stage == "A4_ENCODING" else None
                gold = {canonical_constraint(s.type, s.hard, s.params) for s in instance.constraints}
                pred = {canonical(x) for x in cpp.programs}
                exact = gold == pred
                semantic = semantic_closure(gold) == semantic_closure(pred)
                response = cpp.canonical_json()
                record = {"instance": instance.name, "status": "ok", "cpp": cpp.to_dict(), "cpp_checksum": cpp.checksum, "validation": report.to_dict(), "validation_attempts": [x.to_dict() for x in repair_reports], "compilation": plan.to_dict() if plan else None, "exact_match": exact, "semantic_match": semantic, "gold": sorted(gold), "predicted": sorted(pred)}
                feedback = FeedbackRecord(instance.description, response, True, semantic, report.passed, compile_cost=float(plan.estimated_binary_variables if plan else 0), counterexamples=report.counterexamples)
            except Exception as exc:
                record = {"instance": instance.name, "status": "error", "error": repr(exc)}
                feedback = FeedbackRecord(instance.description, json.dumps(record, ensure_ascii=False), False, False, False)
            item = {"record": record, "feedback": vars(feedback)}
            journal.write(json.dumps(item, ensure_ascii=False) + "\n")
            journal.flush()
            selected[instance.name] = item
            print(json.dumps({"rank": rank, "instance": instance.name, "status": record["status"]}), flush=True)
    records = [selected[instance.name]["record"] for instance in shard]
    feedback = [selected[instance.name]["feedback"] for instance in shard]
    write_jsonl(args.output / f"records_rank{rank}.jsonl", records)
    write_jsonl(args.output / f"feedback_rank{rank}.jsonl", feedback)
    if dist: dist.barrier()
    if rank == 0:
        all_records, all_feedback = [], []
        for r in range(world):
            all_records += [json.loads(x) for x in (args.output / f"records_rank{r}.jsonl").read_text(encoding="utf-8").splitlines() if x]
            all_feedback += [FeedbackRecord(**json.loads(x)) for x in (args.output / f"feedback_rank{r}.jsonl").read_text(encoding="utf-8").splitlines() if x]
        write_jsonl(args.output / "records.jsonl", all_records)
        write_jsonl(args.output / "sft.jsonl", build_sft_records(all_feedback)); write_jsonl(args.output / "dpo.jsonl", build_dpo_pairs(all_feedback)); write_jsonl(args.output / "grpo.jsonl", build_grpo_records(all_feedback))
        ok = [x for x in all_records if x["status"] == "ok"]
        fallback_count = sum("fallback" in x.get("cpp", {}).get("generator", {}).get("backend", "") or "repaired" in x.get("cpp", {}).get("generator", {}).get("backend", "") for x in ok)
        summary = {"n": len(all_records), "success_rate": len(ok) / max(len(all_records), 1), "a4_pass_rate": sum(x["validation"]["passed"] for x in ok) / max(len(all_records), 1), "exact_match_rate": sum(x["exact_match"] for x in ok) / max(len(all_records), 1), "semantic_match_rate": sum(x.get("semantic_match", x["exact_match"]) for x in ok) / max(len(all_records), 1), "constraint_fallback_rate": fallback_count / max(len(all_records), 1)}
        (args.output / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    if dist: dist.barrier(); dist.destroy_process_group()
    return 0


if __name__ == "__main__": raise SystemExit(main())
