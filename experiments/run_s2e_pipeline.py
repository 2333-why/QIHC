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


def canonical(program) -> str:
    kind = "same_resource" if program.type == "same_vehicle" else program.type
    return json.dumps({"type": kind, "hard": program.hard, "params": program.params}, sort_keys=True, ensure_ascii=False)


def write_jsonl(path: Path, rows) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows: handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, required=True); p.add_argument("--output", type=Path, required=True)
    p.add_argument("--backend", choices=["heuristic", "local-llm"], default="local-llm")
    p.add_argument("--model-path"); p.add_argument("--limit", type=int, default=0); p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max-new-tokens", type=int, default=768)
    args = p.parse_args(); rank = int(os.environ.get("RANK", 0)); world = int(os.environ.get("WORLD_SIZE", 1)); local_rank = int(os.environ.get("LOCAL_RANK", rank))
    dist = None
    if world > 1:
        import torch.distributed as td
        td.init_process_group("gloo"); dist = td
    args.output.mkdir(parents=True, exist_ok=True)
    instances = load_jsonl(args.data)[: args.limit or None]
    backend = HeuristicConstraintBackend() if args.backend == "heuristic" else LocalLLMConstraintBackend(args.model_path, f"cuda:{local_rank}", args.temperature, args.max_new_tokens)
    synth = ConstraintSynthesizer(backend); validator = CPPValidator()
    records, feedback = [], []
    for idx, instance in enumerate(instances):
        if idx % world != rank: continue
        try:
            cpp, repair_reports = synth.synthesize_verified(instance, validator); report = repair_reports[-1]; plan = compile_cpp(cpp, instance, report) if report.passed and report.highest_stage == "A4_ENCODING" else None
            gold = {json.dumps({"type": "same_resource" if s.type == "same_vehicle" else s.type, "hard": s.hard, "params": s.params}, sort_keys=True, ensure_ascii=False) for s in instance.constraints}
            pred = {canonical(x) for x in cpp.programs}; semantic = gold == pred
            response = cpp.canonical_json()
            record = {"instance": instance.name, "status": "ok", "cpp": cpp.to_dict(), "cpp_checksum": cpp.checksum, "validation": report.to_dict(), "validation_attempts": [x.to_dict() for x in repair_reports], "compilation": plan.to_dict() if plan else None, "exact_match": semantic, "gold": sorted(gold), "predicted": sorted(pred)}
            feedback.append(FeedbackRecord(instance.description, response, True, semantic, report.passed, compile_cost=float(plan.estimated_binary_variables if plan else 0), counterexamples=report.counterexamples))
        except Exception as exc:
            record = {"instance": instance.name, "status": "error", "error": repr(exc)}
            feedback.append(FeedbackRecord(instance.description, json.dumps(record, ensure_ascii=False), False, False, False))
        records.append(record); print(json.dumps({"rank": rank, "instance": instance.name, "status": record["status"]}), flush=True)
    write_jsonl(args.output / f"records_rank{rank}.jsonl", records)
    write_jsonl(args.output / f"feedback_rank{rank}.jsonl", [vars(x) for x in feedback])
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
        summary = {"n": len(all_records), "success_rate": len(ok) / max(len(all_records), 1), "a4_pass_rate": sum(x["validation"]["passed"] for x in ok) / max(len(all_records), 1), "exact_match_rate": sum(x["exact_match"] for x in ok) / max(len(all_records), 1), "constraint_fallback_rate": fallback_count / max(len(all_records), 1)}
        (args.output / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    if dist: dist.barrier(); dist.destroy_process_group()
    return 0


if __name__ == "__main__": raise SystemExit(main())
