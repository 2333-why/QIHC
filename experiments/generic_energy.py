#!/usr/bin/env python3
"""Generic energy experiment stages; isolated processes release GPU between stages."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from qihc.generic.energy import POLICIES, compile_energy, solve, objective, violations
from qihc.generic.benchmark import (make_dataset, objective_prompt, constraint_prompt, infer,
                                    parse_object, semantic_equal, reference_solve)


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def read_rows(path):
    return [json.loads(s) for s in path.read_text(encoding="utf-8").splitlines() if s.strip()]


def append(path, row):
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()


def teacher(args, cases):
    sft, dpo, audit = [], [], []
    for case in cases:
        if case["split"] != "train" or case["family"] == args.holdout_family:
            continue
        spec = case["gold"]
        first = {"n": spec["n"], "objective": spec["objective"]}
        prompt = constraint_prompt(case["description"], first)
        runs = {p: [solve(spec, policy=p, seed=s, chains=args.chains, steps=args.steps) for s in (11, 29, 47)] for p in POLICIES}

        def score(result):
            # Feasible beats every infeasible answer. Feasible reward lies in [1,2].
            # Semantics are held identical across ALL candidates; no constraint dropping.
            if not result["feasible"]:
                return -min(1, result["violation"] / max(1, len(spec["constraints"])))
            bound = max(1, sum(abs(c) for c in spec["objective"]["linear"]) + sum(abs(c) for _, _, c in spec["objective"]["quadratic"]))
            signed = result["objective"] * (1 if spec["objective"]["sense"] == "min" else -1)
            return 1.5 - signed / (2 * bound)

        ranks = sorted(POLICIES, key=lambda p: (-sum(score(r) for r in runs[p][:2]), p))
        winner, loser = ranks[0], ranks[-1]
        margin = sum(score(r) for r in runs[winner][:2]) / 2 - sum(score(r) for r in runs[loser][:2]) / 2
        confirmation = score(runs[winner][2]) - score(runs[loser][2])
        accepted = margin > .001 and confirmation > .001
        # If execution cannot distinguish policies, do not invent a preference.
        selected = winner if accepted else "fixed"
        completion = lambda policy: json.dumps({"constraints": spec["constraints"], "policy": policy}, separators=(",", ":"))
        sft.extend([
            {"task": "generic_objective", "case_id": case["id"], "prompt": objective_prompt(case["description"]), "completion": json.dumps(first, separators=(",", ":"))},
            {"task": "generic_constraints_policy", "case_id": case["id"], "prompt": prompt, "completion": completion(selected)},
        ])
        if accepted:
            dpo.append({"task": "generic_constraints_policy", "case_id": case["id"], "prompt": prompt,
                        "chosen": completion(winner), "rejected": completion(loser), "margin": margin, "confirmation_margin": confirmation})
        audit.append({"case_id": case["id"], "selected": selected, "accepted": accepted, "margin": margin,
                      "confirmation_margin": confirmation, "runs": runs})
        # Checkpoint per case; interrupted teacher generation keeps usable evidence.
        write_rows(args.output / "sft.jsonl", sft)
        write_rows(args.output / "dpo.jsonl", dpo)
        write_rows(args.output / "teacher_audit.jsonl", audit)
        print(f"TEACHER {case['id']} sft={len(sft)} dpo={len(dpo)}", flush=True)


class Generator:
    def __init__(self, args):
        from qihc.problems.cvrp.direct_llm import LocalDirectLLMGenerator
        self.inner = LocalDirectLLMGenerator(args.model_path, device=args.device, temperature=0,
                                             max_new_tokens=args.max_new_tokens, max_input_tokens=16384)
        if args.adapters:
            if len(args.adapters) != 1:
                raise ValueError("SFT->DPO updates the same adapter; load ONLY the final adapter, never add SFT twice")
            from peft import PeftModel
            for path in args.adapters:
                self.inner.model = PeftModel.from_pretrained(self.inner.model, path, local_files_only=True).merge_and_unload()
            self.inner.model.eval()
        self.path = args.output / f"{args.tag}_generation.jsonl"
        self.calls = 0
        self.input_tokens = self.output_tokens = 0

    def generate(self, prompt):
        start = time.perf_counter()
        text = self.inner.generate(prompt)
        # Actual generated token IDs, including terminal special tokens.
        input_tokens = self.inner.last_usage["input_tokens"]
        output_tokens = self.inner.last_usage["output_tokens"]
        self.calls += 1
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        append(self.path, {"call": self.calls, "prompt": prompt, "response": text, "seconds": time.perf_counter() - start,
                           "input_tokens": input_tokens, "output_tokens": output_tokens})
        return text


def evaluate(args, cases):
    output = args.output / f"{args.tag}.jsonl"
    if output.exists():
        raise FileExistsError(f"Refusing to append duplicate evaluation: {output}")
    gen = Generator(args) if args.mode in ("llm", "direct") else None
    for case in cases:
        if case["split"] == "train":
            continue
        gold = case["gold"]
        common = {"case_id": case["id"], "family": case["family"], "split": case["split"], "size": case["size"],
                  "holdout_family": case["family"] == args.holdout_family, "objective_sense": gold["objective"]["sense"]}
        if args.mode == "oracle":
            ref = reference_solve(gold, args.reference_seconds)
            append(output, {**common, "arm": "cp_sat_gold", "seed": None, "feasible": ref["x"] is not None, **ref})
            for seed in range(args.seeds):
                for arm, method, policy in (("gold_pbit_fixed", "pbit", "fixed"), ("gold_pbit_feedback", "pbit", "adaptive"),
                                             ("gold_metropolis", "metropolis", "adaptive"), ("gold_uniform", "uniform", "fixed")):
                    result = solve(gold, method=method, policy=policy, seed=seed, chains=args.chains, steps=args.steps)
                    append(output, {**common, "arm": arm, "semantic_match": True, **result})
        else:
            started = time.perf_counter()
            calls_before, input_before, output_before = gen.calls, gen.input_tokens, gen.output_tokens
            try:
                if args.mode == "direct":
                    raw = gen.generate('Solve this problem. Return JSON with exactly x, a list of 0/1 values in variable order. No code/prose.\n' + case["description"])
                    answer = parse_object(raw)
                    if set(answer) != {"x"} or not isinstance(answer["x"], list) or len(answer["x"]) != gold["n"] or any(type(v) is not int or v not in (0, 1) for v in answer["x"]):
                        raise ValueError("Invalid direct solution")
                    x = np.asarray(answer["x"])
                    v = float(violations(gold, x).sum())
                    append(output, {**common, "arm": args.tag, "seed": None, "x": answer["x"], "feasible": v == 0,
                                    "violation": v, "objective": float(objective(gold, x)), "seconds": time.perf_counter() - started,
                                    "llm_calls": gen.calls - calls_before, "input_tokens": gen.input_tokens - input_before,
                                    "output_tokens": gen.output_tokens - output_before})
                else:
                    predicted, policy, _ = infer(gen, case["description"])
                    if predicted["n"] != gold["n"]:
                        raise ValueError("Predicted variables cannot be aligned with benchmark variables")
                    compile_energy(predicted)  # Reject resource limits before emitting any paired arm.
                    model_seconds = time.perf_counter() - started
                    # Essential baseline: SAME LLM translation, conventional solver.
                    # Do not replace a failed translation with gold before solving.
                    ref = reference_solve(predicted, args.reference_seconds)
                    x_ref = np.asarray(ref["x"]) if ref["x"] is not None else None
                    ref_violation = float(violations(gold, x_ref).sum()) if x_ref is not None else None
                    append(output, {**common, **ref, "arm": args.tag + "_cp_sat", "seed": None,
                                    "feasible": x_ref is not None and ref_violation == 0, "violation": ref_violation,
                                    "objective": float(objective(gold, x_ref)) if x_ref is not None else None,
                                    "semantic_match": semantic_equal(predicted, gold), "model_seconds": model_seconds,
                                    "translation_shared_across_seeds_and_fixed_arm": True})
                    for seed in range(args.seeds):
                        # Paired no-policy ablation uses the very same model translation.
                        for arm, choice in ((args.tag, policy), (args.tag + "_fixed", "fixed")):
                            result = solve(predicted, policy=choice, seed=seed, chains=args.chains, steps=args.steps)
                            x = np.asarray(result["x"])
                            v = float(violations(gold, x).sum())
                            append(output, {**common, "arm": arm, **result, "predicted_feasible": result["feasible"],
                                            "predicted_objective": result["objective"], "feasible": v == 0, "violation": v,
                                            "objective": float(objective(gold, x)), "semantic_match": semantic_equal(predicted, gold),
                                            "model_seconds": model_seconds, "llm_calls": gen.calls - calls_before,
                                            "input_tokens": gen.input_tokens - input_before, "output_tokens": gen.output_tokens - output_before,
                                            "translation_shared_across_seeds_and_fixed_arm": True})
            except (ValueError, TypeError, KeyError, IndexError, OverflowError) as exc:
                # Infrastructure/OOM errors intentionally propagate and fail the stage.
                for arm in ([args.tag, args.tag + "_fixed", args.tag + "_cp_sat"] if args.mode == "llm" else [args.tag]):
                    for seed in (range(args.seeds) if args.mode == "llm" and not arm.endswith("_cp_sat") else [None]):
                        append(output, {**common, "arm": arm, "seed": seed, "feasible": False, "semantic_match": False,
                                        "error": repr(exc), "seconds": time.perf_counter() - started})
        print(f"EVAL {args.tag} {case['id']}", flush=True)


def summarize(directory):
    rows = []
    for path in directory.glob("*.jsonl"):
        if path.name in {"oracle.jsonl", "base.jsonl", "sft_eval.jsonl", "dpo_eval.jsonl", "direct.jsonl"}:
            rows.extend(read_rows(path))
    refs = {r["case_id"]: r for r in rows if r["arm"] == "cp_sat_gold"}
    groups = {}
    for r in rows:
        groups.setdefault((r["split"], r["family"], r["arm"]), []).append(r)
    summary = []
    for (split, family, arm), records in sorted(groups.items()):
        feasible = [r for r in records if r.get("feasible")]
        gaps = []
        for r in feasible:
            ref = refs.get(r["case_id"], {})
            if ref.get("status") == "OPTIMAL":
                gaps.append(abs(r["objective"] - ref["objective"]) / max(1, abs(ref["objective"])))
        summary.append({"split": split, "family": family, "arm": arm, "jobs": len(records),
                        "feasible_rate": len(feasible) / len(records), "errors": sum("error" in r for r in records),
                        "semantic_match_rate": sum(r.get("semantic_match", False) for r in records) / len(records) if any("semantic_match" in r for r in records) else None,
                        "mean_feasible_objective": float(np.mean([r["objective"] for r in feasible])) if feasible else None,
                        "mean_proven_optimal_gap": float(np.mean(gaps)) if gaps else None, "proven_gap_count": len(gaps),
                        "mean_search_or_direct_seconds": float(np.mean([r.get("seconds", 0) for r in records]))})
    # Case/seed matched comparisons. Never average raw objectives across families.
    bases = {(r["case_id"], r.get("seed")): r for r in rows if r["arm"] == "base"}
    paired = []
    for (split, family, arm), records in sorted(groups.items()):
        pairs = [(r, bases[(r["case_id"], r.get("seed"))]) for r in records if (r["case_id"], r.get("seed")) in bases]
        if not pairs or arm == "base":
            continue
        changes = []
        for r, base in pairs:
            if r.get("feasible") and base.get("feasible"):
                sign = -1 if r["objective_sense"] == "max" else 1
                changes.append(sign * (base["objective"] - r["objective"]) / max(1, abs(base["objective"])))
        paired.append({"split": split, "family": family, "arm": arm, "paired_jobs": len(pairs),
                       "became_feasible": sum(bool(r.get("feasible")) and not b.get("feasible") for r, b in pairs),
                       "lost_feasibility": sum(bool(b.get("feasible")) and not r.get("feasible") for r, b in pairs),
                       "both_feasible": len(changes), "mean_relative_improvement_on_both_feasible": float(np.mean(changes)) if changes else None})
    payload = {"scope": "synthetic binary linear-constraint quadratic-objective problems; NOT arbitrary NLP optimization",
               "energy_measurements": "not measured; software CPU sampler, no FPGA energy claim",
               "cost_note": "Generation logs contain physical calls/time/tokens. Translation metadata repeats across paired arms/seeds; do not sum it.",
               "rows": summary, "paired_vs_base": paired}
    (directory / "comparison.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    lines = ["# 通用能量闭环：实际实验汇总", "", "可行性由独立 gold IR 判定；均值只覆盖可行解，必须同时看可行率。不同实例目标规模不同，原始成本均值仅作描述。", "",
             "|划分|问题|方法|任务数|可行率|建模结构匹配率|可行目标均值|已证最优相对差距|", "|---|---|---|---:|---:|---:|---:|---:|"]
    fmt = lambda v: "—" if v is None else f"{v:.4f}"
    for r in summary:
        lines.append(f"|{r['split']}|{r['family']}|{r['arm']}|{r['jobs']}|{r['feasible_rate']:.1%}|{fmt(r['semantic_match_rate'])}|{fmt(r['mean_feasible_objective'])}|{fmt(r['mean_proven_optimal_gap'])}|")
    lines += ["", "软件采样未测硬件能耗，不代表 FPGA 实测。完整调用成本见 *_generation.jsonl。跨方法优劣必须按同实例/种子配对，不能只比较不同可行子集的均值。"]
    (directory / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return payload


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=["data", "teacher", "eval", "summary"])
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--per-family", type=int, default=48)
    p.add_argument("--holdout-family", default="balanced_cut", choices=["none", "knapsack", "vertex_cover", "balanced_cut", "assignment"])
    p.add_argument("--steps", type=int, default=4096)
    p.add_argument("--chains", type=int, default=64)
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--mode", choices=["oracle", "llm", "direct"], default="oracle")
    p.add_argument("--tag", default="oracle")
    p.add_argument("--model-path")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--adapters", nargs="*", default=[])
    p.add_argument("--max-new-tokens", type=int, default=4096)
    p.add_argument("--reference-seconds", type=float, default=10)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.stage == "data":
        target = args.output / "cases.jsonl"
        if target.exists():
            raise FileExistsError(target)
        write_rows(target, make_dataset(args.per_family))
    elif args.stage == "summary":
        summarize(args.output)
    else:
        cases = read_rows(args.output / "cases.jsonl")
        (teacher if args.stage == "teacher" else evaluate)(args, cases)


if __name__ == "__main__":
    main()
