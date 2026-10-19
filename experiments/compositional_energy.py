#!/usr/bin/env python3
"""Isolated sparse-v1 pilot: prepare data or replay saved model predictions.

Predictions JSONL: {case_id, response}; no gold supplied to inference. Existing
legacy pipelines are intentionally unchanged. All models must use sparse.prompt.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.generic_energy import read_rows, write_rows
from qihc.generic.compositional import dataset, training_rows
from qihc.generic.sparse import parse, prompt, VERSION
from qihc.generic.energy import solve, objective, violations
from qihc.generic.semantic import semantic_metrics


def replay(cases, predictions, steps=4096, chains=64, seeds=3):
    selected = {c["id"]: c for c in cases if c["split"] != "train"}
    ids = [r["case_id"] for r in predictions]
    if len(ids) != len(set(ids)) or set(ids) != set(selected):
        raise ValueError("Require exactly one prediction per evaluation case; no omissions or duplicates")
    rows = []
    for record in predictions:
        case = selected[record["case_id"]]
        base = {"case_id": case["id"], "split": case["split"], "size": case["size"], "rules": case["rules"]}
        try:
            spec = parse(record["response"])
            if spec["n"] != case["gold"]["n"]:
                raise ValueError("Wrong original variable count")
        except (ValueError, TypeError, KeyError) as exc:
            rows.append({**base, "valid": False, "error": str(exc)})
            continue
        metrics = semantic_metrics(spec, case["gold"])
        # Same starting policy, seeds, temperature and update budget. ONLY toggle feedback.
        for seed in range(seeds):
            for feedback in (False, True):
                result = solve(spec, policy="adaptive", feedback=feedback, seed=seed,
                               steps=steps, chains=chains)
                x = result["x"]
                rows.append({**base, **metrics, "valid": True, "feedback": feedback, "seed": seed,
                             "x": x, "feasible": bool((violations(case["gold"], x) == 0).all()),
                             "objective": float(objective(case["gold"], x))})
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("prepare", "replay", "controls", "direct-replay"))
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--count", type=int, default=120)
    p.add_argument("--predictions", type=Path)
    p.add_argument("--steps", type=int, default=4096)
    p.add_argument("--chains", type=int, default=64)
    p.add_argument("--seeds", type=int, default=3)
    a = p.parse_args()
    if a.stage == "prepare":
        a.output.mkdir(parents=True, exist_ok=False)
        cases = dataset(a.count)
        sft, dpo = training_rows(cases)
        for name, rows in (("cases", cases), ("sft", sft), ("dpo", dpo),
                           ("eval_prompts", [{"case_id": c["id"], "prompt": prompt(c["description"])}
                                             for c in cases if c["split"] != "train"] )):
            write_rows(a.output / f"{name}.jsonl", rows)
        (a.output / "protocol.json").write_text(json.dumps({"schema": VERSION, "count_per_split": a.count,
            "scope": "synthetic binary quadratic objectives with linear constraints",
            "test": "unseen rule triples and mixed in-distribution/larger sizes",
            "sft_rows": len(sft), "dpo_rows": len(dpo)}, indent=2), encoding="utf-8")
    elif a.stage == "controls":
        from qihc.generic.direct_control import matched_data, direct_prompt
        cases = read_rows(a.output / "cases.jsonl")
        target = a.output / "matched_controls"
        target.mkdir(exist_ok=False)
        result = matched_data(cases)
        for name, rows in result.items():
            if name != "budget":
                write_rows(target / f"{name}.jsonl", rows)
        write_rows(target / "direct_eval_prompts.jsonl", [
            {"case_id": c["id"], "prompt": direct_prompt(c["description"])}
            for c in cases if c["split"] != "train"])
        (target / "budget.json").write_text(json.dumps(result["budget"], indent=2), encoding="utf-8")
    else:
        if not a.predictions:
            p.error("replay requires --predictions")
        if min(a.steps, a.chains, a.seeds) < 1:
            p.error("search budgets must be positive")
        target = a.output / (a.predictions.stem + "_search.jsonl")
        if target.exists():
            raise FileExistsError(target)
        cases, predictions = read_rows(a.output / "cases.jsonl"), read_rows(a.predictions)
        if a.stage == "direct-replay":
            from qihc.generic.direct_control import evaluate_direct
            rows = evaluate_direct(cases, predictions)
        else:
            rows = replay(cases, predictions, a.steps, a.chains, a.seeds)
        write_rows(target, rows)


if __name__ == "__main__":
    main()
