"""Bounded p-bit teacher -> SFT -> DPO study on held-out medium CVRP instances."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.server_readiness import atomic_json, code_hash


def compare(before, after):
    """Report feasibility separately, never silently score only successful jobs."""
    index = lambda rows: {(r["instance"], int(r["search_seed"])): r for r in rows}
    b, a = index(before), index(after)
    feasible = lambda r: r.get("status") == "ok" and r.get("reference_feasible", False)
    keys = sorted(b.keys() & a.keys())
    paired = [k for k in keys if feasible(b[k]) and feasible(a[k])]
    gains = [100 * (b[k]["final_objective"] - a[k]["final_objective"]) / b[k]["final_objective"]
             for k in paired if b[k]["final_objective"] > 0]
    return {"before_jobs": len(b), "after_jobs": len(a), "matched_jobs": len(keys),
            "missing_after": len(b.keys() - a.keys()), "missing_before": len(a.keys() - b.keys()),
            "before_feasible": sum(feasible(r) for r in b.values()),
            "after_feasible": sum(feasible(r) for r in a.values()), "paired_feasible": len(paired),
            "mean_paired_cost_before": sum(b[k]["final_objective"] for k in paired) / len(paired) if paired else None,
            "mean_paired_cost_after": sum(a[k]["final_objective"] for k in paired) / len(paired) if paired else None,
            "mean_paired_cost_improvement_percent": sum(gains) / len(gains) if gains else None,
            "wins": sum(a[k]["final_objective"] < b[k]["final_objective"] - 1e-9 for k in paired),
            "losses": sum(a[k]["final_objective"] > b[k]["final_objective"] + 1e-9 for k in paired),
            "pairs": [{"instance": k[0], "seed": k[1], "before_cost": b[k]["final_objective"],
                       "after_cost": a[k]["final_objective"]} for k in paired]}


def audit_summary(directory):
    rows = [json.loads(line) for p in directory.glob("llm_audit_rank*.jsonl")
            for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
    return {"calls": len(rows), "successful_calls": sum(bool(r.get("ok")) for r in rows),
            "token_limit_calls": sum(bool(r.get("generation", {}).get("reached_token_limit")) for r in rows)}


def select_training(source, target, tokenizer, stage, max_length=16384):
    rows = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line]
    kept = []
    for row in rows:
        prompt = tokenizer.apply_chat_template([{"role": "user", "content": row["prompt"]}],
                                               tokenize=False, add_generation_prompt=True, enable_thinking=False)
        answers = [row[k] for k in ("completion", "chosen", "rejected") if k in row]
        length = len(tokenizer.encode(prompt, add_special_tokens=False)) + 1 + max(
            len(tokenizer.encode(a, add_special_tokens=False)) for a in answers)
        if length <= max_length:
            kept.append(row)
    target.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in kept), encoding="utf-8")
    return {"stage": stage, "input_rows": len(rows), "kept_rows": len(kept),
            "excluded_over_context": len(rows) - len(kept),
            "proposal_rows": sum(r.get("task") == "proposal" for r in kept), "max_length": max_length}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--hours", type=float, default=8)
    args = p.parse_args()
    if not 0 < args.hours <= 48:
        p.error("--hours must be in (0,48]")
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    (root / "logs").mkdir()
    model = json.loads(args.manifest.read_text(encoding="utf-8"))["model_path"]
    deadline = time.monotonic() + args.hours * 3600
    report = {"status": "running", "scope": "p-bit-guided neighborhood post-training, SFT then DPO; no GRPO or LLM cold-start comparison",
              "model": model, "hours_cap": args.hours, "sizes": [100, 200, 300], "instances_per_size": 18,
              "seeds": [0, 1], "sft_steps": 60, "dpo_steps": 30, "train_max_length": 16384,
              "teacher_iterations": 4, "evaluation_iterations": 3,
              "source_sha256": code_hash(ROOT), "driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "stages": {}, "comparisons": {}, "limitations": "Synthetic constrained CVRP; preliminary single-training-seed evidence, not a full benchmark conclusion"}

    def save():
        atomic_json(root / "summary.json", report)

    def run(name, command):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("Overall time budget exhausted")
        print(f"RUN {name}", flush=True)
        report["current_stage"] = name
        save()
        started = time.monotonic()
        logfile = root / "logs" / (name + ".log")
        with logfile.open("w", encoding="utf-8") as handle:
            proc = subprocess.Popen(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT,
                                    start_new_session=os.name != "nt")
            try:
                code = proc.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                if os.name != "nt":
                    os.killpg(proc.pid, signal.SIGTERM)
                else:
                    proc.terminate()
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    if os.name != "nt":
                        os.killpg(proc.pid, signal.SIGKILL)
                    else:
                        proc.kill()
                    proc.wait()
                raise TimeoutError(f"Time budget reached during {name}; prior results retained")
        report["stages"][name] = {"exit_code": code, "elapsed_s": time.monotonic() - started, "command": command}
        save()
        if code:
            print(logfile.read_text(encoding="utf-8", errors="replace")[-10000:], flush=True)
            raise RuntimeError(f"Stage {name} failed; see {logfile}")

    def python(script, *params):
        return [sys.executable, script, *map(str, params)]

    data = root / "data"
    common = ["--dataset", "jsonl", "--sampler", "torch", "--search-backend", "pbit",
              "--num-chains", "512", "--sampling-steps", "80", "--top-samples", "64",
              "--destroy-size", "12", "--routes-per-customer", "4", "--feedback-mode", "population",
              "--block-updates", "--constraint-aware-decode", "--temperature-ladder", "--normalize-objective",
              "--initialization", "pbit-cold", "--cold-selector-mode", "neutral", "--cold-batch-size", "8",
              "--cold-routes-per-customer", "4", "--constraint-source", "cpp", "--cpp-records", str(data / "oracle_cpp.jsonl"),
              "--patience", "10", "--search-time-limit", "0", "--backend-time-limit", "2",
              "--proposal-bias", "0.25"]
    llm = ["--methods", "llm", "--model-path", model, "--llm-temperature", "0",
           "--llm-refresh-interval", "1", "--llm-max-new-tokens", "1536", "--llm-max-input-tokens", "16384"]
    initials = root / "shared_initials.jsonl"
    try:
        run("preflight", python("experiments/check_research_environment.py", "--model-path", model,
                                 "--nproc", 1, "--training", "--output", root / "environment.json"))
        run("data", python("experiments/prepare_research_data.py", "--output", data,
                           "--sizes", 100, 200, 300, "--instances-per-size", 18, "--clauses-per-type", 2))
        report["split"] = json.loads((data / "split_manifest.json").read_text())["splits"]
        run("shared_initial", python("experiments/run_cvrp_lns.py", *common, "--methods", "knn",
             "--data", data / "all.jsonl", "--search-seeds", 0, 1, "--iterations", 0, "--output", root / "shared_initial"))
        run("export_initial", python("experiments/export_shared_initials.py", "--results",
             root / "shared_initial/results.json", "--output", initials))

        def evaluate(name, adapter=None):
            directory = root / name
            command = python("experiments/run_cvrp_lns.py", *common, *llm, "--data", data / "test.jsonl",
                             "--initial-solutions", initials, "--search-seeds", 0, 1, "--iterations", 3, "--output", directory)
            if adapter:
                command += ["--adapter-path", str(adapter)]
            run(name, command)
            report.setdefault("llm_audits", {})[name] = audit_summary(directory)
            if name != "base":
                report["comparisons"][name] = compare(json.loads((root / "base/results.json").read_text()),
                                                     json.loads((directory / "results.json").read_text()))
                report["comparisons"][name]["quality_gate_passed"] = False
                save()  # Preserve useful results even if a later stage fails.
            run(name + "_check", python("experiments/check_model_smoke.py", "--root", directory))
            if name != "base":
                report["comparisons"][name]["quality_gate_passed"] = True
            save()

        evaluate("base")
        run("teacher", python("experiments/run_cvrp_lns.py", *common, *llm, "--data", data / "train.jsonl",
            "--initial-solutions", initials, "--search-seeds", 0, "--iterations", 4,
            "--export-training-context", "--output", root / "teacher"))
        run("teacher_check", python("experiments/check_model_smoke.py", "--root", root / "teacher"))
        training = root / "training_data"
        run("training_data", python("experiments/prepare_joint_training_data.py", "--data", data / "train.jsonl",
            "--constraint-records", data / "oracle_cpp.jsonl", "--solver-results", root / "teacher/results.json",
            "--llm-audits", root / "teacher", "--verify-preferences", "--output", training))
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
        report["training_data"] = json.loads((training / "summary.json").read_text())
        report["length_filter"] = {}
        previous = None
        for stage, steps, lr in [("sft", 60, "2e-5"), ("dpo", 30, "5e-6")]:
            selected = training / f"{stage}_bounded.jsonl"
            counts = select_training(training / f"{stage}.jsonl", selected, tokenizer, stage)
            report["length_filter"][stage] = counts
            save()
            if stage == "sft" and (counts["kept_rows"] < 60 or counts["proposal_rows"] < 24):
                raise RuntimeError(f"Insufficient p-bit teacher supervision: {counts}")
            if stage == "dpo" and counts["proposal_rows"] < 16:
                report["dpo_skipped"] = "Fewer than 16 executed, verified proposal preference pairs"
                break
            adapter = root / "adapters" / stage
            command = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=1",
                       "experiments/train_s2e_lora.py", "--stage", stage, "--model-path", model,
                       "--data", str(selected), "--output", str(adapter), "--max-steps", str(steps),
                       "--max-length", "16384", "--gradient-accumulation-steps", "4", "--learning-rate", lr,
                       "--save-steps", "20", "--deepspeed-config", str(ROOT / "configs/deepspeed-zero3-bf16.json")]
            if previous:
                command += ["--adapter-path", str(previous)]
            run("train_" + stage, command)
            evaluate(stage, adapter)
            previous = adapter
        report["status"] = "sft_complete_dpo_skipped" if report.get("dpo_skipped") else "complete"
    except TimeoutError as exc:
        report.update(status="time_budget_reached", error=str(exc))
    except Exception as exc:
        report.update(status="failed", error=repr(exc))
        raise
    finally:
        save()
        print(f"RESULTS: {root / 'summary.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
