#!/usr/bin/env python3
"""One-device, offline generic-energy benchmark and real SFT -> DPO pipeline."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.generic_energy import read_rows, summarize, audit_teacher


def check_training_output(target):
    manifest = json.loads((target / "qihc_training_manifest.json").read_text(encoding="utf-8"))
    weights = target / "adapter_model.safetensors"
    losses = [float(row[k]) for row in manifest.get("training_metrics", []) for k in ("loss", "train_loss") if k in row]
    if not weights.is_file() or weights.stat().st_size == 0 or not losses or not all(math.isfinite(x) for x in losses):
        raise RuntimeError(f"Missing weights or missing/nonfinite training loss: {target}")
    return weights.stat().st_size


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--per-family", type=int, default=48)
    p.add_argument("--steps", type=int, default=4096)
    p.add_argument("--chains", type=int, default=64)
    p.add_argument("--hours", type=float, default=36)
    p.add_argument("--cpu-smoke", action="store_true")
    args = p.parse_args()
    if not math.isfinite(args.hours) or args.hours <= 0 or args.per_family < 8 or not 1 <= args.steps <= 1000000 or not 1 <= args.chains <= 4096:
        p.error("Positive hours and >=8 cases/family required")
    args.output = args.output.resolve()
    if args.output.exists():
        raise FileExistsError("Use a NEW output directory; never mix runs")
    args.output.mkdir(parents=True)
    started = time.monotonic()
    report = {"status": "running", "stages": [], "config": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
              "holdout_family": "balanced_cut", "training_scope": "gold-IR SFT + pbit-execution policy DPO; no generic GRPO",
              "sampler": "CPU float64 single-site heat bath, no FPGA/energy measurement"}
    try:
        report["git_commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        report["git_commit"] = "unavailable"

    def save():
        report["elapsed_seconds"] = time.monotonic() - started
        pending = args.output / "run.pending.json"
        pending.write_text(json.dumps(report, indent=2), encoding="utf-8")
        pending.replace(args.output / "run.json")

    def run(label, command):
        remaining = args.hours * 3600 - (time.monotonic() - started)
        if remaining <= 0:
            raise TimeoutError("Experiment wall-clock budget exhausted")
        record = {"stage": label, "command": command, "status": "running"}
        report["stages"].append(record)
        save()
        print(f"RUN {label}", flush=True)
        t = time.monotonic()
        try:
            with (args.output / f"{label}.log").open("w", encoding="utf-8") as log:
                subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=remaining,
                               env={**os.environ, "PYTHONPATH": str(ROOT) + os.pathsep + os.environ.get("PYTHONPATH", "")})
        except Exception as exc:
            record.update(status="failed", seconds=time.monotonic() - t, error=repr(exc))
            save()
            print(f"FAILED {label}; inspect {args.output / (label + '.log')}", flush=True)
            raise
        record.update(status="complete", seconds=time.monotonic() - t)
        save()

    steps, chains = (128, 16) if args.cpu_smoke else (args.steps, args.chains)
    report["effective_search_budget"] = {"steps": steps, "chains": chains, "seeds": 1 if args.cpu_smoke else 3}
    prefix = [sys.executable, "experiments/generic_energy.py"]
    common = ["--output", str(args.output), "--steps", str(steps), "--chains", str(chains)]
    try:
        save()
        if not args.cpu_smoke:
            if args.manifest is None:
                raise ValueError("Prepared manifest required; no automatic downloads")
            model = json.loads(args.manifest.read_text(encoding="utf-8"))["model_path"]
            if not Path(model).is_dir():
                raise FileNotFoundError(f"Prepared model directory unavailable: {model}")
            report["prepared_manifest_sha256"] = hashlib.sha256(args.manifest.read_bytes()).hexdigest()
            report["model_path"] = model
        run("data", prefix + ["data", "--per-family", str(8 if args.cpu_smoke else args.per_family)] + common)
        report["dataset_sha256"] = hashlib.sha256((args.output / "cases.jsonl").read_bytes()).hexdigest()
        run("oracle", prefix + ["eval", "--mode", "oracle", "--tag", "oracle", "--seeds", "1" if args.cpu_smoke else "3"] + common)
        if not args.cpu_smoke:
            run("teacher", prefix + ["teacher"] + common)
            report["teacher_audit"] = audit_teacher(args.output, read_rows(args.output / "cases.jsonl"), "balanced_cut")
            run("training_preflight", prefix + ["preflight", "--model-path", model, "--max-new-tokens", "12288"] + common)
            inference = ["--model-path", model, "--max-new-tokens", "12288"] + common
            run("base", prefix + ["eval", "--mode", "llm", "--tag", "base"] + inference)
            run("direct", prefix + ["eval", "--mode", "direct", "--tag", "direct"] + inference)
            adapters = []
            for stage, cap, lr in (("sft", 128, "1e-5"), ("dpo", 64, "1e-6")):
                data = args.output / f"{stage}.jsonl"
                count = len(read_rows(data))
                report[f"{stage}_rows"] = count
                if stage == "dpo" and count < 16:
                    report["dpo_skipped"] = "Fewer than 16 independently confirmed execution preferences; no fabricated pairs"
                    break
                if count == 0:
                    raise ValueError("Empty training data")
                target = args.output / f"adapter_{stage}"
                command = [sys.executable, "experiments/train_s2e_lora.py", "--stage", stage, "--model-path", model,
                           "--data", str(data), "--output", str(target), "--max-steps", str(min(cap, max(1, math.ceil(count / 4)))),
                           "--learning-rate", lr, "--max-length", "16384", "--gradient-accumulation-steps", "4", "--save-steps", "100",
                           "--lora-rank", "16", "--precision", "bf16"]
                if stage == "dpo":
                    command += ["--adapter-path", adapters[0], "--dpo-beta", "0.2", "--dpo-sft-weight", "0.2"]
                run(f"train_{stage}", command)
                weight_bytes = check_training_output(target)
                report[f"{stage}_training"] = {"rows": count, "weights_bytes": weight_bytes,
                                              "manifest": str(target / "qihc_training_manifest.json")}
                adapters.append(str(target))
                # Trainer continues the existing adapter, not a delta on merged SFT.
                run(f"{stage}_eval", prefix + ["eval", "--mode", "llm", "--tag", f"{stage}_eval"] + inference + ["--adapters", adapters[-1]])
            report["status"] = "complete_without_dpo" if "dpo_skipped" in report else "complete"
        else:
            report["status"] = "cpu_smoke_complete_no_llm_training"
    except Exception as exc:
        report.update(status="failed", error=repr(exc))
        raise
    finally:
        save()
        try:
            summarize(args.output)
        except Exception as exc:
            already_failed = report["status"] == "failed"
            report.update(status="failed", summary_error=repr(exc))
            save()
            if not already_failed:
                raise
        print(f"Results: {args.output}", flush=True)


if __name__ == "__main__":
    main()
