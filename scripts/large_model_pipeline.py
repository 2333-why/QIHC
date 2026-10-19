#!/usr/bin/env python3
"""Separate four-GPU sparse modeling/direct-control protocol, no legacy reseal."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.server_readiness import (download_model, digest, check_files, atomic_json,
                                          package_versions, check_model_headers)
from experiments.generic_energy import read_rows, write_rows, append
from qihc.generic.compositional import dataset
from qihc.generic.direct_control import matched_data, direct_prompt, evaluate_direct
from qihc.generic.sparse import prompt, parse
from qihc.generic.semantic import semantic_metrics
from qihc.generic.energy import solve, objective, violations
from qihc.generic.benchmark import reference_solve
from scripts.generic_energy_pipeline import training_command, check_training_output

REPO = "Qwen/Qwen3-32B"
PACKAGES = ("torch", "transformers", "trl", "peft", "accelerate", "deepspeed", "datasets", "ortools")


def sources():
    paths = sorted({p for folder in ("qihc", "experiments", "scripts", "configs")
                    for p in (ROOT / folder).rglob("*") if p.suffix in (".py", ".sh", ".json")})
    h = hashlib.sha256()
    for p in paths:
        h.update(p.relative_to(ROOT).as_posix().encode())
        h.update(p.read_bytes().replace(b"\r\n", b"\n"))
    return h.hexdigest()


def preflight(directory, model):
    from transformers import AutoConfig, AutoTokenizer
    cfg = AutoConfig.from_pretrained(model, local_files_only=True)
    if cfg.model_type != "qwen3":
        raise ValueError("This protocol is validated for dense Qwen3 only")
    tok = AutoTokenizer.from_pretrained(model, local_files_only=True)
    report = {}
    for arm in ("modeling", "direct"):
        for stage in ("sft", "dpo"):
            name = f"{arm}_{stage}"
            rows = read_rows(directory / f"{name}.jsonl")
            if not rows:
                raise ValueError(f"Empty training set: {name}")
            lengths = []
            for r in rows:
                rendered = tok.apply_chat_template([{"role": "user", "content": r["prompt"]}],
                    tokenize=False, add_generation_prompt=True, enable_thinking=False)
                answers = [r["completion"]] if stage == "sft" else [r["chosen"], r["rejected"]]
                length = max(len(tok.encode(rendered + a + tok.eos_token, add_special_tokens=False)) for a in answers)
                if length > 8192 or any(len(tok.encode(a, add_special_tokens=False)) > 4096 for a in answers):
                    raise ValueError(f"Training/output limit exceeded: {name} {r['case_id']}")
                lengths.append(length)
            report[name] = {"rows": len(rows), "max_tokens": max(lengths),
                            "dataset_sequence_tokens": sum(lengths), "longest_row": lengths.index(max(lengths))}
    return report


def prepare(a):
    dest = a.base / "artifacts/large32b"
    dest.mkdir(parents=True, exist_ok=True)
    sealed = dest / "prepared.json"
    if sealed.exists():
        previous = json.loads(sealed.read_text())
        if previous["source"] != sources() or previous["count_per_split"] != a.count:
            raise RuntimeError("Existing preparation differs; preserve it and choose a separate artifact directory/protocol")
        check_files(Path(previous["model"]["model_path"]), previous["model"]["files"])
        print(f"Already prepared: {sealed}. Use a different QIHC_BASE for a new protocol.")
        return
    versions = package_versions()
    if any(p not in versions for p in PACKAGES):
        raise ValueError("Existing QIHC training environment incomplete; run original environment setup first")
    model = a.base / "models/Qwen3-32B"
    mm = dest / "model.json"
    if not mm.exists():
        download_model(SimpleNamespace(repo=REPO, revision="main", model_path=model, output=mm))
    meta = json.loads(mm.read_text())
    check_files(model, meta["files"])
    check_model_headers(model)
    data = dest / "data"
    data.mkdir(exist_ok=True)
    cases = dataset(a.count)
    write_rows(data / "cases.jsonl", cases)
    matched = matched_data(cases)
    for name, rows in matched.items():
        if name != "budget":
            write_rows(data / f"{name}.jsonl", rows)
    atomic_json(data / "budget.json", matched["budget"])
    report = preflight(data, str(model))
    # Longest real examples, not a tiny toy, are exercised on the GPU before any full training.
    for stage in ("sft", "dpo"):
        name = max((f"{arm}_{stage}" for arm in ("modeling", "direct")), key=lambda k: report[k]["max_tokens"])
        row = read_rows(data / f"{name}.jsonl")[report[name]["longest_row"]]
        write_rows(data / f"smoke_{stage}.jsonl", [row] * 4)
    atomic_json(dest / "preflight.json", report)
    subprocess.run([sys.executable, "-m", "pytest", "tests/test_compositional_energy.py",
                    "tests/test_direct_control.py", "tests/test_zero_runtime.py", "-q"], cwd=ROOT, check=True)
    atomic_json(sealed, {"protocol": "large32b-matched-v1", "model": meta, "source": sources(),
        "packages": {k: versions[k] for k in PACKAGES}, "env_prefix": sys.prefix,
        "data": {p.name: digest(p) for p in data.glob("*.json*")}, "count_per_split": a.count,
        "training": {"sft_steps": 192, "dpo_steps": 128, "world_size": 4, "global_batch": 4,
                     "context": 8192, "max_new_tokens": 4096},
        "scope": "row/update-matched; not token-matched; pbit CPU emulator; synthetic composition"})
    print(f"CPU PREPARATION COMPLETE: {sealed}. Full-model GPU checks remain.", flush=True)


def infer(a):
    import torch
    from qihc.problems.cvrp.direct_llm import LocalDirectLLMGenerator
    gen = LocalDirectLLMGenerator(str(a.model), device="auto", temperature=0,
                                 max_input_tokens=4096, max_new_tokens=4096)
    if a.adapter:
        from peft import PeftModel
        # Keep adapter unmerged: avoids temporary full-matrix copies during merge.
        gen.model = PeftModel.from_pretrained(gen.model, str(a.adapter), local_files_only=True)
        gen.model.eval()
    records = read_rows(a.output) if a.output.exists() else []
    cases = [c for c in read_rows(a.data / "cases.jsonl") if c["split"] != "train"]
    done = {r["case_id"] for r in records}
    if len(done) != len(records) or not done.issubset({c["id"] for c in cases}):
        raise ValueError("Invalid prediction checkpoint")
    for c in cases:
        if c["id"] in done:
            continue
        p = direct_prompt(c["description"]) if a.arm == "direct" else prompt(c["description"])
        start = time.perf_counter()
        raw = gen.generate(p)
        torch.cuda.synchronize()
        append(a.output, {"case_id": c["id"], "response": raw, **gen.last_usage,
                          "seconds": time.perf_counter() - start})
        print(f"PREDICT {a.arm} {c['id']}", flush=True)


def score(a):
    cases = [c for c in read_rows(a.data / "cases.jsonl") if c["split"] != "train"]
    preds = read_rows(a.predictions)
    ids = [r["case_id"] for r in preds]
    if len(set(ids)) != len(ids) or set(ids) != {c["id"] for c in cases}:
        raise ValueError("Incomplete/duplicate prediction set")
    if a.arm == "direct":
        write_rows(a.output, [{**r, "backend": "direct"} for r in evaluate_direct(cases, preds)])
        return
    predicted = {r["case_id"]: r["response"] for r in preds}
    rows = read_rows(a.output) if a.output.exists() else []
    done = {r["case_id"] for r in rows}
    for c in cases:
        if c["id"] in done:
            continue
        batch = []
        common = {"case_id": c["id"], "split": c["split"], "size": c["size"]}
        try:
            spec = parse(predicted[c["id"]])
            if spec["n"] != c["gold"]["n"]:
                raise ValueError("Wrong dimension")
        except (ValueError, TypeError, KeyError) as exc:
            rows.extend({**common, "backend": b, "valid": False, "feasible": False, "error": str(exc)}
                        for b in ("pbit", "no_feedback", "metropolis", "cp_sat"))
            write_rows(a.output, rows)
            continue
        for backend in ("pbit", "no_feedback", "metropolis", "cp_sat"):
            for seed in ([0] if backend == "cp_sat" else range(3)):
                start = time.perf_counter()
                result = reference_solve(spec, 10) if backend == "cp_sat" else solve(spec, policy="adaptive",
                    seed=seed, steps=4096, chains=64, method="metropolis" if backend == "metropolis" else "pbit",
                    feedback=backend != "no_feedback")
                x = result["x"]
                batch.append({**common, **semantic_metrics(spec, c["gold"]), "backend": backend,
                    "seed": seed, "valid": True, "x": x,
                    "feasible": x is not None and bool((violations(c["gold"], x) == 0).all()),
                    "objective": float(objective(c["gold"], x)) if x is not None else None,
                    "search_seconds": time.perf_counter() - start})
        rows.extend(batch)
        write_rows(a.output, rows)  # Atomic complete-case checkpoint.


def run(a):
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() != 4:
        raise RuntimeError("Allocate exactly four visible H100 80GB on one worker")
    for i in range(4):
        free, total = torch.cuda.mem_get_info(i)
        if "H100" not in torch.cuda.get_device_name(i) or total < 75 * 2**30 or free < 70 * 2**30:
            raise RuntimeError("Requires four H100 80GB, each with >=70 GiB free")
    prepared = a.base / "artifacts/large32b"
    manifest = json.loads((prepared / "prepared.json").read_text())
    if manifest["source"] != sources() or manifest["env_prefix"] != sys.prefix:
        raise RuntimeError("Source/environment path changed since CPU preparation; rerun preparation in a new directory")
    versions = package_versions()
    if any(versions.get(k) != v for k, v in manifest["packages"].items()):
        raise RuntimeError("Environment versions changed since preparation")
    data = prepared / "data"
    for name, checksum in manifest["data"].items():
        if digest(data / name) != checksum:
            raise RuntimeError(f"Prepared data changed: {name}")
    model = manifest["model"]["model_path"]
    check_files(Path(model), manifest["model"]["files"])
    out = a.output or a.base / "results" / ("large32b_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + f"-{os.getpid()}")
    out.mkdir(parents=True, exist_ok=True)
    statepath = out / "run.json"
    report = json.loads(statepath.read_text()) if statepath.exists() else {"source": manifest["source"],
        "prepared_sha256": digest(prepared / "prepared.json"), "stages": {}, "status": "running"}
    if report["prepared_sha256"] != digest(prepared / "prepared.json"):
        raise RuntimeError("Cannot resume with different preparation")
    deadline = time.monotonic() + 96 * 3600
    def launch(name, cmd):
        if report["stages"].get(name, {}).get("status") == "complete":
            return
        print(f"RUN {name} -> {out}", flush=True)
        report["stages"][name] = {"status": "running"}
        atomic_json(statepath, report)
        start = time.perf_counter()
        with (out / f"{name}.log").open("a", encoding="utf-8") as log:
            subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True,
                           timeout=max(1, deadline - time.monotonic()))
        report["stages"][name] = {"status": "complete", "seconds": time.perf_counter() - start}
        atomic_json(statepath, report)
    def train(label, arm, stage, steps, adapter=None):
        target = out / label
        source = data / (f"smoke_{stage}.jsonl" if arm == "smoke" else f"{arm}_{stage}.jsonl")
        cmd = training_command(model, source, target, stage, steps, 4, adapter)
        cmd[cmd.index("--max-length") + 1] = "8192"
        if target.exists():
            cmd += ["--resume"]
        launch(label, cmd)
        check_training_output(target)
        return target
    def evaluation(label, arm, adapter=None):
        prediction = out / f"{label}_predictions.jsonl"
        cmd = [sys.executable, __file__, "infer", "--model", model, "--data", str(data),
               "--output", str(prediction), "--arm", arm]
        if adapter:
            cmd += ["--adapter", str(adapter)]
        launch(label + "_infer", cmd)
        launch(label + "_score", [sys.executable, __file__, "score", "--data", str(data),
            "--predictions", str(prediction), "--output", str(out / f"{label}_scores.jsonl"), "--arm", arm])
    try:
        launch("nccl", [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=4",
            "experiments/server_readiness.py", "distributed-smoke", "--deepspeed-config",
            "configs/deepspeed-zero3-bf16.json", "--output", str(out / "nccl.json")])
        smoke = train("smoke_sft", "smoke", "sft", 1)
        train("smoke_dpo", "smoke", "dpo", 1, smoke)
        for arm in ("direct", "modeling"):
            evaluation(f"{arm}_base", arm)
            sft = train(f"{arm}_sft", arm, "sft", 192)
            evaluation(f"{arm}_sft", arm, sft)
            dpo = train(f"{arm}_dpo", arm, "dpo", 128, sft)
            evaluation(f"{arm}_dpo", arm, dpo)
        report["status"] = "complete"
    except Exception as exc:
        report.update(status="failed", error=repr(exc))
        print(f"FAILED. Logs and resume directory: {out}", flush=True)
        raise
    finally:
        atomic_json(statepath, report)
        summarize(out)
        print(f"Results: {out}", flush=True)


def summarize(out):
    """Macro average seeds within each case; never average unlike objectives."""
    summary = {}
    for file in sorted(out.glob("*_scores.jsonl")):
        rows = read_rows(file)
        groups = {}
        for r in rows:
            key = (r["backend"], r["split"])
            groups.setdefault(key, {}).setdefault(r["case_id"], []).append(r)
        for (backend, split), cases in groups.items():
            summary[f"{file.stem}/{backend}/{split}"] = {
                "completed_cases": len(cases),
                "valid_rate": sum(all(r["valid"] for r in rs) for rs in cases.values()) / len(cases),
                "feasible_rate": sum(sum(r["feasible"] for r in rs) / len(rs) for rs in cases.values()) / len(cases)}
    atomic_json(out / "summary.json", {"metrics": summary,
        "warning": "Partial files are partial results. See run.json for completeness. Costs in prediction/stage logs; raw objectives require paired-case comparison."})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="stage", required=True)
    for stage in ("prepare", "run"):
        s = sub.add_parser(stage)
        s.add_argument("--base", type=Path, required=True)
        s.add_argument("--count", type=int, default=240)
        s.add_argument("--output", type=Path)
    for stage in ("infer", "score"):
        s = sub.add_parser(stage)
        s.add_argument("--data", type=Path, required=True)
        s.add_argument("--output", type=Path, required=True)
        s.add_argument("--arm", choices=("direct", "modeling"), required=True)
        if stage == "infer":
            s.add_argument("--model", type=Path, required=True)
            s.add_argument("--adapter", type=Path)
        else:
            s.add_argument("--predictions", type=Path, required=True)
    a = p.parse_args()
    {"prepare": prepare, "run": run, "infer": infer, "score": score}[a.stage](a)


if __name__ == "__main__":
    main()
