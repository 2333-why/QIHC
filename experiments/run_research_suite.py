#!/usr/bin/env python3
"""One-command research suite: paired controls, execution training, held-out evaluation.

Run --profile smoke locally, or --profile formal --model-path ... on a GPU node.
Each stage has a log and a completion marker. Changed source/config cannot reuse
an old suite directory. No Bash installation is required.
"""

from __future__ import annotations

import argparse
import atexit
import hashlib
import importlib.util
from importlib.metadata import version, PackageNotFoundError
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import socket
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


ARMS = {
    "population": [],
    "no_feedback": ["--disable-logit-feedback"],
    "legacy": ["--feedback-mode", "legacy"],
    "best_feedback": ["--feedback-mode", "best"],
    "elite_selected": ["--feedback-source", "elite"],
    "no_trust_region": ["--disable-feedback-trust-region"],
    "no_pair_information": ["--disable-feedback-pairs"],
    "no_prompt": ["--disable-prompt-feedback"],
    "no_token": ["--disable-token-feedback"],
    "no_energy": ["--disable-energy-feedback"],
    "knn": ["--methods", "knn", "--disable-logit-feedback"],
    "random_neighborhood": ["--methods", "random", "--disable-logit-feedback"],
    "pbit_qubo": ["--search-backend", "pbit-qubo"],
    "cp_sat_pool": ["--search-backend", "cp-sat"],
    "metropolis_pool": ["--search-backend", "metropolis"],
    "uniform_pool": ["--search-backend", "uniform"],
    "single_site_pdit": ["--disable-block-updates"],
    "no_semantic_repair": ["--disable-semantic-repair"],
    "single_temperature": ["--disable-temperature-ladder"],
    "raw_objective_scale": ["--disable-objective-normalization"],
    "strict_decode": ["--strict-assignment-decode"],
    "shuffled_quality": ["--shuffle-feedback-quality"],
}


def atomic_json(path: Path, value: dict | list) -> None:
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def model_fingerprint(directory: Path | None, *, hash_weights: bool) -> dict:
    if directory is None or not directory.exists():
        return {}
    files = sorted({*directory.glob("*.safetensors"), *directory.glob("*.bin"), *directory.glob("*.json"),
                    *directory.glob("*.model"), *directory.glob("*.jinja")})
    result = {}
    for path in files:
        weight = path.suffix in {".safetensors", ".bin"}
        if weight and hash_weights:
            print(f"HASH MODEL {path.name}", flush=True)
        stat = path.stat()
        result[path.name] = {"size": stat.st_size,
                             "sha256": file_digest(path) if not weight or hash_weights else None,
                             "mtime_ns": stat.st_mtime_ns if weight and not hash_weights else None}
    return result


class SuiteLock:
    def __init__(self, root: Path):
        self.path = root / ".running.lock"
        self.token = uuid.uuid4().hex
        if self.path.exists():
            import psutil
            previous = json.loads(self.path.read_text(encoding="utf-8"))
            if previous.get("host") != socket.gethostname() or psutil.pid_exists(previous.get("pid", -1)):
                raise RuntimeError(f"Another suite owns {self.path}: {previous}")
            self.path.unlink()  # Only our stale, host-local lock file is removed.
        with self.path.open("x", encoding="utf-8") as handle:
            json.dump({"host": socket.gethostname(), "pid": os.getpid(), "token": self.token}, handle)
        atexit.register(self.release)

    def release(self):
        if self.path.exists() and json.loads(self.path.read_text(encoding="utf-8")).get("token") == self.token:
            self.path.unlink()


def source_fingerprint() -> str:
    digest = hashlib.sha256()
    files = sorted((ROOT / "qihc").rglob("*.py")) + sorted((ROOT / "experiments").glob("*.py"))
    for path in files:
        digest.update(str(path.relative_to(ROOT)).encode()); digest.update(path.read_bytes())
    return digest.hexdigest()


class StageRunner:
    def __init__(self, root: Path, dry_run: bool):
        self.root, self.dry_run, self.plan = root, dry_run, []
        (root / "logs").mkdir(parents=True, exist_ok=True)
        (root / "stages").mkdir(exist_ok=True)

    def run(self, name: str, command: list[str], artifact: Path | None = None, required_artifacts: list[Path] | None = None) -> None:
        marker = self.root / "stages" / f"{name}.json"
        artifacts = list(dict.fromkeys(([artifact] if artifact else []) + (required_artifacts or [])))
        self.plan.append({"stage": name, "command": command, "artifact": str(artifact) if artifact else None})
        atomic_json(self.root / ("planned_execution.json" if self.dry_run else "execution_plan.json"), self.plan)
        if self.dry_run:
            print(f"PLAN {name}", flush=True); return
        if marker.exists():
            try:
                saved = json.loads(marker.read_text(encoding="utf-8"))
                saved_artifacts = [Path(path) for path in saved.get("artifact_sha256", {})]
                checks = list(dict.fromkeys(artifacts + saved_artifacts))
                valid = (saved.get("completed") and saved.get("command") == command
                         and all(path.is_file() and saved.get("artifact_sha256", {}).get(str(path)) == file_digest(path) for path in checks))
            except (ValueError, OSError):
                valid = False
            if valid:
                print(f"RESUME {name}", flush=True); return
        print(f"RUN {name}", flush=True)
        started = time.time()
        with (self.root / "logs" / f"{name}.log").open("w", encoding="utf-8") as handle:
            environment = {**os.environ, "PYTHONUNBUFFERED": "1", "TOKENIZERS_PARALLELISM": "false"}
            if os.name == "nt":
                environment["USE_LIBUV"] = "0"
                environment["QIHC_BARRIER_DIRECTORY"] = str(self.root / "barriers" / (name + "-" + uuid.uuid4().hex))
            result = subprocess.run(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT,
                                    env=environment)
        if result.returncode or any(not path.is_file() for path in artifacts):
            raise RuntimeError(f"Stage {name} failed; inspect {self.root / 'logs' / (name + '.log')}")
        if artifact and artifact.name.endswith("download_manifest.json"):
            for row in json.loads(artifact.read_text(encoding="utf-8")):
                artifacts.extend(artifact.parent / f"{row['name']}.{suffix}" for suffix in row["files"])
        atomic_json(marker, {"completed": True, "command": command,
                            "artifact_sha256": {str(path): file_digest(path) for path in artifacts},
                            "elapsed_s": time.time() - started})


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=["smoke", "pilot", "formal"], default="smoke")
    parser.add_argument("--output", type=Path, default=ROOT / "deployment/results/pbit_llm_research_v4")
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--comparison-model-paths", type=Path, nargs="*", default=[])
    parser.add_argument("--hgs-binary", type=Path)
    parser.add_argument("--data", type=Path, nargs="*", default=[])
    parser.add_argument("--nproc", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument("--benchmark-dir", type=Path, default=ROOT / "deployment/data/CVRPLIB/X")
    parser.add_argument("--cases-per-range", type=int, default=12)
    parser.add_argument("--search-seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--instances-per-size", type=int)
    parser.add_argument("--search-time-limit", type=float)
    parser.add_argument("--training-steps", type=int, default=200)
    parser.add_argument("--grpo-generation-batch-size", type=int, default=0)
    parser.add_argument("--deepspeed-config", type=Path)
    parser.add_argument("--min-training-rows", type=int, default=200)
    parser.add_argument("--arms", nargs="+", choices=list(ARMS))
    parser.add_argument("--skip-training-smoke", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--train-max-length", type=int, default=32768)
    parser.add_argument("--sft-learning-rate", type=float, default=2e-5)
    parser.add_argument("--dpo-learning-rate", type=float, default=5e-6)
    parser.add_argument("--grpo-learning-rate", type=float, default=2e-6)
    parser.add_argument("--llm-max-new-tokens", type=int, default=3072)
    parser.add_argument("--model-fingerprint", choices=["sha256", "metadata"], default="sha256")
    args = parser.parse_args()
    if args.nproc < 1:
        parser.error("--nproc must be positive")
    smoke = args.profile == "smoke"
    if not smoke and args.model_path is None:
        parser.error("pilot/formal require --model-path")
    uses_llm = not smoke
    sizes = [12, 16] if smoke else [20, 50, 100]
    seeds = args.search_seeds if not smoke else [0, 1]
    chains, steps, top, iterations = (24, 8, 8, 2) if smoke else (512, 80, 64, 40)
    if args.profile == "formal":
        chains, steps, top = 2048, 160, 128
    horizon = args.search_time_limit if args.search_time_limit is not None else (0.0 if smoke else 300.0)
    cases = args.instances_per_size or (4 if smoke else (8 if args.profile == "pilot" else 40))
    selected_arms = args.arms or list(ARMS)
    if smoke:
        selected_arms = [name for name in selected_arms if name not in {"no_prompt", "no_token", "no_pair_information"}]
        if importlib.util.find_spec("ortools") is None:
            selected_arms = [name for name in selected_arms if name != "cp_sat_pool"]
    if not smoke and not args.dry_run:
        for package in ("torch", "transformers", "ortools", "scipy"):
            if importlib.util.find_spec(package) is None:
                raise RuntimeError(f"Missing research dependency: {package}; install qihc[formal]")
        if not (args.model_path / "config.json").exists():
            raise RuntimeError(f"Missing model config: {args.model_path}")
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    execution_lock = SuiteLock(root)
    config = {key: ([str(x) for x in value] if isinstance(value, list) and value and isinstance(value[0], Path)
                    else str(value) if isinstance(value, Path) else value)
              for key, value in vars(args).items() if key != "dry_run"}
    config["source_fingerprint"] = source_fingerprint()
    config["runtime_versions"] = {}
    for package in ("numpy", "torch", "transformers", "ortools", "scipy", "datasets", "peft", "trl", "deepspeed"):
        try:
            config["runtime_versions"][package] = version(package)
        except PackageNotFoundError:
            config["runtime_versions"][package] = None
    config["input_sha256"] = {str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest() for path in args.data}
    config["deepspeed_sha256"] = file_digest(args.deepspeed_config) if args.deepspeed_config and args.deepspeed_config.exists() else None
    if not args.dry_run:
        config["model_fingerprints"] = {str(path): model_fingerprint(path, hash_weights=args.model_fingerprint == "sha256")
                                       for path in ([args.model_path] if args.model_path else []) + args.comparison_model_paths}
    lock = root / ("planned_configuration.json" if args.dry_run else "configuration.json")
    if lock.exists() and json.loads(lock.read_text(encoding="utf-8")) != config:
        raise RuntimeError("Source/config changed: use a new --output directory to preserve experiment provenance")
    atomic_json(lock, config)
    runner = StageRunner(root, args.dry_run)
    if uses_llm:
        preflight = [sys.executable, "experiments/check_research_environment.py", "--model-path", str(args.model_path),
                     "--nproc", str(args.nproc), "--output", str(root / "environment.json")]
        if not args.skip_training:
            preflight += ["--training"]
        runner.run("preflight", preflight, root / "environment.json")
        if not args.skip_training and not args.skip_training_smoke:
            runner.run("training_api_check", [sys.executable, "experiments/check_training_integration.py", "--output", str(root / "training_api_check")], root / "training_api_check/summary.json")
        if args.preflight_only:
            execution_lock.release()
            return 0
    data_root = root / "data"
    raw_inputs = list(args.data)
    if not smoke and not raw_inputs:
        for name, low, high in (("medium", 200, 399), ("hard", 400, 1000)):
            download_manifest = args.benchmark_dir / f"{name}_download_manifest.json"
            runner.run(f"download_{name}", [sys.executable, "experiments/download_hard_cvrplib.py", str(args.benchmark_dir),
                       "--min-customers", str(low), "--max-customers", str(high), "--limit", str(args.cases_per_range), "--selection", "spread",
                       "--verify-cached", "--manifest-name", download_manifest.name], download_manifest)
            raw = root / f"raw_{name}.jsonl"
            runner.run(f"prepare_{name}", [sys.executable, "experiments/prepare_hard_nlcvrp.py", str(args.benchmark_dir), str(raw),
                       "--min-customers", str(low), "--max-customers", str(high), "--limit", str(args.cases_per_range),
                       "--selection", "spread", "--clauses-per-type", "3"], raw)
            raw_inputs.append(raw)
    runner.run("data", [sys.executable, "experiments/prepare_research_data.py", "--output", str(data_root),
               "--sizes", *map(str, sizes), "--instances-per-size", str(cases),
               "--clauses-per-type", "1" if smoke else "2", "--inputs", *map(str, raw_inputs)], data_root / "split_manifest.json",
               [data_root / filename for filename in ("all.jsonl", "train.jsonl", "validation.jsonl", "test.jsonl", "semantic_test.jsonl", "oracle_cpp.jsonl")])

    def launch(script: str, command: list[str]) -> list[str]:
        if args.nproc > 1:
            if os.name == "nt":
                return [sys.executable, "experiments/torchrun_compat.py", "--rdzv-backend=c10d",
                        "--rdzv-endpoint=127.0.0.1:0", "--local-addr=127.0.0.1",
                        f"--nproc_per_node={args.nproc}", script, *command]
            return [sys.executable, "-m", "torch.distributed.run", "--standalone", f"--nproc_per_node={args.nproc}", script, *command]
        return [sys.executable, script, *command]

    common = ["--dataset", "jsonl", "--methods", "llm" if uses_llm else "knn", "--sampler", "torch" if uses_llm else "numpy",
              "--search-seeds", *map(str, seeds), "--iterations", str(iterations), "--patience", str(iterations + 1),
              "--num-chains", str(chains), "--sampling-steps", str(steps), "--top-samples", str(top),
              "--destroy-size", "8" if smoke else "20", "--routes-per-customer", "4" if smoke else "8",
              "--initialization", "pbit-cold", "--cold-selector-mode", "neutral", "--cold-quality-attempts", "2",
              "--cold-batch-size", "4" if smoke else "8", "--cold-routes-per-customer", "4" if smoke else "8",
              "--constraint-source", "cpp", "--cpp-records", str(data_root / "oracle_cpp.jsonl"),
              "--feedback-mode", "population", "--feedback-population-limit", str(top),
              "--search-time-limit", str(horizon), "--backend-time-limit", "0.25" if smoke else "5",
              "--proposal-bias", "0.25", "--llm-temperature", "0", "--llm-refresh-interval", "1",
              "--block-updates", "--constraint-aware-decode",
              "--temperature-ladder",
              "--normalize-objective",
              "--resume"]
    if uses_llm:
        common += ["--model-path", str(args.model_path), "--llm-max-new-tokens", str(args.llm_max_new_tokens), "--llm-max-input-tokens", "32768"]
        actual_smoke = root / "model_smoke"
        runner.run("model_smoke", launch("experiments/run_cvrp_lns.py", [*common,
                   "--data", str(data_root / "train.jsonl"), "--limit", "1", "--iterations", "2",
                   "--cold-selector-mode", "selector", "--output", str(actual_smoke)]), actual_smoke / "results.json")
        runner.run("model_smoke_check", [sys.executable, "experiments/check_model_smoke.py", "--root", str(actual_smoke)],
                   actual_smoke / "model_smoke_check.json")
    shared = root / "shared_initial"
    runner.run("shared_initial", launch("experiments/run_cvrp_lns.py", [*common, "--data", str(data_root / "all.jsonl"),
               "--methods", "knn", "--iterations", "0", "--output", str(shared)]), shared / "results.json")
    runner.run("export_initial", [sys.executable, "experiments/export_shared_initials.py", "--results", str(shared / "results.json"),
               "--output", str(data_root / "shared_initials.jsonl")], data_root / "shared_initials.jsonl")
    shared_args = ["--initial-solutions", str(data_root / "shared_initials.jsonl")]
    evaluation_arms = []

    def evaluate(label: str, extra: list[str], split: str = "test") -> None:
        output = root / "evaluation" / label
        runner.run(label, launch("experiments/run_cvrp_lns.py", [*common, *shared_args, *extra,
                   "--data", str(data_root / f"{split}.jsonl"), "--output", str(output)]), output / "results.json")
        if split == "test":
            evaluation_arms.append(label)

    for arm in selected_arms:
        evaluate(arm, ARMS[arm])
    if uses_llm:
        # Frontend accuracy and end-to-end behavior are separate from the controlled backend study.
        cpp_root = root / "generated_cpp"
        runner.run("semantic_frontend", launch("experiments/run_s2e_pipeline.py", ["--data", str(data_root / "all.jsonl"),
                   "--model-path", str(args.model_path), "--output", str(cpp_root), "--max-new-tokens", "1536", "--resume"]), cpp_root / "summary.json")
        evaluate("generated_cpp", ["--cpp-records", str(cpp_root / "records.jsonl")])
        runner.run("semantic_robustness_base", launch("experiments/run_s2e_pipeline.py", ["--data", str(data_root / "semantic_test.jsonl"),
                   "--model-path", str(args.model_path), "--output", str(root / "semantic_robustness/base"),
                   "--max-new-tokens", "1536", "--resume"]), root / "semantic_robustness/base/summary.json")
        for index, model_path in enumerate(args.comparison_model_paths):
            alternate_cpp = root / f"comparison_model_{index}_cpp"
            runner.run(f"semantic_model_{index}", launch("experiments/run_s2e_pipeline.py", ["--data", str(data_root / "test.jsonl"),
                       "--model-path", str(model_path), "--output", str(alternate_cpp), "--max-new-tokens", "1536", "--resume"]), alternate_cpp / "summary.json")
            evaluate(f"comparison_model_{index}", ["--model-path", str(model_path), "--cpp-records", str(alternate_cpp / "records.jsonl")])
        evaluate("llm_direct", ["--methods", "llm_direct", "--constraint-source", "dataset"])
        evaluate("ortools", ["--methods", "ortools", "--constraint-source", "dataset", "--baseline-time-limit", str(int(horizon or 300))])
        if args.hgs_binary:
            evaluate("hgs_constraint_unaware", ["--methods", "hgs", "--constraint-source", "dataset", "--hgs-binary", str(args.hgs_binary), "--baseline-time-limit", str(int(horizon or 300))])
        evaluate("cold_neutral", ["--initial-solutions", str(data_root / "shared_initials.jsonl"), "--iterations", "0"])
        # Remove the shared-initial flag to test genuine construction separately.
        cold_dir = root / "evaluation/cold_llm"
        runner.run("cold_llm", launch("experiments/run_cvrp_lns.py", [*common, "--cold-selector-mode", "selector", "--iterations", "0",
                   "--data", str(data_root / "test.jsonl"), "--output", str(cold_dir)]), cold_dir / "results.json")
        evaluation_arms.append("cold_llm")
        for budget in (60, 180, 600):
            time_args = ["--search-time-limit", str(budget), "--time-budget-only", "--patience", "0"]
            evaluate(f"budget_{budget}_population", time_args)
            evaluate(f"budget_{budget}_no_feedback", [*time_args, "--disable-logit-feedback"])
            evaluate(f"budget_{budget}_pbit_qubo", [*time_args, "--search-backend", "pbit-qubo"])
            evaluate(f"budget_{budget}_cp_sat", [*time_args, "--search-backend", "cp-sat"])
        for population_size in (128, 512, 2048):
            evaluate(f"chains_{population_size}", ["--num-chains", str(population_size)])
        for information_budget in (16, 64, 128):
            evaluate(f"feedback_samples_{information_budget}", ["--feedback-population-limit", str(information_budget)])
        for token_budget in (768, 1536, 3072):
            evaluate(f"tokens_{token_budget}", ["--llm-max-new-tokens", str(token_budget)])
        if not args.skip_training:
            teacher_root = root / "teacher"
            runner.run("teacher", launch("experiments/run_cvrp_lns.py", [*common,
                       "--data", str(data_root / "train.jsonl"), "--output", str(teacher_root),
                       "--cold-selector-mode", "selector", "--export-training-context"]), teacher_root / "results.json")
            training_root = root / "training_data"
            runner.run("training_data", [sys.executable, "experiments/prepare_joint_training_data.py",
                       "--data", str(data_root / "train.jsonl"), "--constraint-records", str(data_root / "oracle_cpp.jsonl"),
                       "--solver-results", str(teacher_root / "results.json"), "--llm-audits", str(teacher_root),
                       "--verify-preferences", "--output", str(training_root)], training_root / "summary.json",
                       [training_root / name for name in ("sft.jsonl", "dpo.jsonl", "grpo.jsonl", "holdout_instances.json")])
            if not args.dry_run:
                summary = json.loads((training_root / "summary.json").read_text(encoding="utf-8"))
                if summary["sft"] < args.min_training_rows or summary["dpo"] == 0 or summary["grpo"] == 0:
                    raise RuntimeError(f"Insufficient verified training data: {summary}; increase --instances-per-size")
            previous_adapter = None
            for stage in ("sft", "dpo", "grpo"):
                adapter = root / "adapters" / stage
                command = ["--stage", stage, "--model-path", str(args.model_path), "--data", str(training_root / f"{stage}.jsonl"),
                           "--output", str(adapter), "--max-steps", str(args.training_steps), "--max-length", str(args.train_max_length),
                           "--reward-mode", "rollout", "--resume"]
                command += ["--max-completion-length", str(args.llm_max_new_tokens)]
                command += ["--learning-rate", str(getattr(args, stage + "_learning_rate"))]
                if previous_adapter:
                    command += ["--adapter-path", str(previous_adapter)]
                if args.deepspeed_config:
                    command += ["--deepspeed-config", str(args.deepspeed_config)]
                if stage == "grpo":
                    command += ["--require-reward-variance", "--generation-batch-size", str(args.grpo_generation_batch_size)]
                runner.run(f"train_{stage}", launch("experiments/train_s2e_lora.py", command), adapter / "qihc_training_manifest.json", [adapter / "adapter_config.json", adapter / "adapter_model.safetensors"])
                if not args.dry_run and not any((adapter / name).is_file() for name in ("adapter_model.safetensors", "adapter_model.bin")):
                    raise RuntimeError(f"Training did not produce adapter weights: {adapter}")
                evaluate(f"{stage}_validation", ["--adapter-path", str(adapter)], "validation")
                evaluate(f"{stage}_test", ["--adapter-path", str(adapter)])
                stage_cpp = root / f"{stage}_cpp"
                runner.run(f"{stage}_semantic", launch("experiments/run_s2e_pipeline.py", ["--data", str(data_root / "test.jsonl"),
                           "--model-path", str(args.model_path), "--adapter-path", str(adapter), "--output", str(stage_cpp),
                           "--max-new-tokens", "1536", "--resume"]), stage_cpp / "summary.json")
                evaluate(f"{stage}_end_to_end", ["--adapter-path", str(adapter), "--cpp-records", str(stage_cpp / "records.jsonl")])
                runner.run(f"semantic_robustness_{stage}", launch("experiments/run_s2e_pipeline.py", ["--data", str(data_root / "semantic_test.jsonl"),
                           "--model-path", str(args.model_path), "--adapter-path", str(adapter),
                           "--output", str(root / "semantic_robustness" / stage), "--max-new-tokens", "1536", "--resume"]),
                           root / "semantic_robustness" / stage / "summary.json")
                previous_adapter = adapter
    manifest = {"profile": args.profile, "uses_llm": uses_llm, "evaluation_arms": evaluation_arms,
                "search_seeds": seeds, "search_time_limit_s": horizon, "test_instances": [],
                "controlled_frontend": "oracle_cpp_shared_by_all_backends",
                "training_skipped": smoke or args.skip_training,
                "unavailable_in_smoke": [name for name in ARMS if name not in selected_arms]}
    if not args.dry_run:
        split_manifest = json.loads((data_root / "split_manifest.json").read_text(encoding="utf-8"))
        manifest["test_instances"] = split_manifest["splits"]["test"]
    atomic_json(root / ("planned_manifest.json" if args.dry_run else "suite_manifest.json"), manifest)
    runner.run("analysis", [sys.executable, "experiments/analyze_research_suite.py", str(root)], root / "research_results.json")
    print(f"{'Plan' if args.dry_run else 'Results'}: {root}", flush=True)
    execution_lock.release()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
