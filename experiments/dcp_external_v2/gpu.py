"""Four independent fixed case shards. No gold data enters inference or feedback prompts."""

import contextlib
import os
from pathlib import Path
import subprocess
import signal
import sys
import time

from qihc.experiment_runtime.storage import load, save, worker_lock
from qihc.experiment_runtime.pbit_control import seed
from qihc.modeling.active_protocol_v2 import model_prompt, parse_model
from experiments.boundary_feedback_v1 import policy
from experiments.pbit_joint_v1 import search
from .common import (
    RUN,
    MODELS,
    FEEDBACK,
    SEARCH,
    cases,
    path,
    SEED,
    SEARCH_SEEDS,
    stage_name,
)


class Calls:
    def __init__(self, engine):
        self.engine = engine
        self.log = []

    def __call__(self, prompt, random_seed):
        start = time.monotonic()
        response = self.engine.evaluate(prompt, random_seed & 0x7FFFFFFF)
        choice = response["choices"][0]
        self.log.append(
            dict(
                seed=random_seed,
                usage=response.get("usage", {}),
                finish=choice["finish_reason"],
                seconds=time.monotonic() - start,
            )
        )
        return choice["message"]["content"] if choice["finish_reason"] == "stop" else ""


def draft(case, ask, prompt, random_seed):
    text = ask(prompt, random_seed)
    try:
        spec = parse_model(text, case)
        error = None
    except (ValueError, KeyError, TypeError, IndexError, OverflowError) as exc:
        spec, error = None, str(exc)[:500]
    return dict(model=spec, error=error, raw=text, calls=ask.log[:])


def predict(engine, case, model):
    cid = case["id"]
    dest = path("predictions", model, cid)
    if not dest.exists():
        save(dest, draft(case, Calls(engine), model_prompt(case), seed(cid, "model")))
    first = load(dest)
    if model not in FEEDBACK:
        return
    repaired = path("shared", model + "_repair", cid)
    if not repaired.exists():
        if first["model"] is None:
            prompt = (
                model_prompt(case)
                + "\nRepair the unparseable draft. Parser: "
                + str(first["error"])
                + "\nDraft: "
                + first["raw"]
            )
            fix = draft(case, Calls(engine), prompt, seed(cid, "repair"))
        else:
            fix = dict(model=first["model"], calls=[])
        save(repaired, fix)
    fix = load(repaired)
    item = dict(case=case, seed=SEED, model=first["model"], baseline=fix["model"])
    shared = path("shared", model + "_candidate", cid)
    if not shared.exists():
        ask = Calls(engine)
        save(shared, dict(policy.shared(item, ask), calls=ask.log))
    candidate = load(shared)
    for arm in FEEDBACK[model]:
        dest = path("predictions", model + "_" + arm, cid)
        if not dest.exists():
            ask = Calls(engine)
            from qihc.generic.solver_baselines.milp import UnsupportedModel

            try:
                decision = policy.decide(item, arm, candidate, ask, seconds=20)
            except UnsupportedModel as exc:
                decision = dict(
                    model=fix["model"], action="probe_unsupported_keep", detail=str(exc)
                )
            calls = (
                first["calls"]
                + fix["calls"]
                + ([] if arm == "B" else candidate["calls"])
                + ask.log
            )
            save(dest, dict(decision, calls=calls))


def control(engine, case, model, arm, manifest, random_seed):
    import torch
    from experiments.pbit_joint_v1.run import action_ids, logits
    from experiments.pbit_joint_v2.signals import control_prompt

    cid = case["id"]
    dest = path(stage_name("episodes", random_seed), model + "__" + arm, cid)
    if dest.exists():
        return
    spec = load(path("predictions", model, cid))["model"]
    ids = action_ids(engine)
    choices = [
        i
        for i, a in enumerate(search.ACTIONS)
        if a == "cool" or a in manifest["winners"][arm]
    ]

    def choose(spec, state, remaining):
        with torch.inference_mode():
            scores = logits(engine, control_prompt(spec, state, remaining), ids)
            return search.ACTIONS[choices[int(scores[choices].argmax())]]

    try:
        result = (
            search.episode(spec, cid, dict(SEARCH, seed=random_seed), choose)
            if spec
            else dict(x=None, status="input_invalid")
        )
    except ValueError as exc:
        if "context overflow" not in str(exc).lower():
            raise
        result = dict(x=None, status="controller_context_limit")
    save(dest, result)


def worker(shard):
    import torch
    from experiments.posttraining_v1.engine import Engine

    if (
        torch.cuda.device_count() != 1
        or not torch.cuda.is_bf16_supported()
        or torch.cuda.get_device_properties(0).total_memory < 75 * 1024**3
    ):
        raise RuntimeError("Expected one visible BF16 80 GB GPU per worker")
    manifest = load(RUN / "manifest.json")
    torch.manual_seed(SEED)
    engine = Engine(Path(manifest["model"]))
    subset = cases()[shard::4]
    for arm in MODELS + ("rank", "balanced"):
        if arm != "base":
            engine.restore(
                torch.load(
                    manifest["paths"][arm], map_location="cpu", weights_only=True
                )
            )
        engine.net.eval()
        print("shard", shard, "arm", arm, "cases", len(subset), flush=True)
        for case in subset:
            if arm in MODELS:
                predict(engine, case, arm)
            else:
                for model in ("parent", "weighted"):
                    for random_seed in SEARCH_SEEDS:
                        control(engine, case, model, arm, manifest, random_seed)


def launch():
    import torch
    from .identity import verify

    devices = os.getenv("CUDA_VISIBLE_DEVICES", "0,1,2,3").split(",")
    if torch.cuda.device_count() != 4 or len(devices) != 4 or len(set(devices)) != 4:
        raise ValueError("Allocate four visible GPUs")
    (RUN / "logs").mkdir(parents=True, exist_ok=True)
    with worker_lock(RUN / "gpu.lock"), contextlib.ExitStack() as stack:

        def interrupted(*_):
            raise KeyboardInterrupt("Four-GPU job interrupted")

        if hasattr(signal, "SIGTERM"):
            previous = signal.signal(signal.SIGTERM, interrupted)
            stack.callback(signal.signal, signal.SIGTERM, previous)
        verify(models=True)
        processes = []
        try:
            for shard, device in enumerate(devices):
                log = stack.enter_context((RUN / "logs" / f"gpu{shard}.log").open("a"))
                proc = subprocess.Popen(
                    [sys.executable, "-u", "-m", __spec__.name, str(shard)],
                    env=dict(os.environ, CUDA_VISIBLE_DEVICES=device),
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
                processes.append(proc)
            while any(p.poll() is None for p in processes):
                for i, p in enumerate(processes):
                    if p.poll() not in (0, None):
                        raise RuntimeError(f"GPU {i} failed: {RUN}/logs/gpu{i}.log")
                time.sleep(1)
            if any(p.returncode for p in processes):
                raise RuntimeError(f"GPU worker failed: inspect {RUN}/logs")
        finally:
            for p in processes:
                if p.poll() is None:
                    p.terminate()
            for p in processes:
                try:
                    p.wait(10)
                except subprocess.TimeoutExpired:
                    p.kill()
                    p.wait()


if __name__ == "__main__":
    if len(sys.argv) == 1:
        launch()
    else:
        worker(int(sys.argv[1]))
