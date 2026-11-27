"""Four independent training jobs, followed by four disjoint inference shards."""

import os
import random
import signal
import subprocess
import sys
import time
from contextlib import ExitStack
from pathlib import Path

from experiments.dcp_external_v2.gpu import Calls, draft
from experiments.pbit_joint_v1.run import optimize
from experiments.pbit_joint_v1.modeling import token_weights
from experiments.pbit_stack_v1.gpu import restore
from experiments.pbit_stack_v1.feedback import verify_patch
from experiments.posttraining_v1.data import target
from qihc.experiment_runtime.pbit_control import seed
from qihc.experiment_runtime.storage import digest, load, save, worker_lock
from qihc.modeling.active_protocol_v2 import model_prompt
from . import feedback
from .common import RUN, TASKS, manifest, path


def training_order(corpus, arm):
    patch = arm == "feedback"
    base = list(corpus["nlco_patch" if patch else "nlco"])
    addition = list(corpus["dcp_patch" if patch else "dcp"])
    rng = random.Random(17)
    rng.shuffle(base)
    rng.shuffle(addition)
    count = len(addition)
    replay = [base[i % len(base)] for i in range(2 * count)]
    # Identical NLCO rows at the even steps, replace only the odd steps.
    if arm.endswith("mixed") or patch:
        replay[1::2] = addition
    return replay


def train(engine, m, arm):
    import torch

    patch = arm == "feedback"
    anchor = m["adapters"][arm.split("_")[0]]
    reference = torch.load(anchor, map_location="cpu", weights_only=True)
    engine.restore(reference)
    rows = training_order(load(RUN / "corpus.json"), arm)

    def backward(i):
        r = rows[i]
        if patch:
            prompt, answer, cid = r["prompt"], r["target"], r["case_id"]
            weights = None
        else:
            prompt, answer, cid = (
                model_prompt(r["case"]),
                target(r["gold"]),
                r["case"]["id"],
            )
            weights = token_weights(
                engine.tok, answer, r["candidate"], r["gold"], r["cards"]
            ) + [1.0]
        ids = engine.encode(answer) + [engine.tok.eos_token_id]
        loss = engine.backward(prompt, ids, "nll", weights=weights, reference=reference)
        return dict(
            case_id=cid, loss=loss, weighted_tokens=sum(w > 1 for w in (weights or []))
        )

    binding = dict(arm=arm, manifest=digest(RUN / "manifest.json"))
    optimize(
        engine, RUN / "trained" / arm, list(range(len(rows))), 2.5e-6, backward, binding
    )


def infer(engine, m, shard):
    subset = load(RUN / "cases.json")[shard::4]
    for dataset in ("nlco", "dcp"):
        cases = [c for c in subset if c["dataset"] == dataset]
        for method in ("replay", "mixed"):
            restore(engine, RUN / "trained" / f"{dataset}_{method}" / "adapter.pt")
            for c in cases:
                out = path("predictions", c, method)
                if not out.exists():
                    save(
                        out,
                        draft(
                            c, Calls(engine), model_prompt(c), seed(c["id"], "model")
                        ),
                    )
    restore(engine, RUN / "trained/feedback/adapter.pt")
    for c in subset:
        for arm, kind, parent in (
            ("feedback_only", "inputs", "anchor"),
            ("joint", "predictions", "mixed"),
        ):
            out = path("proposals", c, arm)
            if not out.exists():
                ask = Calls(engine)
                proposed = feedback.propose(c, load(path(kind, c, parent)), ask)
                save(out, dict(proposed, calls=ask.log))
    for dataset in ("nlco", "dcp"):
        restore(engine, m["judges"][dataset])
        for c in (c for c in subset if c["dataset"] == dataset):
            for arm, kind, parent in (
                ("feedback_only", "inputs", "anchor"),
                ("joint", "predictions", "mixed"),
            ):
                out = path("predictions", c, arm)
                if not out.exists():
                    ask = Calls(engine)
                    proposal = load(path("proposals", c, arm))
                    result = verify_patch(c, load(path(kind, c, parent)), proposal, ask)
                    save(
                        out,
                        dict(
                            result,
                            retry=proposal["retry"],
                            parse_failed=proposal["parse_failed"],
                            feedback_calls=len(proposal["calls"]) + len(ask.log),
                            judge_calls=ask.log,
                            proposer_calls=proposal["calls"],
                            sampling_seconds=proposal["sampling_seconds"],
                            sampling_attempts=proposal["sampling_attempts"],
                        ),
                    )
    save(RUN / "gpu" / f"{shard}.json", dict(cases=len(subset)))


def worker(task):
    import torch
    from experiments.posttraining_v1.engine import Engine

    if (
        torch.cuda.device_count() != 1
        or not torch.cuda.is_bf16_supported()
        or torch.cuda.get_device_properties(0).total_memory < 75 * 1024**3
    ):
        raise RuntimeError("Worker requires one visible BF16 GPU with at least 75 GiB")
    torch.manual_seed(17)
    m = manifest()
    with worker_lock(RUN / "locks" / (task + ".lock")):
        engine = Engine(Path(m["model"]))
        if task in TASKS:
            train(engine, m, task)
        else:
            infer(engine, m, int(task))


def batch(tasks, devices):
    pending = list(tasks)
    with ExitStack() as cleanup:
        live = {}
        while pending or live:
            for device in devices:
                if device in live:
                    p, log = live[device]
                    code = p.poll()
                    if code is None:
                        continue
                    log.close()
                    del live[device]
                    if code:
                        raise RuntimeError(f"GPU worker exited {code}; see {log.name}")
                if pending:
                    task = pending.pop(0)
                    log = cleanup.enter_context(
                        (RUN / "logs" / f"{task}.log").open("ab")
                    )
                    args = [
                        sys.executable,
                        "-m",
                        "experiments.dcp_transfer_v1.run",
                        "worker",
                        str(task),
                    ]
                    p = subprocess.Popen(
                        args,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        env=dict(os.environ, CUDA_VISIBLE_DEVICES=device),
                    )
                    cleanup.callback(stop, p)
                    live[device] = p, log
            if live:
                time.sleep(1)


def stop(p):
    if p.poll() is None:
        p.terminate()
        try:
            p.wait(timeout=8)
        except subprocess.TimeoutExpired:
            p.kill()
            p.wait()


def launch():
    import torch
    from experiments.dcp_external_v2.identity import sha

    devices = os.getenv("CUDA_VISIBLE_DEVICES", "0,1,2,3").split(",")
    if len(devices) != 4 or len(set(devices)) != 4 or torch.cuda.device_count() != 4:
        raise RuntimeError("Allocate four distinct visible GPUs")

    def interrupt(*_):
        raise KeyboardInterrupt

    with worker_lock(RUN / "gpu.lock"), ExitStack() as cleanup:
        cleanup.callback(
            signal.signal, signal.SIGTERM, signal.signal(signal.SIGTERM, interrupt)
        )
        m = manifest(True)
        if any(sha(Path(m["model"]) / f) != h for f, h in m["model_files"].items()):
            raise ValueError("Frozen model files changed")
        (RUN / "logs").mkdir(exist_ok=True)
        tasks = [t for t in TASKS if not (RUN / "trained" / t / "adapter.pt").exists()]
        batch(tasks, devices)
        batch(
            [str(i) for i in range(4) if not (RUN / "gpu" / f"{i}.json").exists()],
            devices,
        )
