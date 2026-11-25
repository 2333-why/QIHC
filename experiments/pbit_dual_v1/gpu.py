"""Train two adapters, then shard new inference across four GPUs."""

import contextlib
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from experiments.boundary_feedback_v1 import policy
from experiments.dcp_external_v2.gpu import Calls, draft
from experiments.pbit_joint_v1 import run as training, search
from experiments.pbit_joint_v1.modeling import token_weights
from experiments.pbit_joint_v3.methods import balanced_replay
from qihc.experiment_runtime.pbit_control import seed
from qihc.experiment_runtime.storage import (
    digest,
    load,
    save,
    value_digest,
    worker_lock,
)
from qihc.modeling.active_protocol_v2 import model_prompt
from .common import RUN, SEARCH, artifact, cases, seeds, verify
from .methods import extended_prompt, feedback


def train(engine, state, arm):
    import torch
    from experiments.posttraining_v1.data import target

    folder = RUN / "trained" / arm
    rows = [load(Path(p)) for p in state["training"]]
    binding = dict(manifest=digest(RUN / "manifest.json"), arm=arm)
    parent = torch.load(state["paths"]["parent"], map_location="cpu", weights_only=True)
    if arm == "model":
        engine.restore(parent)
        originals = len(rows)
        rows += [child for row in rows for child in row["augmented"]]

        def backward(i):
            row = rows[i]
            text = target(row["gold"])
            weights = token_weights(
                engine.tok, text, row["candidate"], row["gold"], row["cards"]
            )
            loss = engine.backward(
                model_prompt(row["case"]),
                engine.encode(text) + [engine.tok.eos_token_id],
                "nll",
                weights=weights + [1.0],
                reference=parent,
            )
            return dict(
                case_id=row["case"]["id"],
                loss=loss,
                weighted_tokens=sum(w > 1 for w in weights),
            )

        # Keep all original examples in each epoch. Train on each augmented prompt once.
        order = training.order_for(originals, 3)
        order += [originals + i for i in training.order_for(len(rows) - originals, 1)]
        lr = 5e-6
    else:
        items = [(r["case"]["id"], x) for r in rows for x in r["controls"]]
        if not state["supervision"]["positive"]:
            save(folder / "abstain.json", binding)
            return
        action_ids = training.action_ids(engine)

        def backward(i):
            cid, item = items[i]
            action = search.ACTIONS.index(item["action"])
            scores = training.logits(engine, item["prompt"], action_ids)
            diff = scores[action] - scores[0]
            loss = torch.nn.functional.smooth_l1_loss(
                diff, diff.new_tensor(item["target"])
            )
            loss.backward()
            return dict(
                case_id=cid,
                loss=float(loss.detach()),
                action=item["action"],
                target=item["target"],
            )

        replay = [
            (cid, row["prompt"], (row["action"], row["target"] > 0.10), 0)
            for cid, row in items
        ]
        order = balanced_replay(replay, min(2880, 2 * len(items)), 17)
        lr = 2e-5
    with worker_lock(folder / "train.lock"):
        training.optimize(engine, folder, order, lr, backward, binding)


def make_feedback(engine, case, pred, names):
    pending = [(name, artifact("predictions", case, name)) for name in names]
    pending = [(name, path) for name, path in pending if not path.exists()]
    if not pending:
        return
    original = pred["model"]
    shared_path = artifact("candidates", case, names[-1])
    if not shared_path.exists():
        ask = Calls(engine)
        shared = policy.shared(dict(case=case, seed=17, baseline=original), ask)
        save(shared_path, dict(shared, calls=ask.log))
    shared = load(shared_path)
    for name, dest in pending:
        ask = Calls(engine)
        decision = feedback(
            case, original, shared, ask, multi=name != "weighted_quadratic"
        )
        calls = shared["calls"] + ask.log
        save(
            dest,
            dict(
                decision, calls=pred.get("calls", []) + calls, feedback_calls=len(calls)
            ),
        )


def predict(engine, state, subset):
    import torch

    weights = lambda name: torch.load(name, map_location="cpu", weights_only=True)
    engine.restore(weights(state["paths"]["weighted"]))
    for case in subset:
        old = load(RUN / "frozen" / case["dataset"] / (case["id"] + ".json"))[
            "prediction"
        ]
        make_feedback(engine, case, old, ("weighted_quadratic", "weighted_feedback"))
    engine.restore(weights(RUN / "trained/model/adapter.pt"))
    for case in subset:
        dest = artifact("predictions", case, "trained")
        if not dest.exists():
            save(
                dest,
                draft(
                    case, Calls(engine), model_prompt(case), seed(case["id"], "model")
                ),
            )
        make_feedback(engine, case, load(dest), ("trained_feedback",))


def get_prediction(case, name):
    if name == "weighted":
        return load(RUN / "frozen" / case["dataset"] / (case["id"] + ".json"))[
            "prediction"
        ]
    return load(artifact("predictions", case, name))


def evaluate_control(engine, state, subset):
    import torch

    control = RUN / "trained/control/adapter.pt"
    if control.exists():
        engine.restore(torch.load(control, map_location="cpu", weights_only=True))
        ids = training.action_ids(engine)
        engine.net.eval()
    elif not (control.parent / "abstain.json").exists():
        raise RuntimeError("Missing controller training output")
    support = state["supervision"]["positive"]

    def decide(spec, observed, remaining):
        choices = [
            i
            for i, a in enumerate(search.ACTIONS)
            if a in support and (a != "clamp" or search.intervention(spec, observed))
        ]
        if not choices:
            return "cool"
        with torch.inference_mode():
            try:
                scores = training.logits(
                    engine, extended_prompt(spec, observed, remaining), ids
                )
            except ValueError as exc:
                if "context overflow" not in str(exc).lower():
                    raise
                return "cool"
            winner = choices[int(scores[choices].argmax())]
            return (
                search.ACTIONS[winner]
                if float(scores[winner] - scores[0]) > 0.10
                else "cool"
            )

    for case in subset:
        memo = {}
        for name in ("weighted", "trained", "trained_feedback"):
            spec = get_prediction(case, name)["model"]
            for random_seed in seeds(case, "control"):
                dest = artifact("episodes", case, f"{name}__control", random_seed)
                fingerprint = (random_seed, value_digest(spec))
                if dest.exists():
                    memo[fingerprint] = load(dest)
                    continue
                if fingerprint not in memo:
                    if not support:
                        result = dict(reuse_schedule=True)
                    else:
                        result = (
                            search.episode(
                                spec, case["id"], dict(SEARCH, seed=random_seed), decide
                            )
                            if spec
                            else dict(x=None, status="input_invalid")
                        )
                    memo[fingerprint] = result
                save(dest, memo[fingerprint])


def worker(task):
    import torch
    from experiments.posttraining_v1.engine import Engine

    if (
        torch.cuda.device_count() != 1
        or not torch.cuda.is_bf16_supported()
        or (torch.cuda.get_device_properties(0).total_memory < 75 * 1024**3)
    ):
        raise RuntimeError("Expected one BF16 GPU with about 80 GB per worker")
    torch.manual_seed(17)
    state = load(RUN / "manifest.json")
    engine = Engine(Path(state["model"]))
    if task in ("model", "control"):
        train(engine, state, task)
    else:
        subset = cases()[int(task) :: 4]
        predict(engine, state, subset)
        evaluate_control(engine, state, subset)
        save(RUN / "shards" / f"{task}.json", dict(cases=len(subset)))


def run_workers(tasks, devices):
    with contextlib.ExitStack() as stack:
        running = []
        try:
            for task, device in zip(tasks, devices):
                log = stack.enter_context((RUN / "logs" / f"gpu_{task}.log").open("ab"))
                command = [
                    sys.executable,
                    "-m",
                    "experiments.pbit_dual_v1.run",
                    "worker",
                    task,
                ]
                running.append(
                    (
                        task,
                        subprocess.Popen(
                            command,
                            stdout=log,
                            stderr=subprocess.STDOUT,
                            env=dict(os.environ, CUDA_VISIBLE_DEVICES=device),
                        ),
                    )
                )
            while any(p.poll() is None for _, p in running):
                for task, p in running:
                    if p.poll() not in (None, 0):
                        raise RuntimeError(
                            f"Worker failed: {RUN / 'logs' / f'gpu_{task}.log'}"
                        )
                time.sleep(1)
            for task, p in running:
                if p.returncode:
                    raise RuntimeError(
                        f"Worker failed: {RUN / 'logs' / f'gpu_{task}.log'}"
                    )
        finally:
            for _, p in running:
                if p.poll() is None:
                    p.terminate()
            for _, p in running:
                try:
                    p.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    p.kill()
                    p.wait()


def launch():
    import torch

    devices = os.getenv("CUDA_VISIBLE_DEVICES", "0,1,2,3").split(",")
    if torch.cuda.device_count() != 4 or len(devices) != 4 or len(set(devices)) != 4:
        raise ValueError("Allocate four visible GPUs")
    with worker_lock(RUN / "gpu.lock"), contextlib.ExitStack() as stack:

        def interrupt(*_):
            raise KeyboardInterrupt("GPU job interrupted")

        old = signal.signal(signal.SIGTERM, interrupt)
        stack.callback(signal.signal, signal.SIGTERM, old)
        verify(checkpoints=True)
        (RUN / "logs").mkdir(exist_ok=True)
        tasks = [
            name
            for name in ("model", "control")
            if not any(
                (RUN / "trained" / name / item).exists()
                for item in ("adapter.pt", "abstain.json")
            )
        ]
        run_workers(tasks, devices)
        shards = [
            str(i) for i in range(4) if not (RUN / "shards" / f"{i}.json").exists()
        ]
        run_workers(shards, devices)
