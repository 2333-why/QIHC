"""Four independent model replicas for frozen and improved evaluations."""

import contextlib
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from experiments.boundary_feedback_v1 import policy
from experiments.dcp_external_v2.gpu import Calls, draft
from experiments.dcp_external_v2.identity import sha
from experiments.pbit_dual_v1 import common as prior
from experiments.pbit_joint_v1 import run as optimization
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
from . import feedback, penalty
from .common import CONVENTIONAL, RUN, TRAINERS, old_model, path, seed_list, state


def restore(engine, path):
    import torch

    engine.restore(torch.load(path, map_location="cpu", weights_only=True))
    engine.net.eval()


def tokens(engine):
    ids = [engine.encode(c) for c in "ABCD"]
    if any(len(x) != 1 for x in ids) or len({x[0] for x in ids}) != 4:
        raise ValueError("Expected distinct single-token action labels")
    return [x[0] for x in ids]


def train(engine, manifest, arm):
    import torch
    from experiments.posttraining_v1.data import target
    from experiments.pbit_joint_v1.modeling import token_weights

    corpus = load(RUN / "training.json")
    rows = [load(Path(p)) for p in corpus["files"]]
    out = RUN / "trained" / arm
    binding = dict(
        manifest=digest(RUN / "manifest.json"),
        corpus=digest(RUN / "training.json"),
        arm=arm,
    )
    if arm in ("nlco", "dcp"):
        reference = torch.load(
            manifest["checkpoints"][arm], map_location="cpu", weights_only=True
        )
        engine.restore(reference)
        originals = [row["modeling"][0] for row in rows]
        augmented = [x for row in rows for x in row["modeling"][1:]]
        examples = originals + augmented

        def backward(i):
            row = examples[i]
            answer = target(row["gold"])
            prompt = model_prompt(row["case"])
            weights = token_weights(
                engine.tok, answer, row["candidate"], row["gold"], row["cards"]
            )
            loss = engine.backward(
                prompt,
                engine.encode(answer) + [engine.tok.eos_token_id],
                "nll",
                weights=weights + [1.0],
                reference=reference,
            )
            return dict(
                case_id=row["case"]["id"],
                loss=loss,
                weighted_tokens=sum(w > 1 for w in weights),
            )

        order = optimization.order_for(len(originals), 2) + [
            len(originals) + j for j in optimization.order_for(len(augmented), 1)
        ]
        lr = 2.5e-6
    elif arm == "feedback":
        restore(engine, manifest["checkpoints"]["parent"])
        examples = [x for row in rows for x in row["repairs"]]

        def backward(i):
            row = examples[i]
            ids = engine.encode(row["target"]) + [engine.tok.eos_token_id]
            return dict(
                case_id=row["case_id"], loss=engine.backward(row["prompt"], ids, "nll")
            )

        order = optimization.order_for(len(examples), 2)
        lr = 5e-6
    else:
        support = corpus["support"]
        if not support:
            save(out / "abstain.json", binding)
            return
        examples = [
            x for row in rows for x in row["controls"] if x["action"] in support
        ]
        ids = tokens(engine)

        def backward(i):
            row = examples[i]
            scores = optimization.logits(engine, row["text"], ids)
            delta = scores[penalty.ACTIONS.index(row["action"])] - scores[0]
            loss = torch.nn.functional.smooth_l1_loss(
                delta, delta.new_tensor(row["target"])
            )
            loss.backward()
            return dict(
                case_id=row["case_id"], loss=float(loss.detach()), target=row["target"]
            )

        replay = [
            (
                x["case_id"],
                x["text"],
                2 * penalty.ACTIONS.index(x["action"]) + int(x["target"] > 0.05),
                0,
            )
            for x in examples
        ]
        order = balanced_replay(replay, min(2048, len(examples) * 2), 17)
        lr = 2e-5
    with worker_lock(out / "train.lock"):
        optimization.optimize(engine, out, order, lr, backward, binding)


def current(engine, manifest, subset):
    for name in CONVENTIONAL:
        missing = [
            c
            for c in subset
            if c["dataset"] == "nlco" and not path("predictions", c, name).exists()
        ]
        if missing and name != "base":
            restore(engine, manifest["checkpoints"][name])
        if missing:
            print("filling archived prediction gaps", name, len(missing), flush=True)
        for case in missing:
            out = path("predictions", case, name)
            save(
                out,
                draft(
                    case, Calls(engine), model_prompt(case), seed(case["id"], "model")
                ),
            )
    for dataset in ("nlco", "dcp"):
        restore(engine, manifest["checkpoints"][dataset])
        print("reference feedback", dataset, flush=True)
        for case in (c for c in subset if c["dataset"] == dataset):
            out = path("predictions", case, "incumbent_C")
            compile_only = path("predictions", case, "incumbent_B")
            old = load(path("predictions", case, "incumbent"))
            if out.exists():
                if not compile_only.exists():
                    save(
                        compile_only,
                        (
                            load(out)
                            if old["model"] is None
                            else dict(
                                model=old["model"],
                                action="already_valid",
                                feedback_calls=0,
                            )
                        ),
                    )
                continue
            cid = case["id"]
            ask = Calls(engine)
            base = old["model"]
            if base is None:
                repair = draft(
                    case,
                    ask,
                    model_prompt(case)
                    + "\nRepair the invalid draft. Parser: "
                    + str(old.get("error"))
                    + "\nDraft: "
                    + old.get("raw", ""),
                    seed(cid, "repair"),
                )
                decision = dict(
                    model=repair["model"],
                    action=(
                        "compiled" if repair["model"] is not None else "invalid_draft"
                    ),
                )
                candidate_calls = 0
            else:
                item = dict(case=case, seed=17, baseline=base)
                cached = prior.artifact(
                    "candidates", case, old_model(case) + "_feedback"
                )
                shared = load(cached) if cached.exists() else policy.shared(item, ask)
                decision = policy.decide(item, "C", shared, ask)
                candidate_calls = len(shared.get("calls", [])) if cached.exists() else 0
            final = dict(decision, feedback_calls=len(ask.log) + candidate_calls)
            save(out, final)
            save(
                compile_only,
                (
                    final
                    if base is None
                    else dict(model=base, action="already_valid", feedback_calls=0)
                ),
            )


def improved(engine, manifest, subset):
    for dataset in ("nlco", "dcp"):
        restore(engine, RUN / "trained" / dataset / "adapter.pt")
        print("new modeling", dataset, flush=True)
        for case in (c for c in subset if c["dataset"] == dataset):
            out = path("predictions", case, "new")
            if not out.exists():
                save(
                    out,
                    draft(
                        case,
                        Calls(engine),
                        model_prompt(case),
                        seed(case["id"], "model"),
                    ),
                )
    restore(engine, RUN / "trained/feedback/adapter.pt")
    print("new feedback proposals", len(subset), flush=True)
    for case in subset:
        for model in ("incumbent", "new"):
            dest = path("proposals", case, model)
            if not dest.exists():
                pred = load(path("predictions", case, model))
                ask = Calls(engine)
                proposal = feedback.propose(case, pred, ask)
                save(dest, dict(proposal, proposer_calls=len(ask.log)))
    # Frozen judge per dataset, identical for both proposer inputs.
    for dataset in ("nlco", "dcp"):
        restore(engine, manifest["checkpoints"][dataset])
        print("witness verification", dataset, flush=True)
        for case in (c for c in subset if c["dataset"] == dataset):
            for model in ("incumbent", "new"):
                out = path("predictions", case, model + "_F")
                if out.exists():
                    continue
                pred = load(path("predictions", case, model))
                proposal = load(path("proposals", case, model))
                ask = Calls(engine)
                decision = feedback.verify_patch(case, pred, proposal, ask)
                save(
                    out,
                    dict(
                        decision,
                        feedback_calls=proposal["proposer_calls"] + len(ask.log),
                    ),
                )
    evaluate_penalty(engine, subset)


def evaluate_penalty(engine, subset):
    import torch

    folder = RUN / "trained/penalty"
    support = load(RUN / "training.json")["support"]
    print("penalty episodes", len(subset), support, flush=True)
    if support:
        restore(engine, folder / "adapter.pt")
        ids = tokens(engine)
    elif not (folder / "abstain.json").exists():
        raise ValueError("Missing penalty training output")

    def choose(spec, observed, remaining):
        _, rates, _ = penalty.observations(spec, observed)
        allowed = [
            i
            for i, a in enumerate(penalty.ACTIONS)
            if a in support and any(penalty.weights(rates, a) != 3)
        ]
        if not allowed:
            return "fixed"
        with torch.inference_mode():
            scores = optimization.logits(
                engine, penalty.prompt(spec, observed, remaining), ids
            )
        winner = allowed[int(scores[allowed].argmax())]
        return (
            penalty.ACTIONS[winner]
            if float(scores[winner] - scores[0]) > 0.05
            else "fixed"
        )

    for case in subset:
        memo = {}
        for model in ("incumbent", "new", "incumbent_F", "new_F"):
            spec = load(path("predictions", case, model))["model"]
            for s in seed_list(case, "learned_penalty"):
                out = path("episodes", case, f"{model}__learned_penalty", s)
                key = (value_digest(spec), s)
                if out.exists():
                    memo[key] = load(out)
                    continue
                if key not in memo:
                    memo[key] = (
                        penalty.episode(spec, case["id"], s, choose)
                        if support
                        else dict(reuse_schedule=True)
                    )
                save(out, memo[key])


def worker(stage, task):
    import torch
    from experiments.posttraining_v1.engine import Engine

    if (
        torch.cuda.device_count() != 1
        or not torch.cuda.is_bf16_supported()
        or torch.cuda.get_device_properties(0).total_memory < 75 * 1024**3
    ):
        raise ValueError("Expected one visible BF16 GPU of at least 75 GiB")
    torch.manual_seed(17)
    manifest = state()
    engine = Engine(Path(manifest["model"]))
    if task in TRAINERS:
        train(engine, manifest, task)
    else:
        subset = load(RUN / "cases.json")[int(task) :: 4]
        execute = current if stage == "current" else improved
        execute(engine, manifest, subset)
        save(RUN / "gpu" / f"{stage}_{task}.json", dict(cases=len(subset)))


def run_workers(stage, tasks, devices):
    with contextlib.ExitStack() as stack:
        processes = []
        try:
            for task, device in zip(tasks, devices):
                logpath = RUN / "logs" / f"{stage}_{task}.log"
                log = stack.enter_context(logpath.open("ab"))
                cmd = [
                    sys.executable,
                    "-m",
                    "experiments.pbit_stack_v1.run",
                    "worker",
                    stage,
                    str(task),
                ]
                p = subprocess.Popen(
                    cmd,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    env=dict(os.environ, CUDA_VISIBLE_DEVICES=device),
                )
                processes.append((p, logpath))
            while True:
                for p, logpath in processes:
                    if p.poll() not in (0, None):
                        raise RuntimeError(f"Worker failed: {logpath}")
                if all(p.poll() == 0 for p, _ in processes):
                    return
                time.sleep(1)
        finally:
            for p, _ in processes:
                if p.poll() is None:
                    p.terminate()
            for p, _ in processes:
                try:
                    p.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    p.kill()
                    p.wait()


def launch(stage):
    import torch

    devices = os.getenv("CUDA_VISIBLE_DEVICES", "0,1,2,3").split(",")
    if len(set(devices)) != 4 or len(devices) != 4 or torch.cuda.device_count() != 4:
        raise ValueError("Allocate four visible GPUs")
    with worker_lock(RUN / "gpu.lock"), contextlib.ExitStack() as clean:

        def interrupted(*_):
            raise KeyboardInterrupt("GPU job interrupted")

        clean.callback(
            signal.signal, signal.SIGTERM, signal.signal(signal.SIGTERM, interrupted)
        )
        manifest = state(verify=True)
        for file, expected in manifest["model_files"].items():
            if sha(Path(manifest["model"]) / file) != expected:
                raise ValueError(f"Backbone changed: {file}")
        (RUN / "logs").mkdir(exist_ok=True)
        if stage == "improved":
            corpus = load(RUN / "training.json")
            if not (RUN / "current.json").exists():
                raise ValueError("Current evaluation is not complete")
            if any(digest(Path(p)) != d for p, d in corpus["files"].items()):
                raise ValueError("Training corpus changed")
            tasks = [
                a
                for a in TRAINERS
                if not any(
                    (RUN / "trained" / a / k).exists()
                    for k in ("adapter.pt", "abstain.json")
                )
            ]
            run_workers(stage, tasks, devices)
        shards = [
            i for i in range(4) if not (RUN / "gpu" / f"{stage}_{i}.json").exists()
        ]
        run_workers(stage, shards, devices)
