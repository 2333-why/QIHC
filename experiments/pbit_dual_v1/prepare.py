"""Bind completed runs and build training evidence only from NLCO train."""

from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import os
from pathlib import Path
import subprocess

from experiments.dcp_external_v2.common import ARMS, repeats
from experiments.pvcd_cycle_v1.evidence import cards_from_states
from qihc.experiment_runtime.pbit_control import seed
from qihc.experiment_runtime.storage import digest, load, save, worker_lock
from .common import HERE, OLD14, OLD15, OLD17, RUN, SEARCH, freeze
from .methods import clamp_examples, probe, training_controls


def training_evidence(task):
    row, prior = task
    cid = row["case"]["id"]
    out = RUN / "training" / (cid + ".json")
    if out.exists():
        return load(out)["counts"]
    gold, candidate = row["gold"], prior["candidate"]
    cards = list(prior["cards"])
    probes = []
    if candidate is not None:
        states = []
        for replicate in range(3):
            points, stats = probe(
                (candidate, gold), seed(cid, "train_probe", replicate)
            )
            states.extend(points)
            probes.append(stats)
        if states:
            cards = cards_from_states(candidate, gold, states, limit=64) + cards
    controls = training_controls(gold, cid + "/gold", SEARCH)
    if candidate is not None and candidate != gold:
        controls += training_controls(candidate, cid + "/draft", SEARCH)
    augmented = clamp_examples(row["case"], gold)
    for child in augmented:
        controls += training_controls(
            child["gold"], child["case"]["id"], SEARCH, (0.25, 1.0)
        )
    counts = dict(
        cards=len(cards) + len(augmented),
        controls=len(controls),
        augmented=len(augmented),
        positive=dict(Counter(r["action"] for r in controls if r["target"] > 0.10)),
    )
    save(
        out,
        dict(
            case=row["case"],
            gold=gold,
            candidate=candidate,
            cards=cards,
            controls=controls,
            probes=probes,
            counts=counts,
            augmented=augmented,
        ),
    )
    return counts


def prepare():
    outputs = RUN.resolve()
    for old in (OLD14, OLD15, OLD17):
        old = old.resolve()
        if outputs == old or outputs in old.parents or old in outputs.parents:
            raise ValueError("Use an independent E005-1.18 result path")
    with worker_lock(RUN / "prepare.lock"):
        bind()


def bind():
    files = {}

    def read(path):
        files[str(path.resolve())] = digest(path)
        return load(path)

    train = read(OLD14 / "inputs/train.json")
    ready = read(OLD14 / "ready.json")
    old14 = read(OLD14 / "manifest.json")
    old15 = read(OLD15 / "manifest.json")
    old17 = read(OLD17 / "manifest.json")
    old15_report = read(OLD15 / "report.json")
    old17_report = read(OLD17 / "report.json")
    if (
        old14["experiment_id"],
        old15["experiment_id"],
        old15_report["evaluations"],
    ) != ("E005-1.14", "E005-1.15", 7020):
        raise ValueError(
            "Expected completed E005-1.15 and frozen E005-1.14 training data"
        )
    for group, total in (("all235", 20210), ("main159", 13674)):
        if old17_report[group]["all_seed_completed"] != total:
            raise ValueError(f"Incomplete E005-1.17 group {group}")
    if old14["model"] != old17["model"]:
        raise ValueError("Historical runs use different backbone paths")
    if (
        old14["inputs"]["train.json"]
        != files[str((OLD14 / "inputs/train.json").resolve())]
    ):
        raise ValueError("Changed E005-1.14 training file")
    gold = {
        r["id"]: r["gold"] for path in old14["gold_paths"] for r in read(Path(path))
    }
    cohort, inputs = [], {}
    nlco = [
        c for c in read(OLD15 / "inputs/cohort.json") if c["stage"] in ("main", "scale")
    ]
    if Counter(c["stage"] for c in nlco) != {"main": 480, "scale": 300}:
        raise ValueError("Unexpected NLCO cohort")
    dcp = read(OLD17 / "public.json")
    if (len(dcp), sum(c["main"] for c in dcp)) != (235, 159):
        raise ValueError("Unexpected DCP cohort")
    if files[str((OLD17 / "public.json").resolve())] != old17["public"]:
        raise ValueError("Changed DCP manifest input")
    train_ids = {r["case"]["id"] for r in train}
    if len({c["id"] for c in nlco + dcp}) != 1015:
        raise ValueError("Duplicate evaluation identifiers")
    if (
        len(train) != 90
        or len(train_ids) != 90
        or train_ids & {c["id"] for c in nlco + dcp}
    ):
        raise ValueError("Unexpected training overlap or size")
    for dataset, subset, old in (("nlco", nlco, OLD15), ("dcp", dcp, OLD17)):
        for case in subset:
            cid = case["id"]
            case = dict(case, dataset=dataset)
            cohort.append(case)
            prediction = read(old / "predictions/weighted" / (cid + ".json"))
            if dataset == "nlco":
                original = read(OLD14 / "predictions/weighted" / (cid + ".json"))
                if original["model"] != prediction["model"]:
                    raise ValueError(f"Changed NLCO weighted prediction {cid}")
                ref = read(old / "references" / (cid + ".json"))
                legacy = read(old / "scores" / (cid + ".json"))["cells"]
                first = read(OLD14 / "scores" / (cid + ".json"))["cells"]
                value = dict(
                    prediction=prediction,
                    gold=gold[cid],
                    reference=ref,
                    legacy=legacy,
                    legacy14=first,
                )
            else:
                ref = old / "reference" / (cid + ".json")
                read(ref)
                if files[str(ref.resolve())] != old17["references"][cid]:
                    raise ValueError(f"Changed DCP reference {cid}")
                inputs[str(ref.resolve())] = files[str(ref.resolve())]
                legacy = {}
                for m, method in ARMS:
                    arm = f"{m}__{method}"
                    for s in repeats(m, method):
                        stage = "results" if s == 17 else f"results{s}"
                        legacy[f"{arm}__{s}"] = read(
                            old / stage / arm / (cid + ".json")
                        )
                value = dict(
                    prediction=prediction, reference=str(ref.resolve()), legacy=legacy
                )
            path = RUN / "frozen" / dataset / (cid + ".json")
            freeze(path, value)
            inputs[str(path.resolve())] = digest(path)
    freeze(RUN / "cases.json", cohort)
    inputs[str((RUN / "cases.json").resolve())] = digest(RUN / "cases.json")
    source = sorted(HERE.glob("*.py")) + sorted(HERE.glob("*.sh"))
    legacy_sources = old17["source"]
    repo = HERE.parents[1]
    for path, fingerprint in legacy_sources.items():
        if digest(repo / path) != fingerprint:
            raise ValueError(f"E005-1.17 implementation changed: {path}")
    inputs.update({str(p): digest(p) for p in source})
    inputs.update({str(repo / p): h for p, h in legacy_sources.items()})
    replay_module = repo / "experiments/pbit_joint_v3/methods.py"
    inputs[str(replay_module)] = digest(replay_module)
    adapters = {m: digest(Path(old17["paths"][m])) for m in ("parent", "weighted")}
    if any(adapters[m] != old17["adapters"][m] for m in adapters):
        raise ValueError("Historical adapter hash mismatch")
    config = dict(
        experiment="E005-1.18",
        code=subprocess.check_output(
            ["git", "log", "-1", "--format=%H", "--", "experiments/pbit_dual_v1"],
            cwd=repo,
            text=True,
        ).strip(),
        model=old17["model"],
        model_files=old17["model_files"],
        paths=old17["paths"],
        adapters=adapters,
        inputs=inputs,
        sources=files,
    )
    freeze(RUN / "preparation.json", config)
    jobs = []
    for row in train:
        cid = row["case"]["id"]
        file = OLD14 / "evidence" / (cid + ".json")
        if ready["files"][cid] != digest(file):
            raise ValueError(f"Changed training evidence {cid}")
        jobs.append((row, load(file)))
    with ProcessPoolExecutor(
        max_workers=int(os.getenv("QIHC_CPU_WORKERS", "6"))
    ) as pool:
        totals = Counter()
        positive = Counter()
        for i, counts in enumerate(pool.map(training_evidence, jobs), 1):
            totals.update({k: counts[k] for k in ("cards", "controls", "augmented")})
            positive.update(counts["positive"])
            if i % 10 == 0:
                print("training evidence", i, "/", len(jobs), flush=True)
    training = [
        str((RUN / "training" / (r["case"]["id"] + ".json")).resolve()) for r in train
    ]
    bound = dict(inputs, **{p: digest(Path(p)) for p in training})
    freeze(
        RUN / "manifest.json",
        dict(
            config,
            files=bound,
            training=training,
            supervision=dict(totals, positive=dict(positive)),
        ),
    )
    print("PREPARED", len(cohort), dict(totals), dict(positive), flush=True)
