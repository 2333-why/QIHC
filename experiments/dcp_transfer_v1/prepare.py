"""Materialize template-disjoint training data and immutable paired comparisons."""

import json
import subprocess
from pathlib import Path

from experiments.dcp_external_v2.common import UP
from experiments.dcp_external_v2.inventory import REVISION
from experiments.dcp_external_v2.identity import sha
from experiments.pbit_stack_v1 import common as stack
from experiments.posttraining_v1.data import target
from qihc.experiment_runtime.storage import digest, load, value_digest, worker_lock
from qihc.modeling.active_protocol_v2 import model_prompt
from .common import (
    ROOT,
    RUN,
    TERMINALS,
    OTHER,
    TRADITIONAL,
    historical,
    manifest,
    path,
    prior,
    solver_seeds,
    version,
)
from .data import split, variants
from .feedback import FULL, PATCH, supervise


def training_data(train, tok):
    previous_corpus = load(stack.RUN / "training.json")
    nlco, repairs = [], []
    for f, h in previous_corpus["files"].items():
        if digest(Path(f)) != h:
            raise ValueError(f"Training parent changed: {f}")
        row = load(Path(f))
        nlco.extend(row["modeling"])
        for r in row["repairs"]:
            suffix = PATCH if "edits" in json.loads(r["target"]) else FULL
            repairs.append(dict(r, prompt=r["prompt"] + suffix))
    if len(nlco) != 360 or len(repairs) != 313:
        raise ValueError("Unexpected E005-1.19 replay corpus")
    refs = {
        c["id"]: load(Path(load(prior.original_artifact(c))["reference"]))
        for c in train
    }
    seen = {
        (r["template"], json.dumps(r["data"], sort_keys=True)) for r in refs.values()
    }
    dcp, patches, ledger = [], [], {}
    for c in train:
        rows, audit = variants(c, refs[c["id"]], seen)
        for child, gold, witness in rows:
            model, fix = supervise(child, gold, witness, len(dcp))
            dcp.append(model)
            patches.append(fix)
        ledger[c["id"]] = audit
        print("DCP supervision", c["id"], len(rows), flush=True)

    def usable(prompt, answer):
        length = len(tok.encode(answer, add_special_tokens=False))
        # Conservative overhead allowance for the Qwen user/assistant chat template.
        return length + 1 <= 4096 and len(tok.encode(prompt)) + length + 128 < 32768

    pools = dict(nlco=nlco, dcp=dcp, nlco_patch=repairs, dcp_patch=patches)
    excluded = {}
    for name, rows in pools.items():
        if name.endswith("patch"):
            kept = [r for r in rows if usable(r["prompt"], r["target"])]
        else:
            kept = [
                r for r in rows if usable(model_prompt(r["case"]), target(r["gold"]))
            ]
        if not kept:
            raise ValueError(f"No usable examples for {name}")
        excluded[name] = len(rows) - len(kept)
        pools[name] = kept
    return pools, dict(
        variants=ledger,
        token_exclusions=excluded,
        certified_dcp=len(dcp),
        usable={k: len(v) for k, v in pools.items()},
    )


def prepare():
    if (RUN / "manifest.json").exists():
        print("READY", manifest(True)["counts"])
        return
    from transformers import AutoTokenizer

    with worker_lock(RUN / "prepare.lock"):
        report = stack.RUN / "improved_report.json"
        snapshot = (
            ROOT
            / "docs/experiments/E005_posttraining/E005-1.19_improved_report_20261127_001.json"
        )
        if load(report) != load(snapshot):
            raise ValueError("Expected completed E005-1.19 archived report")
        previous = stack.state(verify=True)
        if (
            subprocess.check_output(
                ["git", "-C", str(UP), "rev-parse", "HEAD"], text=True
            ).strip()
            != REVISION
        ):
            raise ValueError("Unexpected DCP release")
        subprocess.run(
            [
                "git",
                "-C",
                str(UP),
                "diff",
                "--quiet",
                "HEAD",
                "--",
                "dataset",
                "evaluation",
            ],
            check=True,
        )
        train, test = split(load(stack.RUN / "cases.json"))
        tok = AutoTokenizer.from_pretrained(
            previous["model"], local_files_only=True, use_fast=True
        )
        pools, audit = training_data(train, tok)
        files = {}

        def freeze(p, obj):
            prior.freeze(p, obj)
            files[str(p)] = digest(p)

        freeze(RUN / "corpus.json", pools)
        freeze(RUN / "training_audit.json", audit)
        freeze(RUN / "cases.json", test)
        freeze(
            RUN / "split.json",
            dict(
                training_candidates=[c["id"] for c in train],
                training_templates=sorted({c["template"] for c in train}),
                dcp_test=[c["id"] for c in test if c["dataset"] == "dcp"],
                dcp_main=[
                    c["id"] for c in test if c.get("main") and c["dataset"] == "dcp"
                ],
            ),
        )
        for c in test:
            old = historical(c)
            freeze(
                path("inputs", c, "anchor"),
                load(stack.path("predictions", c, old["anchor"])),
            )
            fingerprints = {
                name: value_digest(load(stack.path("predictions", c, arm))["model"])
                for name, arm in dict(
                    anchor=old["anchor"],
                    old_feedback=old["old_feedback"],
                    champion=old["champion"],
                    double_review="incumbent_C",
                ).items()
            }
            freeze(path("historical", c, "fingerprints"), fingerprints)
            original_path = prior.original_artifact(c)
            files[str(original_path)] = digest(original_path)
            original = load(original_path)
            if c["dataset"] == "dcp":
                ref = Path(original["reference"])
                files[str(ref)] = digest(ref)
            rows = {}
            sources = {name: name for name in OTHER}
            sources.update(old, double_review="incumbent_C")
            for name, arm in sources.items():
                for t in TERMINALS:
                    for s in stack.seed_list(c, t):
                        key = f"{name}__{t}__{s}"
                        src = stack.path("scores", c, f"{arm}__{t}", s)
                        if src.exists():
                            rows[key] = load(src)
                        else:
                            if c["dataset"] != "dcp":
                                raise FileNotFoundError(src)
                            rows[key] = original["legacy"][f"{arm}__{t}__{s}"]
            for method in TRADITIONAL:
                for s in solver_seeds(c, method):
                    row = (
                        original["legacy"][f"gold__{method}__{s}"]
                        if c["dataset"] == "dcp"
                        else load(stack.path("scores", c, "gold20__" + method, s))
                    )
                    rows[f"traditional__{method}__{s}"] = row
            freeze(path("historical", c, "rows"), rows)
        adapters = dict(
            nlco=str(stack.RUN / "trained/nlco/adapter.pt"),
            dcp=previous["checkpoints"]["dcp"],
            feedback=str(stack.RUN / "trained/feedback/adapter.pt"),
        )
        # Match the verification model of the earlier feedback experiment on each dataset.
        judges = {k: previous["checkpoints"][k] for k in ("nlco", "dcp")}
        for p in set(adapters.values()) | set(judges.values()):
            files[str(p)] = sha(Path(p))
        for f, h in previous["model_files"].items():
            p = Path(previous["model"]) / f
            if sha(p) != h:
                raise ValueError(f"Backbone changed: {p}")
        counts = dict(audit["usable"], test=len(test), main=159, dcp_test=210)
        prior.freeze(
            RUN / "manifest.json",
            dict(
                code=version(),
                model=previous["model"],
                model_files=previous["model_files"],
                adapters=adapters,
                judges=judges,
                files=files,
                counts=counts,
            ),
        )
        print("READY", counts, flush=True)
