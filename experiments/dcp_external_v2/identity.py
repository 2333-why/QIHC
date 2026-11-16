"""Bind frozen checkpoints, model, code and all 235 local reference files."""

import hashlib
import os
from pathlib import Path
import subprocess
from qihc.experiment_runtime.identity import model_identity
from qihc.experiment_runtime.storage import load, digest
from .common import BASE, RUN, ROOT, cases, frozen


def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def source_hashes():
    roots = [ROOT / "qihc"] + [
        ROOT / "experiments" / name
        for name in (
            "dcp_external_v2",
            "boundary_feedback_v1",
            "pbit_joint_v1",
            "pbit_joint_v2",
            "posttraining_v1",
            "feedback_active_v3",
            "pvcd_cycle_v1",
        )
    ]
    return {
        str(p.relative_to(ROOT)): sha(p) for root in roots for p in root.rglob("*.py")
    }


def code_id():
    dirs = [
        "qihc",
        "experiments/dcp_external_v2",
        "experiments/boundary_feedback_v1",
        "experiments/pbit_joint_v1",
        "experiments/pbit_joint_v2",
        "experiments/posttraining_v1",
        "experiments/feedback_active_v3",
        "experiments/pvcd_cycle_v1",
    ]
    if subprocess.check_output(
        ["git", "status", "--porcelain", "--", *dirs], cwd=ROOT
    ).strip():
        raise RuntimeError("Experiment source differs from checkout")
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def bind():
    from experiments.feedback_active_v3.check_local_model import check
    from importlib.metadata import version

    for pkg in (
        "torch",
        "transformers",
        "ortools",
        "cpmpy",
        "highspy",
        "pyscipopt",
        "httpx",
    ):
        print(pkg, version(pkg), flush=True)
    old = Path(os.getenv("QIHC_114_RUN", BASE / "results/E005-1.14_four"))
    controls = Path(os.getenv("QIHC_115_RUN", BASE / "results/E005-1.15_four"))
    old_manifest = load(old / "manifest.json")
    if old_manifest["experiment_id"] != "E005-1.14":
        raise ValueError("Expected E005-1.14 manifest")
    model = Path(old_manifest["model"])
    check(model)
    expected = {r["path"]: r["sha256"] for r in old_manifest["model_files"]}
    files = {r["path"]: sha(model / r["path"]) for r in model_identity(model)}
    if expected != files:
        raise ValueError("Base model differs from E005-1.14")
    for suffix in ("*.txt", "*.tiktoken"):
        for p in model.glob(suffix):
            files[p.name] = sha(p)
    history = (
        Path(os.getenv("QIHC_HISTORY", BASE / "results/E005-1.7_eight"))
        / "formal/models/17"
    )
    adapters = {
        a: history / a / "adapter.pt" for a in ("sft", "solver_rl", "sdpo", "pvcd")
    }
    adapters.update(
        parent=Path(old_manifest["adapter"]),
        weighted=old / "trained/weighted/adapter.pt",
        rank=controls / "trained/rank/adapter.pt",
        balanced=controls / "trained/balanced/adapter.pt",
    )
    winners = load(controls / "ready.json")["diagnostics"]
    state = dict(
        code=code_id(),
        source=source_hashes(),
        model=str(model),
        model_files=files,
        public=digest(RUN / "public.json"),
        references={
            r["id"]: sha(RUN / "reference" / (r["id"] + ".json")) for r in cases()
        },
        paths={a: str(p) for a, p in adapters.items()},
        adapters={a: sha(p) for a, p in adapters.items()},
        winners={a: winners[a]["winners"] for a in ("rank", "balanced")},
    )
    frozen(RUN / "manifest.json", state)


def verify(models=False):
    state = load(RUN / "manifest.json")
    if (
        source_hashes() != state["source"]
        or digest(RUN / "public.json") != state["public"]
    ):
        raise ValueError("Run identity changed")
    if any(
        sha(RUN / "reference" / (cid + ".json")) != h
        for cid, h in state["references"].items()
    ):
        raise ValueError("Reference files changed")
    if models:
        if any(
            sha(Path(state["model"]) / k) != h for k, h in state["model_files"].items()
        ):
            raise ValueError("Base model changed")
        if any(sha(state["paths"][a]) != h for a, h in state["adapters"].items()):
            raise ValueError("Adapter changed")
    return state
