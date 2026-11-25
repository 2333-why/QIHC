"""Frozen E005-1.18 cohorts and separate result paths for E005-1.19."""

import os
from pathlib import Path

from experiments.pbit_dual_v1 import common as prior
from qihc.experiment_runtime.storage import digest, load

ROOT = Path(__file__).resolve().parents[2]
RUN = Path(os.getenv("QIHC_STACK_RUN", prior.BASE / "results/E005-1.19_pbit_stack"))
HISTORY = Path(os.getenv("QIHC_HISTORY", prior.BASE / "results/E005-1.7_eight"))
E003 = Path(os.getenv("QIHC_E003_RUN", prior.BASE / "results/E003-1.1_public_solvers"))
OLD = prior.RUN
SEED = 17
SEARCH = prior.SEARCH
TRADITIONAL = ("cp_sat", "scip", "highs", "sa", "pt", "penalty_pt", "pbit")
CONVENTIONAL = ("base", "sft", "solver_rl", "sdpo", "pvcd")
TRAINERS = ("nlco", "dcp", "feedback", "penalty")


def path(kind, case, arm, seed=None):
    group = arm if seed is None else f"{arm}__{seed}"
    return RUN / kind / case["dataset"] / group / (case["id"] + ".json")


def old_model(case):
    return "weighted" if case["dataset"] == "nlco" else "trained"


def seed_list(case, terminal):
    return (17, 29, 43) if terminal != "scip" and case["dataset"] == "dcp" else (17,)


def arms(case, stage):
    names = (
        ("incumbent", "incumbent_B", "incumbent_C", "incumbent_pbit")
        if stage == "current"
        else ("new", "incumbent_F", "new_F")
    )
    for name in names:
        for terminal in ("schedule", "scip"):
            yield name, terminal
    if stage == "current" and case["dataset"] == "nlco":
        for name in CONVENTIONAL:
            for terminal in ("schedule", "scip"):
                yield name, terminal
    if stage == "improved":
        for name in ("incumbent", "new", "incumbent_F", "new_F"):
            for terminal in ("rule_penalty", "learned_penalty"):
                yield name, terminal


def state(verify=False):
    manifest = load(RUN / "manifest.json")
    if verify:
        prior.verify()
        for name, sha in manifest["files"].items():
            if digest(Path(name)) != sha:
                raise ValueError(f"Frozen input changed: {name}")
    return manifest
