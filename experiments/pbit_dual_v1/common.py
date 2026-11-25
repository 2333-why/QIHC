"""Frozen inputs for the two evaluation cohorts."""

import os
from pathlib import Path

from experiments.dcp_external_v2.common import SEARCH
from qihc.experiment_runtime.storage import digest, load, save

HERE = Path(__file__).resolve().parent
BASE = Path(os.getenv("QIHC_BASE", "/inspire/hdd/global_user/czxs25210022/why"))
RUN = Path(os.getenv("QIHC_DUAL_RUN", BASE / "results/E005-1.18_pbit_dual"))
OLD14 = Path(os.getenv("QIHC_114_RUN", BASE / "results/E005-1.14_four"))
OLD15 = Path(os.getenv("QIHC_115_RUN", BASE / "results/E005-1.15_four"))
OLD17 = Path(os.getenv("QIHC_DCP_RUN", BASE / "results/E005-1.17_dcp235"))
MODELS = (
    "weighted",
    "trained",
    "weighted_quadratic",
    "weighted_feedback",
    "trained_feedback",
)


def freeze(path, value):
    if path.is_file():
        if load(path) != value:
            raise ValueError(f"Frozen artifact differs at {path}")
    else:
        save(path, value)


def cases():
    return load(RUN / "cases.json")


def artifact(kind, case, name, random_seed=None):
    group = name if random_seed is None else f"{name}__{random_seed}"
    return RUN / kind / case["dataset"] / group / (case["id"] + ".json")


def methods(case):
    for model in MODELS:
        if model != "weighted":
            yield model, "schedule"
            if case["dataset"] == "dcp":
                yield model, "scip"
        if model in ("weighted", "trained", "trained_feedback"):
            yield model, "control"


def original_artifact(case):
    return RUN / "frozen" / case["dataset"] / (case["id"] + ".json")


def get_prediction(case, name):
    if name == "weighted":
        return load(original_artifact(case))["prediction"]
    return load(artifact("predictions", case, name))


def seeds(case, method):
    return (17, 29, 43) if case["dataset"] == "dcp" and method != "scip" else (17,)


def verify(checkpoints=False):
    state = load(RUN / "manifest.json")
    for path, expected in state["files"].items():
        if digest(Path(path)) != expected:
            raise ValueError(f"Frozen input changed: {path}")
    if checkpoints:
        for name, expected in state["adapters"].items():
            if digest(Path(state["paths"][name])) != expected:
                raise ValueError(f"Frozen adapter changed: {name}")
        for name, expected in state["model_files"].items():
            from experiments.dcp_external_v2.identity import sha

            if sha(Path(state["model"]) / name) != expected:
                raise ValueError(f"Frozen backbone changed: {name}")
    return state
