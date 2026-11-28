"""E005-1.20 paths, frozen input checks and method definitions."""

import os
import subprocess
from pathlib import Path

from experiments.pbit_dual_v1 import common as prior
from qihc.experiment_runtime.storage import digest, load

ROOT = Path(__file__).resolve().parents[2]
RUN = Path(
    os.getenv("QIHC_TRANSFER_RUN", prior.BASE / "results/E005-1.20_dcp_transfer")
)
NEW = ("replay", "mixed", "feedback_only", "joint")
HIST = ("base", "anchor", "champion", "old_feedback")
OTHER = ("sft", "solver_rl", "sdpo", "pvcd", "double_review")
TRADITIONAL = ("cp_sat", "scip", "highs", "sa", "pt", "pbit")
TERMINALS = ("schedule", "scip")
TASKS = ("nlco_replay", "dcp_replay", "nlco_mixed", "dcp_mixed", "feedback")


def path(kind, case, name, s=None):
    group = name if s is None else f"{name}__{s}"
    return RUN / kind / case["dataset"] / group / (case["id"] + ".json")


def historical(case):
    nlco = case["dataset"] == "nlco"
    return dict(
        base="base",
        anchor="new" if nlco else "incumbent",
        old_feedback="new_F" if nlco else "incumbent_F",
        champion="new_F" if nlco else "incumbent_C",
    )


def solver_seeds(case, method):
    return (
        (17, 29, 43)
        if case["dataset"] == "dcp" and method in ("sa", "pt", "pbit")
        else (17,)
    )


def version():
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def manifest(verify=False):
    m = load(RUN / "manifest.json")
    current = version()
    if m["code"] != current:
        # These two published preparations differ only in documentation. The
        # recovery release changes feedback activation storage and reporting.
        supported = {
            "d71ee91c414c41965323c535ab039387ede7a5d3",
            "8e3d6e76636bcff56b1001fcf912d324f05676bd",
        }
        recovery_parent = "8e3d6e76636bcff56b1001fcf912d324f05676bd"
        parent = subprocess.check_output(
            ["git", "rev-parse", current + "^"], cwd=ROOT, text=True
        ).strip()
        if (
            m["code"] not in supported
            or parent != recovery_parent
            or subprocess.run(
                [
                    "git",
                    "diff",
                    "--quiet",
                    m["code"],
                    current,
                    "--",
                    "qihc",
                    "experiments",
                    ":(exclude)experiments/dcp_transfer_v1/common.py",
                    ":(exclude)experiments/dcp_transfer_v1/gpu.py",
                    ":(exclude)experiments/dcp_transfer_v1/score.py",
                ],
                cwd=ROOT,
            ).returncode
        ):
            raise ValueError("Run from a compatible E005-1.20 recovery checkout")
    if verify:
        for f, h in m["files"].items():
            if digest(Path(f)) != h:
                raise ValueError(f"Frozen input changed: {f}")
    return m
