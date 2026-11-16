"""Frozen DCP cohort, method matrix and filesystem layout."""

import os
from pathlib import Path
from qihc.experiment_runtime.storage import load, save

ROOT = Path(__file__).resolve().parents[2]
BASE = Path(os.getenv("QIHC_BASE", "/inspire/hdd/global_user/czxs25210022/why"))
RUN = Path(os.getenv("QIHC_DCP_RUN", BASE / "results/E005-1.17_dcp235"))
UP = Path(os.getenv("QIHC_DCP_UPSTREAM", BASE / "DCP-Bench-Open-v0.3.0"))
ASSET = Path(
    os.getenv(
        "QIHC_DCP_ASSET", BASE / "results/E005-1.16_dcp/dcp-bench-open-v0.3.0.jsonl"
    )
)
CENSUS = (
    ROOT / "docs/experiments/E005_posttraining/E005-1.17_inventory_20261116_001.json"
)
SEED = 17
SEARCH_SEEDS = (17, 29, 43)
SECONDS = 20
SEARCH = dict(seed=SEED, chains=32, decisions=4, window_steps=128)
MODELS = ("base", "sft", "parent", "solver_rl", "sdpo", "pvcd", "weighted")
FEEDBACK = {"base": ("B", "C", "S", "P", "Q"), "pvcd": ("C", "Q")}
GOLD = ("cp_sat", "scip", "highs", "sa", "pt", "pbit")
PRED = MODELS + tuple(f"{m}_{a}" for m, arms in FEEDBACK.items() for a in arms)
ARMS = (
    tuple(("gold", b) for b in GOLD)
    + tuple((m, b) for m in PRED for b in ("schedule", "scip"))
    + tuple(
        (m, b) for m in ("parent", "weighted") for b in ("rule", "rank", "balanced")
    )
)


def cases():
    return load(RUN / "public.json")


def path(kind, name, cid):
    return RUN / kind / name / (cid + ".json")


def repeats(model, method):
    stochastic = method in ("sa", "pt", "pbit") if model == "gold" else method != "scip"
    return SEARCH_SEEDS if stochastic else (SEED,)


def stage_name(kind, seed):
    return kind if seed == SEED else kind + str(seed)


def rowpath(model, method, cid, seed=SEED):
    return path(stage_name("results", seed), model + "__" + method, cid)


def frozen(path, contents):
    if path.exists() and load(path) != contents:
        raise ValueError(f"Frozen input changed at {path}. Use a fresh QIHC_DCP_RUN.")
    if not path.exists():
        save(path, contents)
