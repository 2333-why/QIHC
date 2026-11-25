"""Constraint-selective exploratory penalties and paired training rollouts."""

from copy import deepcopy
import json
import time

import numpy as np

from experiments.pbit_dual_v1.methods import bit_responses
from experiments.pbit_joint_v1 import search
from qihc.experiment_runtime.pbit_control import key, quality, seed
from qihc.generic.factor_search import encoding
from qihc.generic.feedback.search import Energy, sample
from .common import SEARCH

ACTIONS = ("fixed", "harden", "soften", "redistribute")


def observations(spec, state):
    x = np.asarray(state["x"], dtype=int)
    energy = Energy(spec)
    _, residual, _ = energy.values(x)
    rates = (residual > 0).mean(axis=0)
    return x, rates, float(np.mean((residual == 0).all(axis=1)))


def weights(rates, action):
    if action == "harden":
        return 3 + 3 * rates
    if action == "soften":
        return 3 - 1.5 * rates
    if action == "redistribute" and len(rates):
        return np.clip(3 + 3 * (rates - rates.mean()), 1.5, 6)
    return np.full(len(rates), 3.0)


def advance(spec, state, action, rng, steps):
    if action not in ACTIONS:
        raise ValueError(action)
    x, rates, _ = observations(spec, state)
    w = weights(rates, action)
    if not len(w) or np.all(w == 3):
        out = search.advance(spec, state, "cool", rng, steps)
        if encoding(spec)[0]:
            out["penalty_writes"] = state.get("penalty_writes", 0)
        return out
    out = deepcopy(state)
    temp = max(0.05, state["temperature"] / 2)
    half = len(x) // 2
    energy = Energy(spec)
    parts = []
    for part in range(2):
        row = sample(
            spec,
            3600,
            seed(rng, part),
            chains=half,
            initial=x[part * half : (part + 1) * half],
            temperature=temp,
            weights=w if part else None,
            max_steps=steps,
            read_q=True,
        )
        if row["transition_attempts"] != half * steps:
            raise RuntimeError("Incomplete penalty window")
        f, r, _ = energy.values(np.asarray([row["x"]]))
        candidate = dict(x=row["x"], objective=float(f[0]), violation=float(r.sum()))
        if key(candidate, spec) < key(out["best"], spec):
            out["best"] = candidate
        parts.append(row)
    out["x"] = parts[0]["final"] + parts[1]["final"]
    out["q"] = parts[0]["q"]
    out["temperature"] = temp
    out["steps"] += steps
    out["penalty_writes"] = state.get("penalty_writes", 0) + 2 * int(
        np.count_nonzero(w != 3)
    )
    out["history"].append(
        dict(
            action=action,
            best=out["best"],
            feasible_mass=float(np.mean([p["feasible_mass"] for p in parts])),
            diversity=len(np.unique(out["x"], axis=0)) / len(x),
        )
    )
    return out


def prompt(spec, state, remaining):
    x, rates, mass = observations(spec, state)
    sites, _ = encoding(spec)
    selected = sorted(state["q"], key=lambda v: abs(v["conditional_q_mean"] - 0.5))[:8]
    response = bit_responses(
        spec, x, state["temperature"], [(v["bit"], sites[v["bit"]]) for v in selected]
    )
    data = dict(
        temperature=state["temperature"],
        windows=remaining,
        feasibility=mass,
        population_diversity=len(np.unique(x, axis=0)) / len(x),
        best_violation=state["best"]["violation"],
        objective=state["best"]["objective"],
        rates=sorted(rates.tolist(), reverse=True)[:16],
        conditionals={k: v.mean(axis=0).round(4).tolist() for k, v in response.items()},
    )
    return (
        "Choose a penalty for the exploratory half of a p-bit population for one window. "
        "A fixed, B harden violated constraints, C soften violated constraints, "
        "D redistribute weight toward often violated constraints. Protected half always fixed. "
        "Archive prefers original feasibility then objective. Conditional columns are "
        "quadratic residual, linear residual, violation count. Return A, B, C or D.\n"
        + json.dumps(data, separators=(",", ":"))
        + "\nAction:"
    )


def rule(spec, state, _):
    _, rates, mass = observations(spec, state)
    if not len(rates) or not rates.max():
        return "fixed"
    if mass < 0.15:
        return "harden"
    if mass > 0.75:
        return "soften"
    return "redistribute"


def returns(spec, outcome):
    x, _, _ = observations(spec, outcome)
    energy = Energy(spec)
    f, residual, _ = energy.values(x)
    values = [
        quality(spec, dict(objective=float(v), violation=float(r.sum())))
        for v, r in zip(f, residual)
    ]
    return quality(spec, outcome["best"]), float(np.mean(values))


def labels(spec, cid, cfg=SEARCH):
    if not encoding(spec)[0] or not spec["constraints"] and not spec["logic"]:
        return []
    rows = []
    steps = cfg["window_steps"]
    for t in (0.25, 1.0):
        st = search.initial(spec, seed(cid, t, "init"), cfg["chains"])
        st["temperature"] = t
        for i in range(cfg["decisions"]):
            remaining = cfg["decisions"] - i
            text = prompt(spec, st, remaining)
            evidence = {a: [] for a in ACTIONS[1:]}
            for repeat in range(3):
                rng = seed(cid, t, i, repeat, "label")
                base = returns(spec, advance(spec, st, "fixed", rng, steps))
                for action in evidence:
                    values = returns(spec, advance(spec, st, action, rng, steps))
                    evidence[action].append([values[0] - base[0], values[1] - base[1]])
            for action, gains in evidence.items():
                primary, population = np.asarray(gains).T
                # Population progress contributes only when the archive is tied.
                payoff = np.where(
                    abs(primary) > 1e-8,
                    np.tanh(primary / 0.04),
                    0.5 * np.tanh(population / 0.10),
                )
                rows.append(
                    dict(
                        case_id=cid,
                        text=text,
                        action=action,
                        gains=gains,
                        target=float(payoff.mean()),
                    )
                )
            st = advance(spec, st, "fixed", seed(cid, t, i, "behavior"), steps)
    return rows


def episode(spec, cid, seed_value, choose, cfg=SEARCH):
    if spec is None:
        return dict(x=None, status="input_invalid")
    if not encoding(spec)[0]:
        return search.episode(spec, cid, dict(cfg, seed=seed_value), lambda *_: "cool")
    start = time.monotonic()
    steps = cfg["window_steps"]
    st = search.initial(spec, seed(cid, seed_value, "init"), cfg["chains"])
    st = search.advance(spec, st, "cool", seed(cid, "warm"), steps)
    actions = []
    decision = 0.0
    for i in range(cfg["decisions"]):
        tick = time.monotonic()
        action = choose(spec, st, cfg["decisions"] - i)
        decision += time.monotonic() - tick
        st = advance(spec, st, action, seed(cid, seed_value, i, "eval"), steps)
        actions.append(action)
    return dict(
        x=st["best"]["x"],
        actions=actions,
        attempts=st["steps"] * cfg["chains"],
        penalty_writes=st.get("penalty_writes", 0),
        seconds=time.monotonic() - start,
        decision_seconds=decision,
    )
