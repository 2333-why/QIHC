"""Conditional p-bit evidence and conservative learned interventions."""

from copy import deepcopy
import time

import numpy as np

from experiments.boundary_feedback_v1 import policy
from experiments.pbit_joint_v1 import search
from experiments.pbit_joint_v2.signals import control_prompt
from qihc.experiment_runtime.pbit_control import seed
from qihc.generic.bounded import validate, violations
from qihc.generic.factor_search import encoding
from qihc.generic.feedback.search import (
    Energy,
    conditional_probability,
    encode_states,
    sample,
)

ENERGY_FORMS = ("quadratic_residual", "linear_residual", "violated_count")


def landscape(energy, states):
    objective, residual, _ = energy.values(states)
    r = residual / energy.div
    objective = energy.sign * objective / energy.scale
    return objective[:, None] + 3 * np.stack(
        ((r**2).sum(axis=1), r.sum(axis=1), (r > 0).sum(axis=1)), axis=1
    )


def bit_responses(spec, xs, temperature, sites=None):
    sites = list(enumerate(encoding(spec)[0])) if sites is None else sites
    energy, bits = Energy(spec), encode_states(spec, xs)
    current = landscape(energy, xs)
    response = {}
    for k, (i, w) in sites:
        other = xs.copy()
        bit_zero = bits[:, k, None] == 0
        other[:, i] += (1 - 2 * bits[:, k]) * w
        flipped = landscape(energy, other)
        response[k] = conditional_probability(
            np.where(bit_zero, current, flipped),
            np.where(bit_zero, flipped, current),
            temperature,
        )
    return response


def probe(models, random_seed, steps=128, chains=32, multi=True):
    """Choose clamp sites by conditional disagreement on identical states."""
    first, second = models
    if first["variables"] != second["variables"]:
        return [], {"reason": "different_domains", "attempts": 0}
    sites, _ = encoding(first)
    if not sites:
        return [], {"reason": "fixed_domain", "attempts": 0}
    started = time.monotonic()
    states, attempts = [], 0
    temperature = 0.5
    for model in models:
        row = sample(
            model,
            3600,
            seed(random_seed, "warm"),
            chains=chains,
            temperature=temperature,
            read_q=False,
            max_steps=steps,
        )
        states.extend(row["final"])
        attempts += row["transition_attempts"]
    xs = np.asarray(states)
    left, right = [bit_responses(model, xs, temperature) for model in models]
    divergence = {}
    for k, (i, _) in enumerate(sites):
        by_form = abs(left[k] - right[k]).mean(axis=0)
        divergence[i] = np.maximum(
            divergence.get(i, np.zeros(len(ENERGY_FORMS))), by_form
        )
    targets = sorted(
        divergence,
        key=lambda i: (-float(divergence[i].mean() if multi else divergence[i][0]), i),
    )[:2]
    if len(divergence) > 1:
        for i in targets:
            var = first["variables"][i]
            for endpoint in ("lower", "upper"):
                for model in models:
                    row = sample(
                        model,
                        3600,
                        seed(random_seed, i, endpoint),
                        chains=chains,
                        clamps={i: var[endpoint]},
                        initial=xs[:chains],
                        temperature=temperature,
                        read_q=False,
                        max_steps=steps,
                    )
                    states.extend(row["final"])
                    attempts += row["transition_attempts"]
    return states, dict(
        sites=targets,
        energy_forms=ENERGY_FORMS,
        response=[divergence[i].tolist() for i in targets],
        attempts=attempts,
        seconds=time.monotonic() - started,
    )


def feedback(case, original, shared, ask, multi=True):
    """Keep a candidate only when two disjoint batches corroborate it."""
    if original is None:
        return dict(model=None, action="invalid_draft", evidence=[])
    cid = case["id"]
    candidate = shared["candidate"]
    evidence = []
    if candidate is None or candidate == original:
        return dict(model=original, action=shared["action"], evidence=evidence)
    models = (original, candidate)
    excluded = set()
    for phase in range(2):
        states, telemetry = probe(models, seed(cid, "feedback", phase), multi=multi)
        witnesses, pairs = policy.witnesses(models, states, excluded)
        if not witnesses:
            evidence.append(dict(probe=telemetry))
            return dict(model=original, action="no_fresh_witness", evidence=evidence)
        text = ask(
            policy.judge_prompt(case, witnesses, pairs), seed(cid, "judge", phase)
        )
        verdict = policy.assess(text, case, models, witnesses, pairs)
        evidence.append(
            dict(probe=telemetry, verdict=verdict, assignments=witnesses, pairs=pairs)
        )
        if verdict["status"] != "judged":
            return dict(model=original, action=verdict["status"], evidence=evidence)
        if verdict["covered"] < 2 or verdict["errors"][1] or not verdict["errors"][0]:
            return dict(
                model=original, action="insufficient_support", evidence=evidence
            )
        excluded.update(map(tuple, witnesses))
    return dict(model=candidate, action="accepted", evidence=evidence)


def extended_prompt(spec, state, remaining):
    import json

    xs = np.asarray(state["x"])
    sites = encoding(spec)[0]
    q = sorted(
        state["q"], key=lambda b: (abs(b["conditional_q_mean"] - 0.5), b["bit"])
    )[:8]
    selected = [(row["bit"], sites[row["bit"]]) for row in q]
    values = bit_responses(spec, xs, max(0.05, state["temperature"]), selected)
    observation = {
        str(k): [round(float(a), 4) for a in v.mean(axis=0)] for k, v in values.items()
    }
    prefix = control_prompt(spec, state, remaining).removesuffix("\nAction:")
    return (
        prefix
        + "\nMean conditional q by bit, form order "
        + json.dumps(ENERGY_FORMS)
        + ": "
        + json.dumps(observation)
        + "\nAction:"
    )


def training_controls(spec, cid, cfg, temperatures=(0.125, 0.25, 0.5, 1.0)):
    if not encoding(spec)[0]:
        return []
    rows = []
    for variant, temperature in enumerate(temperatures):
        state = search.initial(spec, seed(cid, variant, "init"), cfg["chains"])
        state["temperature"] = temperature
        for index in range(cfg["decisions"]):
            state = search.advance(
                spec,
                state,
                "cool",
                seed(cid, variant, index, "warm"),
                cfg["window_steps"],
            )
            observed = (
                search.collapse_training_state(spec, state)
                if variant and index % 2
                else state
            )
            remaining = cfg["decisions"] - index
            prompt = extended_prompt(spec, observed, remaining)
            options = (
                search.ACTIONS
                if search.intervention(spec, observed)
                else search.ACTIONS[:2]
            )
            gains = {a: [] for a in options[1:]}
            for k in range(3):
                stream = seed(cid, variant, index, k, "return")
                cool = search.outcome(
                    spec, observed, "cool", stream, remaining, cfg["window_steps"]
                )
                for action in gains:
                    delta = (
                        search.outcome(
                            spec,
                            observed,
                            action,
                            stream,
                            remaining,
                            cfg["window_steps"],
                        )
                        - cool
                    )
                    gains[action].append(float(delta))
            for action, delta in gains.items():
                # Ties have zero target, and contradictory rollouts move it toward zero.
                target = float(np.tanh(np.asarray(delta) / 0.04).mean())
                rows.append(
                    dict(prompt=prompt, action=action, target=target, gains=delta)
                )
    return rows


def clamp_examples(case, gold):
    """Both child prompts have a verified witness and a verified negative."""
    population = sample(
        gold,
        3600,
        seed(case["id"], "augmentation"),
        chains=32,
        temperature=0.8,
        read_q=False,
        max_steps=512,
    )
    states = np.unique(
        [*population["states"], *population["final"], population["x"]], axis=0
    )
    feasible = states[violations(gold, states).sum(axis=1) == 0]
    if len(feasible) < 2:
        return []
    bounds = [(v["lower"], v["upper"]) for v in gold["variables"]]
    eligible = [
        i
        for i, (lo, hi) in enumerate(bounds)
        if lo != hi and lo in feasible[:, i] and hi in feasible[:, i]
    ]
    if not eligible or len(gold["logic"]) + len(gold["constraints"]) >= 512:
        return []
    index = eligible[seed(case["id"], "augmentation_site") % len(eligible)]
    children = []
    for value, opposite in (bounds[index], bounds[index][::-1]):
        spec = deepcopy(gold)
        name = case["variable_names"][index]
        coeff = [int(i == index) for i in range(len(bounds))]
        used = {c["id"] for c in [*gold["constraints"], *gold["logic"]]}
        constraint_id = next(
            f"pbit_clamp_{i}" for i in range(513) if f"pbit_clamp_{i}" not in used
        )
        spec["constraints"].append(dict(id=constraint_id, a=coeff, op="eq", rhs=value))
        validate(spec)
        positive = feasible[feasible[:, index] == value][0].astype(int).tolist()
        negative = feasible[feasible[:, index] == opposite][0].astype(int).tolist()
        if violations(spec, positive).sum() or not violations(spec, negative).sum():
            raise ValueError("Invalid clamp augmentation witness")
        child = dict(
            case,
            id=f'{case["id"]}__clamp_{index}_{value}',
            question=case["question"]
            + (
                f"\nFor this variant, also require the decision variable {name} "
                f"in the specified output order to equal {value}. "
                "Keep the original objective and all original constraints."
            ),
        )
        card = dict(
            kind="feasibility",
            x=negative,
            correct_feasible=False,
            candidate_feasible=True,
        )
        children.append(
            dict(
                case=child,
                gold=spec,
                candidate=gold,
                cards=[card],
                witness=positive,
                source=case["id"],
            )
        )
    return children
