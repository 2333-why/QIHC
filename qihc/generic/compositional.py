"""Synthetic constraint-composition benchmark, NOT evidence of arbitrary NL coverage.

Single rule types are seen in training. Their three-way combinations and larger
sizes are held out. A hidden feasible assignment constructs bounds, never prompts.
"""
from __future__ import annotations

import json
import random
from .energy import validate
from .sparse import encode, prompt
from .semantic import mutations, verified_preference

RULES = ("budget", "quota", "implication", "exclusion", "balance")


def make_case(n, seed, rules, split):
    rng = random.Random(seed)
    hidden = [rng.randrange(2) for _ in range(n)]
    linear = [rng.randint(-9, 9) for _ in range(n)]
    quadratic = [[i, i + 1, rng.choice([-5, -2, 2, 5])] for i in range(0, n - 1, 3)]
    sense = rng.choice(["min", "max"])
    text = [f"There are {n} optional activities indexed 0 through {n-1}. x_i is 1 if activity i is chosen.",
            f"{'Minimize total cost' if sense == 'min' else 'Maximize total benefit'}. Individual contributions in index order: {linear}.",
            f"Additional contributions apply only when both activities are selected (i,j,contribution): {quadratic}."]
    cons = []
    for kind in rules:
        a = [0] * n
        subset = sorted(rng.sample(range(n), min(n, 5)))
        if kind == "budget":
            a = [rng.randint(1, 6) for _ in range(n)]
            rhs, op = sum(v * x for v, x in zip(a, hidden)) + 2, "le"
            sentence = f"Resource usage per activity is {a}; total use must not exceed {rhs}."
        elif kind in ("quota", "balance"):
            for i in subset:
                a[i] = 1
            rhs = sum(hidden[i] for i in subset)
            op = "ge" if kind == "quota" else "eq"
            sentence = f"Select {'at least' if op == 'ge' else 'exactly'} {rhs} activities from {subset}."
        elif kind == "implication":
            i, j = rng.sample(range(n), 2)
            if hidden[i] > hidden[j]:
                i, j = j, i
            a[i], a[j], rhs, op = 1, -1, 0, "le"
            sentence = f"Activity {i} may be chosen only if activity {j} is chosen."
        elif kind == "exclusion":
            pairs = [(i, j) for i in range(n) for j in range(i + 1, n) if hidden[i] + hidden[j] <= 1]
            if not pairs:
                # Still a valid explicit cardinality rule, without modifying hidden state.
                i, j, rhs = 0, 1, 2
            else:
                i, j = rng.choice(pairs)
                rhs = 1
            a[i] = a[j] = 1
            op = "le"
            sentence = f"Among activities {i} and {j}, at most {rhs} may be chosen."
        else:
            raise ValueError(kind)
        cons.append({"id": f"c{len(cons)}", "a": a, "op": op, "rhs": rhs})
        text.append(sentence)
    rule_text = text[3:]
    rng.shuffle(rule_text)
    text[3:] = rule_text
    spec = validate({"n": n, "objective": {"sense": sense, "linear": linear, "quadratic": quadratic}, "constraints": cons})
    return {"id": f"composition-{seed}", "family": "composition", "split": split,
            "rules": list(rules), "size": n, "description": "\n".join(text), "gold": spec}


def dataset(count=120, seed=20261019):
    if count < 20:
        raise ValueError("Use at least 20 cases per split")
    rows = []
    for si, split in enumerate(("train", "validation", "test")):
        for i in range(count):
            # Train/validation use individual primitives; test uses novel triples.
            rules = (RULES[i % len(RULES)],) if split != "test" else tuple(RULES[(i + j) % len(RULES)] for j in range(3))
            n = (12, 20, 32)[i % 3] if split != "test" else (20, 32, 48)[i % 3]
            rows.append(make_case(n, seed + si * count + i, rules, split))
    return rows


def training_rows(cases):
    """Modeling SFT plus witness-checked repair SFT and same-policy semantic DPO.

    No strategy labels: isolates modeling from search-policy optimization.
    Counterexamples are audit-only; never shown to the inference model.
    """
    sft, dpo = [], []
    dump = lambda s: json.dumps(encode(s), separators=(",", ":"))
    for case in cases:
        if case["split"] != "train":
            continue
        p, chosen = prompt(case["description"]), dump(case["gold"])
        sft.append({"case_id": case["id"], "task": "modeling", "prompt": p, "completion": chosen})
        repairs = 0
        for kind, bad in mutations(case["gold"]):
            verified = verified_preference(case, bad, "adaptive", kind)
            if verified is None:
                continue
            rejected = dump(bad)
            dpo.append({"case_id": case["id"], "task": "semantic_modeling", "prompt": p,
                        "chosen": chosen, "rejected": rejected, "error_type": kind,
                        "witness": verified["witness"]})
            if repairs < 2 and kind in ("quadratic_sign", "constraint_bound"):
                sft.append({"case_id": case["id"], "task": "repair", "prompt": p +
                            "\nAudit this candidate against the request and return a corrected full model:\n" + rejected,
                            "completion": chosen})
                repairs += 1
    return sft, dpo
