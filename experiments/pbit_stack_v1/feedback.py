"""Train local repairs on verified witnesses and apply them with source review."""

from copy import deepcopy
import json

import numpy as np

from experiments.boundary_feedback_v1 import policy
from experiments.pbit_dual_v1.methods import probe
from experiments.pvcd_cycle_v1.evidence import cards_from_states
from qihc.experiment_runtime.pbit_control import seed
from qihc.generic.bounded import validate, violations
from qihc.generic.feedback.search import Energy, sample
from qihc.modeling.active_protocol_v2 import compact, model_prompt, parse_model

BLOCKS = ("objective", "constraints", "logic")


def sample_states(model, seed_value, count=16):
    row = sample(
        model, 3600, seed_value, chains=16, temperature=0.6, read_q=False, max_steps=128
    )
    return list(dict.fromkeys(map(tuple, [row["x"], *row["final"], *row["states"]])))[
        :count
    ]


def objective_example(case, gold):
    """Source-backed objective perturbation only when an order change is verified."""
    x = [list(v) for v in sample_states(gold, seed(case["id"], "cost"), 32)]
    x = np.asarray(x)
    x = x[violations(gold, x).sum(axis=1) == 0]
    if len(x) < 2:
        return None
    for i in np.flatnonzero(np.ptp(x, axis=0)):
        i = int(i)
        for delta in (4, -4, 12, -12):
            new = deepcopy(gold)
            new["objective"]["linear"][i] += delta
            validate(new)
            cards = cards_from_states(gold, new, x, limit=8)
            if cards:
                name = case["variable_names"][i]
                child = dict(
                    case,
                    id=f'{case["id"]}__cost_{i}_{delta}',
                    question=case["question"]
                    + f"\nFor this variant add ({delta}) times "
                    f"the decision variable {name} to the objective expression. "
                    "Keep the original optimization direction and constraints.",
                )
                return dict(case=child, gold=new, candidate=gold, cards=cards)
    return None


def repair_prompt(case, draft, states, error=None):
    prompt = model_prompt(case)
    if draft is None:
        return (
            prompt
            + "\nRepair the invalid draft. Return a complete model.\n"
            + (error or "")[:500]
        )
    _, residuals, _ = Energy(draft).values(np.asarray(states))
    names = [r["id"] for key in ("constraints", "logic") for r in draft[key]]
    signatures = [
        {
            names[i]: round(float(r[i]), 6)
            for i in np.argsort(-r, kind="stable")[:4]
            if r[i] > 0
        }
        for r in residuals
    ]
    diagnostic = dict(
        model=compact(draft), assignments=states, draft_residuals=signatures
    )
    return (
        prompt
        + "\nAudit the draft against ORIGINAL. The assignments are p-bit probe states; "
        "draft_residuals lists the largest violated draft constraints for each state. "
        "Evaluate the relevant requirements against ORIGINAL. Locate a "
        "specific modeling defect. Return edits to only the objective, constraints, or logic "
        'blocks. Use {"edits":[]} if the evidence does not support a change. '
        "Never change variable order or domains.\n"
        + json.dumps(diagnostic, separators=(",", ":"))
    )


def repair_example(row):
    old, gold = row["candidate"], row["gold"]
    if old is None:
        case = row["case"]
        return dict(
            case_id=case["id"],
            prompt=repair_prompt(case, None, [], "Incomplete model"),
            target=json.dumps(
                {"model": compact(gold)}, ensure_ascii=False, separators=(",", ":")
            ),
        )
    if old["variables"] != gold["variables"] or not row["cards"]:
        return None
    old_compact, gold_compact = compact(old), compact(gold)
    changes = [
        dict(target=k, value=gold_compact[k])
        for k in BLOCKS
        if old_compact[k] != gold_compact[k]
    ]
    if not changes:
        return None
    case = row["case"]
    states = [list(x) for x in sample_states(old, seed(case["id"], "repair_train"))]
    text = json.dumps(dict(edits=changes), ensure_ascii=False, separators=(",", ":"))
    if parse_model(text, case, old) is None:
        raise ValueError("Invalid certified repair")
    return dict(
        case_id=case["id"], prompt=repair_prompt(case, old, states), target=text
    )


def propose(case, old, ask):
    model = old["model"]
    states = (
        []
        if model is None
        else [list(x) for x in sample_states(model, seed(case["id"], "repair_test"))]
    )
    text = ask(
        repair_prompt(case, model, states, old.get("error")), seed(case["id"], "patch")
    )
    try:
        obj = json.loads(text)
        if model is not None and (
            set(obj) != {"edits"}
            or any(edit["target"] not in BLOCKS for edit in obj["edits"])
        ):
            raise ValueError("Expected a local patch")
        candidate = parse_model(text, case, model)
        if model is not None and candidate["variables"] != model["variables"]:
            raise ValueError("Variable domains changed")
    except (ValueError, KeyError, TypeError, IndexError, OverflowError):
        return dict(model=model, action="patch_parse_error")
    return dict(
        model=candidate, action="candidate_ready" if candidate != model else "keep"
    )


def verify_patch(case, base, proposal, ask):
    old, candidate = base["model"], proposal["model"]
    if candidate is None or candidate == old:
        return dict(model=old, action=proposal["action"])
    cid = case["id"]
    if old is None:
        text = ask(
            model_prompt(case)
            + "\nCheck whether this proposed model follows ORIGINAL. "
            'Return only {"accept":true} or {"accept":false}.\n'
            + json.dumps(compact(candidate)),
            seed(cid, "compile_review"),
        )
        try:
            accepted = json.loads(text) == {"accept": True}
        except ValueError:
            accepted = False
        return dict(
            model=candidate if accepted else old,
            action="accepted_compile" if accepted else "compile_rejected",
        )
    xs, telemetry = probe((old, candidate), seed(cid, "patch_probe"))
    assignments, pairs = policy.witnesses((old, candidate), xs)
    base_f, base_obj = policy.signatures(old, assignments) if assignments else ([], [])
    new_f, new_obj = (
        policy.signatures(candidate, assignments) if assignments else ([], [])
    )
    decisive = any(a != b for a, b in zip(base_f, new_f)) or any(
        all((base_f[i], base_f[j], new_f[i], new_f[j]))
        and np.sign(base_obj[i] - base_obj[j]) != np.sign(new_obj[i] - new_obj[j])
        for i, j in pairs
    )
    if not decisive:
        return dict(model=old, action="no_discriminating_witness", probe=telemetry)
    verdict = policy.assess(
        ask(policy.judge_prompt(case, assignments, pairs), seed(cid, "patch_judge")),
        case,
        (old, candidate),
        assignments,
        pairs,
    )
    passed = (
        verdict["status"] == "judged"
        and verdict["covered"] >= 2
        and verdict["errors"][0] > 0
        and verdict["errors"][1] == 0
    )
    if not passed:
        return dict(
            model=old,
            action=(
                verdict["status"]
                if verdict["status"] != "judged"
                else "insufficient_support"
            ),
            probe=telemetry,
            verdict=verdict,
        )
    # An independent presentation reviews the candidate against the source.
    review = policy.decide(
        dict(case=case, seed=17),
        "C",
        dict(model=old, candidate=candidate, action="candidate_ready"),
        ask,
    )
    return dict(review, probe=telemetry, verdict=verdict)
