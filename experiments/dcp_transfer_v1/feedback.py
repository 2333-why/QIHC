"""Train and decode the same local edit contract; retry a malformed response once."""

import json
from copy import deepcopy

from experiments.pbit_stack_v1.feedback import BLOCKS, repair_prompt, sample_states
from experiments.pvcd_cycle_v1.evidence import cards_from_states
from experiments.posttraining_v1.data import target
from qihc.experiment_runtime.pbit_control import seed
from qihc.generic.feedback.search import sample
from qihc.modeling.active_protocol_v2 import compact, parse_model
from qihc.modeling.protocol import load_object

PATCH = '\nOUTPUT CONTRACT: Return only {"edits":[{"target":"objective","value":REPLACEMENT_BLOCK}]} or analogous constraints/logic edits. Each target occurs once. Keep with {"edits":[]}.'
FULL = '\nOUTPUT CONTRACT: Return only {"model":MODEL}. Supply every required field and the public variable order.'


def prompt(case, old, states, error=None, raw=""):
    return repair_prompt(case, old, states, error) + (
        PATCH
        if old is not None
        else "\nUnparseable draft excerpt:\n" + (raw or "")[:1200] + FULL
    )


def decode(text, case, old):
    obj = load_object(text)
    if old is None:
        if set(obj) != {"model"}:
            raise ValueError("Missing complete model envelope")
    else:
        if set(obj) != {"edits"} or not isinstance(obj["edits"], list):
            raise ValueError("Expected edits list")
        keys = [e["target"] for e in obj["edits"]]
        if any(k not in BLOCKS for k in keys) or len(set(keys)) != len(keys):
            raise ValueError(
                "Only unique objective, constraints or logic replacements allowed"
            )
    result = parse_model(text, case, old)
    if old is not None and old["variables"] != result["variables"]:
        raise ValueError("Public variable domains changed")
    return result


def propose(case, base, ask):
    old = base["model"]
    states, sampling_seconds, sampling_attempts = [], 0, 0
    if old is not None:
        probe = sample(
            old,
            3600,
            seed(case["id"], "repair_test"),
            chains=16,
            temperature=0.6,
            read_q=False,
            max_steps=128,
        )
        assignments = [probe["x"], *probe["final"], *probe["states"]]
        states = [list(x) for x in list(dict.fromkeys(map(tuple, assignments)))[:16]]
        sampling_seconds = probe["seconds"]
        sampling_attempts = probe["transition_attempts"]
    initial = prompt(case, old, states, base.get("error"), base.get("raw"))
    request = initial
    for attempt in range(2):
        text = ask(request, seed(case["id"], "schema_patch", attempt))
        try:
            candidate = decode(text, case, old)
            return dict(
                model=candidate,
                action="keep" if candidate == old else "candidate_ready",
                retry=bool(attempt),
                parse_failed=False,
                sampling_attempts=sampling_attempts,
                sampling_seconds=sampling_seconds,
            )
        except (ValueError, KeyError, TypeError, IndexError, OverflowError) as exc:
            request = (
                initial
                + "\nParser rejected the previous response: "
                + str(exc)[:200]
                + "\nPrevious response excerpt:\n"
                + text[:900]
                + "\nRe-emit valid JSON."
            )
    return dict(
        model=old,
        action="patch_parse_error",
        retry=True,
        parse_failed=True,
        sampling_attempts=sampling_attempts,
        sampling_seconds=sampling_seconds,
    )


def supervise(case, gold, witness, index):
    # Source-labeled defect, no-op and compile examples. Weight only observed differences.
    old = deepcopy(gold)
    if index % 2 or not old["constraints"]:
        old["objective"]["sense"] = (
            "max" if old["objective"]["sense"] == "min" else "min"
        )
        block = "objective"
    else:
        del old["constraints"][index % len(old["constraints"])]
        block = "constraints"
    states = [list(x) for x in sample_states(old, seed(case["id"], "train_patch"))]
    cards = cards_from_states(old, gold, states + [witness], limit=8)
    modeling = dict(case=case, gold=gold, candidate=old, cards=cards)
    if index % 3 == 0:
        baseline, xs = None, []
        request = prompt(case, None, [], "JSON ended early", '{"model":')
        answer = target(gold)
    elif cards and index % 3 == 1:
        baseline, xs = old, states
        request = prompt(case, old, xs)
        answer = json.dumps(
            dict(edits=[dict(target=block, value=compact(gold)[block])]),
            separators=(",", ":"),
        )
    else:
        baseline = gold
        xs = [list(x) for x in sample_states(gold, seed(case["id"], "train_keep"))]
        request, answer = prompt(case, gold, xs), '{"edits":[]}'
    if decode(answer, case, baseline) != gold:
        raise ValueError("Training response failed production parser round trip")
    return modeling, dict(case_id=case["id"], prompt=request, target=answer)
