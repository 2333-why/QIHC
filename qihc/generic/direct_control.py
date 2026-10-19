"""No-p-bit post-training control; gold/CP-SAT are training-only teachers."""
import json
import time
from .benchmark import reference_solve, parse_object
from .energy import objective, violations
from .compositional import training_rows


def direct_prompt(description):
    return ('Solve this optimization request. Return JSON only: {"x":[0,1,...]}, '
            'one binary decision per original variable in index order. Satisfy every rule '
            'and optimize the stated objective.\n' + description)


def parse_solution(text, n):
    obj = parse_object(text)
    if set(obj) != {"x"} or not isinstance(obj["x"], list) or len(obj["x"]) != n:
        raise ValueError("Expected exactly n binary decisions")
    if any(type(v) is not int or v not in (0, 1) for v in obj["x"]):
        raise ValueError("Decisions must be integer 0 or 1")
    return obj["x"]


def matched_data(cases, seconds=10):
    """Match per-case SFT/pair counts, not token or FLOP budgets. Never call p-bit.

    Direct SFT repairs are single-bit errors; preferences use true feasibility
    then objective. Only strictly worse candidates are accepted. Both arms lose
    a case when no feasible teacher or insufficient preference evidence exists.
    """
    start = time.perf_counter()
    modeling_sft, modeling_dpo = training_rows(cases)
    out = {k: [] for k in ("modeling_sft", "modeling_dpo", "direct_sft", "direct_dpo", "teacher_audit")}
    dump = lambda x: json.dumps({"x": x}, separators=(",", ":"))
    for case in cases:
        if case["split"] != "train":
            continue
        spec, cid = case["gold"], case["id"]
        ref = reference_solve(spec, seconds)
        audit = {"case_id": cid, "teacher": "cp_sat", **ref, "included": False}
        out["teacher_audit"].append(audit)
        x = ref["x"]
        if x is None or not (violations(spec, x) == 0).all():
            continue
        sign = 1 if spec["objective"]["sense"] == "min" else -1
        bads = []
        for i in range(spec["n"]):
            bad = list(x)
            bad[i] ^= 1
            if not (violations(spec, bad) == 0).all() or sign * objective(spec, bad) > sign * objective(spec, x):
                bads.append(bad)
        ms = [r for r in modeling_sft if r["case_id"] == cid]
        md = [r for r in modeling_dpo if r["case_id"] == cid]
        count = min(len(md), len(bads))
        if count == 0 or len(bads) < len(ms) - 1:
            continue
        audit.update(included=True, matched_pairs=count, matched_sft=len(ms))
        out["modeling_sft"].extend(ms)
        out["modeling_dpo"].extend(md[:count])
        p, chosen = direct_prompt(case["description"]), dump(x)
        out["direct_sft"].append({"case_id": cid, "task": "direct_solution", "prompt": p, "completion": chosen})
        for bad in bads[:len(ms) - 1]:
            out["direct_sft"].append({"case_id": cid, "task": "direct_repair", "prompt": p +
                '\nCorrect this candidate and return the complete solution:\n' + dump(bad), "completion": chosen})
        for bad in bads[:count]:
            out["direct_dpo"].append({"case_id": cid, "task": "direct_preference", "prompt": p,
                "chosen": chosen, "rejected": dump(bad)})
    out["budget"] = {"matching": "per-case SFT rows and DPO pairs only; NOT tokens/FLOPs",
                     "preparation_seconds": time.perf_counter() - start,
                     "pbit_teacher_calls": 0,
                     "training_cases": sum(r["included"] for r in out["teacher_audit"])}
    return out


def evaluate_direct(cases, predictions):
    selected = {c["id"]: c for c in cases if c["split"] != "train"}
    ids = [r["case_id"] for r in predictions]
    if len(ids) != len(set(ids)) or set(ids) != set(selected):
        raise ValueError("Predictions must cover every evaluation case exactly once")
    rows = []
    for r in predictions:
        case = selected[r["case_id"]]
        row = {"case_id": case["id"], "split": case["split"], "size": case["size"], "feasible": False}
        try:
            x = parse_solution(r["response"], case["gold"]["n"])
            row.update(valid=True, x=x, feasible=bool((violations(case["gold"], x) == 0).all()),
                       objective=float(objective(case["gold"], x)))
        except (ValueError, TypeError, KeyError) as exc:
            row.update(valid=False, error=str(exc))
        rows.append(row)
    return rows
